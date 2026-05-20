"""Assemble retrieved-context bundles from the WikiGraphRAG index.

This module sits between the raw graph/lexical layers and the higher-level
answer service. Given a query plus an :class:`WikiGraphIndex`, it returns a
list of :class:`WikiGraphRetrievedContext` items with provenance traces.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
from rapidfuzz import fuzz

from graphwiki_kb.wikigraph.graph_store import (
    WikiGraphStore,
    collect_neighbors,
    node_pagerank,
)
from graphwiki_kb.wikigraph.lexical_index import (
    LexicalDocument,
    LexicalIndex,
)
from graphwiki_kb.wikigraph.models import (
    WikiGraphIndex,
    WikiGraphNode,
    WikiGraphRetrievedContext,
)


@dataclass
class ContextBuilderConfig:
    """Tunable knobs for context assembly."""

    max_context_chunks: int = 8
    max_context_tokens: int = 6000
    max_hops: int = 2
    fuzzy_entity_match_threshold: int = 82


class WikiGraphContextBuilder:
    """Builds retrieved contexts for various WikiGraphRAG query methods."""

    def __init__(
        self,
        index: WikiGraphIndex,
        *,
        config: ContextBuilderConfig | None = None,
    ) -> None:
        self.index = index
        self.config = config or ContextBuilderConfig()
        self._nodes_by_id: dict[str, WikiGraphNode] = {
            node.id: node for node in index.nodes
        }
        self._graph: nx.MultiGraph = WikiGraphStore.to_networkx(index)
        self._lexical = self._build_lexical_index()
        self._pagerank = node_pagerank(self._graph)

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def basic_search(
        self, question: str, *, limit: int | None = None
    ) -> list[WikiGraphRetrievedContext]:
        """BM25-style retrieval over chunk text."""
        if limit is None:
            limit = self.config.max_context_chunks
        hits = self._lexical.search(question, limit=limit * 2)
        contexts = self._hits_to_contexts(hits, base_trace=["basic"])
        return contexts[:limit]

    def local_search(
        self, question: str, *, limit: int | None = None
    ) -> tuple[list[WikiGraphRetrievedContext], list[str]]:
        """Entity-centered retrieval with 1-2 hop expansion."""
        if limit is None:
            limit = self.config.max_context_chunks
        seed_entities = self._match_entities(question)
        if not seed_entities:
            return self.basic_search(question, limit=limit), []
        expanded_chunks: dict[str, tuple[float, list[str]]] = {}
        seed_titles: list[str] = []
        for entity_node in seed_entities:
            seed_titles.append(entity_node.title)
            neighbors = collect_neighbors(
                self._graph,
                entity_node.id,
                max_hops=self.config.max_hops,
            )
            for neighbor_id, distance, edge_kind in neighbors:
                neighbor = self._nodes_by_id.get(neighbor_id)
                if neighbor is None:
                    continue
                base_weight = 1.0 / float(distance + 1)
                pagerank_boost = float(self._pagerank.get(neighbor_id, 0.0)) * 5
                score = base_weight + pagerank_boost
                if neighbor.kind == "chunk":
                    _record_score(
                        expanded_chunks,
                        neighbor_id,
                        score,
                        [f"local:{entity_node.title}->{edge_kind}({distance})"],
                    )
                elif neighbor.kind in {"source_page", "concept_page", "analysis_page"}:
                    for chunk_id in self._chunks_for_page(neighbor_id):
                        _record_score(
                            expanded_chunks,
                            chunk_id,
                            score * 0.75,
                            [f"local:{entity_node.title}->page:{neighbor.title}"],
                        )
                elif neighbor.kind == "claim":
                    _record_score(
                        expanded_chunks,
                        neighbor_id,
                        score * 0.9,
                        [f"local:{entity_node.title}->claim"],
                    )
        lex_hits = self._lexical.search(question, limit=limit * 2)
        for hit in lex_hits:
            _record_score(
                expanded_chunks, hit.doc_id, hit.score * 0.5, ["local:bm25-boost"]
            )
        ranked = sorted(
            expanded_chunks.items(), key=lambda item: item[1][0], reverse=True
        )
        contexts: list[WikiGraphRetrievedContext] = []
        for node_id, (score, trace) in ranked:
            node = self._nodes_by_id.get(node_id)
            if node is None:
                continue
            if node.kind not in {"chunk", "claim"}:
                continue
            contexts.append(self._node_to_context(node, score=score, trace=trace))
            if len(contexts) >= limit:
                break
        return contexts, seed_titles

    def global_search(
        self, question: str, *, limit: int | None = None
    ) -> tuple[list[WikiGraphRetrievedContext], list[str]]:
        """Community-summary retrieval with map-reduce-ish reduction."""
        if limit is None:
            limit = self.config.max_context_chunks
        if not self.index.communities:
            return self.basic_search(question, limit=limit), []
        scored_communities: list[tuple[float, str]] = []
        for community in self.index.communities:
            summary_score = self._lexical_score_for_text(question, community.summary)
            entity_score = sum(
                self._lexical_score_for_text(question, name)
                for name in community.top_entities
            )
            scored_communities.append(
                (summary_score + 0.5 * entity_score, community.id)
            )
        scored_communities.sort(reverse=True)
        selected_ids = [cid for score, cid in scored_communities[:5] if score > 0]
        if not selected_ids and scored_communities:
            selected_ids = [scored_communities[0][1]]

        global_contexts: list[WikiGraphRetrievedContext] = []
        communities_by_id = {c.id: c for c in self.index.communities}
        for community_id in selected_ids:
            selected_community = communities_by_id.get(community_id)
            if selected_community is None:
                continue
            global_contexts.append(
                WikiGraphRetrievedContext(
                    node_id=selected_community.id,
                    node_kind="community",
                    title=selected_community.title,
                    path=None,
                    text=selected_community.summary,
                    score=1.0,
                    source_ids=list(selected_community.source_ids),
                    trace=[f"global:community={selected_community.id}"],
                )
            )
            member_chunks = [
                self._nodes_by_id[m]
                for m in selected_community.members
                if m in self._nodes_by_id
                and self._nodes_by_id[m].kind in {"chunk", "claim"}
            ]
            for chunk_node in member_chunks[
                : max(1, limit // max(1, len(selected_ids)))
            ]:
                global_contexts.append(
                    self._node_to_context(
                        chunk_node,
                        score=0.75,
                        trace=[f"global:community={selected_community.id}->member"],
                    )
                )
            if len(global_contexts) >= limit:
                break
        return global_contexts[:limit], selected_ids

    def drift_lite(
        self, question: str, *, limit: int | None = None
    ) -> tuple[list[WikiGraphRetrievedContext], list[str], list[str]]:
        """Local search expanded by deterministic sub-questions."""
        if limit is None:
            limit = self.config.max_context_chunks
        local_contexts, seed_entities = self.local_search(
            question, limit=max(limit // 2, 3)
        )
        sub_questions = self._derive_sub_questions(question, seed_entities)
        combined: dict[str, WikiGraphRetrievedContext] = {
            ctx.node_id: ctx for ctx in local_contexts
        }
        for sub_question in sub_questions:
            sub_contexts, _ = self.local_search(sub_question, limit=3)
            for ctx in sub_contexts:
                if ctx.node_id in combined:
                    continue
                combined[ctx.node_id] = ctx.model_copy(
                    update={"trace": [*ctx.trace, f"drift:sub={sub_question}"]}
                )
                if len(combined) >= limit:
                    break
            if len(combined) >= limit:
                break
        return list(combined.values())[:limit], seed_entities, sub_questions

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _build_lexical_index(self) -> LexicalIndex:
        index = LexicalIndex()
        for node in self.index.nodes:
            if node.kind not in {"chunk", "claim"}:
                continue
            text = node.text or node.title
            index.add(
                LexicalDocument(
                    doc_id=node.id,
                    text=text,
                    metadata={
                        "kind": node.kind,
                        "title": node.title,
                        "path": node.path,
                    },
                )
            )
        index.fit()
        return index

    def _lexical_score_for_text(self, question: str, text: str) -> float:
        if not text:
            return 0.0
        from graphwiki_kb.wikigraph.lexical_index import tokenize

        question_tokens = set(tokenize(question))
        if not question_tokens:
            return 0.0
        text_tokens = tokenize(text)
        if not text_tokens:
            return 0.0
        hits = sum(1 for token in text_tokens if token in question_tokens)
        if hits == 0:
            return 0.0
        return hits / (1 + len(text_tokens) ** 0.5)

    def _hits_to_contexts(
        self, hits, base_trace: list[str]
    ) -> list[WikiGraphRetrievedContext]:
        contexts: list[WikiGraphRetrievedContext] = []
        for hit in hits:
            node = self._nodes_by_id.get(hit.doc_id)
            if node is None:
                continue
            contexts.append(
                self._node_to_context(node, score=hit.score, trace=base_trace)
            )
        return contexts

    def _node_to_context(
        self,
        node: WikiGraphNode,
        *,
        score: float,
        trace: list[str],
    ) -> WikiGraphRetrievedContext:
        chunk_index = node.metadata.get("chunk_index") if node.metadata else None
        section = ""
        if node.metadata and isinstance(node.metadata.get("section"), str):
            section = str(node.metadata["section"])
        return WikiGraphRetrievedContext(
            node_id=node.id,
            node_kind=node.kind,
            title=node.title,
            path=node.path,
            text=node.text,
            score=float(score),
            source_ids=list(node.source_ids),
            section=section,
            chunk_index=int(chunk_index) if isinstance(chunk_index, int) else None,
            trace=list(trace),
        )

    def _chunks_for_page(self, page_id: str) -> list[str]:
        chunks: list[str] = []
        for neighbor in self._graph.neighbors(page_id):
            node = self._nodes_by_id.get(neighbor)
            if node is None or node.kind != "chunk":
                continue
            chunks.append(neighbor)
        return chunks

    def _match_entities(self, question: str) -> list[WikiGraphNode]:
        matches: dict[str, tuple[float, WikiGraphNode]] = {}
        lowered = question.lower()
        for node in self.index.nodes:
            if node.kind != "entity":
                continue
            ratio = max(
                fuzz.token_set_ratio(node.title, question),
                *(fuzz.token_set_ratio(alias, question) for alias in node.aliases),
                int(node.title.lower() in lowered) * 100,
                int(any(alias.lower() in lowered for alias in node.aliases)) * 100,
            )
            if ratio < self.config.fuzzy_entity_match_threshold:
                continue
            matches[node.id] = (float(ratio), node)
        return [
            node
            for _, node in sorted(
                matches.values(), key=lambda item: item[0], reverse=True
            )
        ][:6]

    def _derive_sub_questions(
        self,
        question: str,
        seed_entities: list[str],
    ) -> list[str]:
        sub_questions: list[str] = []
        if not seed_entities:
            return sub_questions
        templates = [
            "What does the corpus say about {entity}?",
            "How does {entity} relate to the other key topics in this question?",
        ]
        for entity in seed_entities[:2]:
            for template in templates:
                sub = template.format(entity=entity)
                if sub.lower() != question.lower() and sub not in sub_questions:
                    sub_questions.append(sub)
                if len(sub_questions) >= 4:
                    break
            if len(sub_questions) >= 4:
                break
        return sub_questions


def _record_score(
    scores: dict[str, tuple[float, list[str]]],
    node_id: str,
    score: float,
    trace: list[str],
) -> None:
    existing = scores.get(node_id)
    if existing is None:
        scores[node_id] = (score, list(trace))
        return
    current_score, current_trace = existing
    merged_trace = list(current_trace)
    for entry in trace:
        if entry not in merged_trace:
            merged_trace.append(entry)
    scores[node_id] = (current_score + score, merged_trace)


def merge_contexts(
    *bundles: list[WikiGraphRetrievedContext],
    limit: int,
) -> list[WikiGraphRetrievedContext]:
    """Merge several context bundles using reciprocal rank fusion."""
    scores: dict[str, float] = defaultdict(float)
    storage: dict[str, WikiGraphRetrievedContext] = {}
    for bundle in bundles:
        for rank, context in enumerate(bundle, start=1):
            scores[context.node_id] += 1 / (60 + rank)
            current = storage.get(context.node_id)
            if current is None or context.score > current.score:
                storage[context.node_id] = context
    ordered = sorted(
        storage.values(),
        key=lambda ctx: scores[ctx.node_id],
        reverse=True,
    )
    return ordered[:limit]
