"""Assemble retrieved-context bundles from the WikiGraphRAG index.

This module sits between the raw graph/lexical layers and the higher-level
answer service. Given a query plus an :class:`WikiGraphIndex`, it returns a
list of :class:`WikiGraphRetrievedContext` items with provenance traces.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rapidfuzz import fuzz

from graphwiki_kb.wikigraph.graph_store import (
    WikiGraphStore,
    collect_neighbors,
    node_pagerank,
)

if TYPE_CHECKING:
    import networkx as nx
from graphwiki_kb.wikigraph.lexical_index import (
    LexicalDocument,
    LexicalIndex,
)
from graphwiki_kb.wikigraph.lexical_index import (
    tokenize as _tokenize_for_boost,
)
from graphwiki_kb.wikigraph.models import (
    EVIDENCE_NODE_KINDS,
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
    lexical_backend: str = "bm25s"
    # Phase 4 — retrieval improvements (enabled by default; flip off
    # for the baseline ablation row by setting
    # ``retrieval_improvements_enabled=False``).
    retrieval_improvements_enabled: bool = True
    # RRF constant: standard Cormack et al. 2009 value.
    rrf_k: int = 60
    # Cap on alias tokens appended to a BM25 query during alias-aware
    # query expansion; tuned to keep BM25 from overweighting alias spam.
    alias_query_token_budget: int = 16
    # Per-hit score nudge when at least one query token also appears in
    # the chunk's section title. Kept small (additive) so it cannot
    # upend a clearly-better-ranked chunk.
    section_title_overlap_boost: float = 0.10


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
        """BM25 retrieval over wiki chunks, claims, and source TextUnits.

        TextUnits get a small ranking nudge so paper-body evidence
        consistently shows up alongside the LLM-summarized wiki chunks
        when the question is body-content-heavy. The nudge is small
        (15%) so it cannot upend a clearly-better wiki chunk.

        Phase 4: when ``retrieval_improvements_enabled`` is true, chunks
        whose ``section`` title shares one or more non-stopword tokens
        with the question receive a small additive boost
        (``section_title_overlap_boost``). Helps surface section-anchor
        evidence (e.g. the "Methods" section of a paper) on
        method-specific questions.
        """
        if limit is None:
            limit = self.config.max_context_chunks
        hits = self._lexical.search(question, limit=limit * 3)
        # Promote TextUnit hits slightly; keeps the comparison fair
        # without making them dominate purely on volume.
        nudged: list = []
        question_tokens = set(_tokenize_for_boost(question))
        for hit in hits:
            node = self._nodes_by_id.get(hit.doc_id)
            if node is None:
                nudged.append(hit)
                continue
            from dataclasses import replace

            new_score = hit.score
            if node.kind == "text_unit":
                new_score *= 1.15
            if self.config.retrieval_improvements_enabled:
                new_score += self._section_boost(node, question_tokens)
            nudged.append(replace(hit, score=new_score))
        nudged.sort(key=lambda h: h.score, reverse=True)
        contexts = self._hits_to_contexts(nudged, base_trace=["basic"])
        return self._enforce_token_budget(contexts[:limit])

    def _section_boost(self, node: WikiGraphNode, question_tokens: set[str]) -> float:
        """Return an additive boost when the node's section overlaps the question."""
        if not self.config.retrieval_improvements_enabled:
            return 0.0
        if not question_tokens:
            return 0.0
        section_text = ""
        if node.metadata and isinstance(node.metadata.get("section"), str):
            section_text = node.metadata["section"]
        if not section_text:
            return 0.0
        section_tokens = set(_tokenize_for_boost(section_text))
        if not section_tokens:
            return 0.0
        if section_tokens & question_tokens:
            return float(self.config.section_title_overlap_boost)
        return 0.0

    def local_search(
        self, question: str, *, limit: int | None = None
    ) -> tuple[list[WikiGraphRetrievedContext], list[str]]:
        """Entity-centered retrieval with 1-2 hop expansion.

        Phase 4 (``retrieval_improvements_enabled``):

        * **Reciprocal-rank fusion** across three ranked lists --
          entity-hop expansion, BM25 over the bare question, and BM25
          over the question augmented with seed-entity titles+aliases.
          RRF (Cormack et al., 2009) is robust to score-scale
          differences across the three signals.
        * **Alias-aware query expansion**: BM25 query is rewritten to
          include matched entity titles and aliases. Surfaces text
          units that mention the spelled-out form when the question
          uses the acronym (and vice-versa).
        """
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
            for neighbor_id, distance, _edge_kind in neighbors:
                neighbor = self._nodes_by_id.get(neighbor_id)
                if neighbor is None:
                    continue
                base_weight = 1.0 / float(distance + 1)
                pagerank_boost = float(self._pagerank.get(neighbor_id, 0.0)) * 5
                score = base_weight + pagerank_boost
                if neighbor.kind in {"chunk", "text_unit"}:
                    label = "chunk" if neighbor.kind == "chunk" else "text_unit"
                    _record_score(
                        expanded_chunks,
                        neighbor_id,
                        score,
                        [f"local:{entity_node.title}->{label}({distance})"],
                    )
                elif neighbor.kind in {"source_page", "concept_page", "analysis_page"}:
                    for chunk_id in self._chunks_for_page(neighbor_id):
                        _record_score(
                            expanded_chunks,
                            chunk_id,
                            score * 0.75,
                            [f"local:{entity_node.title}->page:{neighbor.title}"],
                        )
                elif neighbor.kind == "source_document":
                    for unit_id in self._text_units_for_document(neighbor_id):
                        _record_score(
                            expanded_chunks,
                            unit_id,
                            score * 0.85,
                            [f"local:{entity_node.title}->document:{neighbor.title}"],
                        )
                elif neighbor.kind == "claim":
                    _record_score(
                        expanded_chunks,
                        neighbor_id,
                        score * 0.9,
                        [f"local:{entity_node.title}->claim"],
                    )

        improvements = self.config.retrieval_improvements_enabled
        if improvements:
            # Reciprocal-rank fusion across three signals.
            entity_hop_order = [
                node_id
                for node_id, _ in sorted(
                    expanded_chunks.items(), key=lambda item: item[1][0], reverse=True
                )
            ]
            bm25_hits = self._lexical.search(question, limit=limit * 3)
            bm25_order = [hit.doc_id for hit in bm25_hits]
            expanded_query = self._expand_query_with_aliases(question, seed_entities)
            alias_hits = (
                self._lexical.search(expanded_query, limit=limit * 3)
                if expanded_query != question
                else bm25_hits
            )
            alias_order = [hit.doc_id for hit in alias_hits]
            fused = _reciprocal_rank_fusion(
                [entity_hop_order, bm25_order, alias_order],
                k=self.config.rrf_k,
            )
            # Carry over entity-hop trace where present; fall back to a
            # generic RRF trace.
            ranked_ids: list[tuple[str, float, list[str]]] = []
            for node_id, fusion_score in fused:
                trace = expanded_chunks.get(node_id, (0.0, []))[1]
                if not trace:
                    trace = [
                        (
                            "local:rrf-bm25"
                            if node_id in set(bm25_order)
                            else "local:rrf-alias"
                        )
                    ]
                ranked_ids.append((node_id, fusion_score, trace))
        else:
            # Pre-Phase-4 behavior: simple weighted BM25 boost.
            lex_hits = self._lexical.search(question, limit=limit * 2)
            for hit in lex_hits:
                _record_score(
                    expanded_chunks, hit.doc_id, hit.score * 0.5, ["local:bm25-boost"]
                )
            ranked = sorted(
                expanded_chunks.items(), key=lambda item: item[1][0], reverse=True
            )
            ranked_ids = [(node_id, score, trace) for node_id, (score, trace) in ranked]

        question_tokens = set(_tokenize_for_boost(question))
        contexts: list[WikiGraphRetrievedContext] = []
        for node_id, score, trace in ranked_ids:
            node = self._nodes_by_id.get(node_id)
            if node is None:
                continue
            if node.kind not in EVIDENCE_NODE_KINDS:
                continue
            adjusted_score = score + self._section_boost(node, question_tokens)
            contexts.append(
                self._node_to_context(node, score=adjusted_score, trace=trace)
            )
            if len(contexts) >= limit:
                break
        return self._enforce_token_budget(contexts), seed_titles

    def _expand_query_with_aliases(
        self,
        question: str,
        seed_entities: list[WikiGraphNode],
    ) -> str:
        """Append entity title + aliases to ``question`` for BM25 expansion."""
        if not seed_entities:
            return question
        budget = max(0, int(self.config.alias_query_token_budget))
        if budget <= 0:
            return question
        alias_terms: list[str] = []
        for entity in seed_entities:
            for candidate in [entity.title, *entity.aliases]:
                if candidate and candidate not in alias_terms:
                    alias_terms.append(candidate)
        if not alias_terms:
            return question
        alias_blob = " ".join(alias_terms)
        # Token-budget the alias blob so we do not flood BM25.
        tokens = _tokenize_for_boost(alias_blob)
        if not tokens:
            return question
        capped = tokens[:budget]
        return f"{question} {' '.join(capped)}"

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
                and self._nodes_by_id[m].kind in EVIDENCE_NODE_KINDS
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
        return (
            self._enforce_token_budget(global_contexts[:limit]),
            selected_ids,
        )

    def drift_lite(
        self, question: str, *, limit: int | None = None
    ) -> tuple[list[WikiGraphRetrievedContext], list[str], list[str]]:
        """Local search expanded by deterministic sub-questions.

        Phase 4: when ``retrieval_improvements_enabled`` is true the
        sub-question bundles are fused via reciprocal-rank fusion
        instead of the first-write-wins dict merge, so sub-questions
        contribute to ranking instead of only filling tail slots.
        """
        if limit is None:
            limit = self.config.max_context_chunks
        local_contexts, seed_entities = self.local_search(
            question, limit=max(limit // 2, 3)
        )
        sub_questions = self._derive_sub_questions(question, seed_entities)
        if not self.config.retrieval_improvements_enabled or not sub_questions:
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
            return (
                self._enforce_token_budget(list(combined.values())[:limit]),
                seed_entities,
                sub_questions,
            )

        # RRF over the main local result + each sub-question's local
        # result. Highest fusion score wins regardless of which bundle
        # surfaced it first.
        bundles: list[list[str]] = [[ctx.node_id for ctx in local_contexts]]
        contexts_by_id: dict[str, WikiGraphRetrievedContext] = {
            ctx.node_id: ctx for ctx in local_contexts
        }
        for sub_question in sub_questions:
            sub_contexts, _ = self.local_search(sub_question, limit=3)
            bundles.append([ctx.node_id for ctx in sub_contexts])
            for ctx in sub_contexts:
                if ctx.node_id in contexts_by_id:
                    continue
                contexts_by_id[ctx.node_id] = ctx.model_copy(
                    update={"trace": [*ctx.trace, f"drift:sub={sub_question}"]}
                )

        fused = _reciprocal_rank_fusion(bundles, k=self.config.rrf_k)
        ordered: list[WikiGraphRetrievedContext] = []
        for node_id, fusion_score in fused:
            ctx = contexts_by_id.get(node_id)
            if ctx is None:
                continue
            ordered.append(
                ctx.model_copy(update={"score": float(ctx.score) + fusion_score})
            )
            if len(ordered) >= limit:
                break
        return (
            self._enforce_token_budget(ordered),
            seed_entities,
            sub_questions,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _enforce_token_budget(
        self,
        contexts: list[WikiGraphRetrievedContext],
    ) -> list[WikiGraphRetrievedContext]:
        """Trim ``contexts`` to ``self.config.max_context_tokens``.

        Token count is approximated as ``max(1, len(text) // 4)`` which
        tracks GPT-style BPE tokenization closely enough for prompt-safety
        budgeting. Order is preserved so the highest-ranked contexts are
        kept first; one trace entry is appended to the last retained
        context to make the budget cut observable in run records.
        """
        budget = max(0, int(self.config.max_context_tokens))
        if budget <= 0 or not contexts:
            return contexts
        kept: list[WikiGraphRetrievedContext] = []
        total = 0
        for ctx in contexts:
            estimate = max(1, len(ctx.text) // 4)
            if kept and total + estimate > budget:
                break
            kept.append(ctx)
            total += estimate
        if len(kept) < len(contexts) and kept:
            last = kept[-1]
            kept[-1] = last.model_copy(
                update={
                    "trace": [
                        *last.trace,
                        f"budget:{total}/{budget}-tokens",
                    ]
                }
            )
        return kept

    def _build_lexical_index(self) -> LexicalIndex:
        prefer_simple = str(self.config.lexical_backend).strip().lower() == "simple"
        index = LexicalIndex(prefer_simple=prefer_simple)
        for node in self.index.nodes:
            if node.kind not in EVIDENCE_NODE_KINDS:
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
        # Pass through the node metadata so ``WikiGraphRetrievedContext.
        # citation_ref`` can produce ``#text-unit-N`` anchors for
        # TextUnits without a second lookup.
        metadata = dict(node.metadata or {})
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
            metadata=metadata,
        )

    def _chunks_for_page(self, page_id: str) -> list[str]:
        chunks: list[str] = []
        for neighbor in self._graph.neighbors(page_id):
            node = self._nodes_by_id.get(neighbor)
            if node is None or node.kind != "chunk":
                continue
            chunks.append(neighbor)
        return chunks

    def _text_units_for_document(self, document_id: str) -> list[str]:
        """Return ``text_unit`` neighbors of a ``source_document`` node."""
        units: list[str] = []
        for neighbor in self._graph.neighbors(document_id):
            node = self._nodes_by_id.get(neighbor)
            if node is None or node.kind != "text_unit":
                continue
            units.append(neighbor)
        return units

    def _match_entities(self, question: str) -> list[WikiGraphNode]:
        """Find curated entities mentioned in (or paraphrased by) ``question``.

        Improvements over the original fuzzy-only matcher:

        * **Word-boundary substring match** -- avoids the case where
          fuzzy ``token_set_ratio`` was high enough to clear the
          threshold even when the entity name was not actually a
          standalone token in the question (which caused both noisy
          matches and missed matches in equal measure).
        * **Acronym match** -- treats uppercase tokens in the question
          (``DPR``, ``FiD``, ``RAG``, ``REALM``, ``REPLUG``, ``ORQA``,
          ``RALM``) as case-sensitive exact matches against entity
          names/aliases. This is what makes WikiGraphRAG win on
          acronym-heavy questions where the curated page title spells
          out the full phrase but the question uses the acronym (and
          vice versa).
        * **Acronym-from-title alias generation** -- for multi-word
          entities like ``Dense Passage Retrieval``, ``Fusion-in-
          Decoder``, ``Self-RAG``, the implicit acronym (``DPR``,
          ``FiD``, ``SELFRAG``) is treated as an additional alias.
        """
        import re as _re

        matches: dict[str, tuple[float, WikiGraphNode]] = {}
        lowered = question.lower()
        question_tokens = _re.findall(r"[A-Za-z][A-Za-z0-9\-]+", question)
        question_token_set_lower = {token.lower() for token in question_tokens}
        question_acronyms = {
            token
            for token in question_tokens
            if len(token) >= 2 and token.upper() == token
        }

        def _word_boundary_in(needle: str, haystack: str) -> bool:
            if not needle:
                return False
            pattern = r"(?<![A-Za-z0-9_])" + _re.escape(needle) + r"(?![A-Za-z0-9_])"
            return bool(_re.search(pattern, haystack, _re.IGNORECASE))

        for node in self.index.nodes:
            if node.kind != "entity":
                continue
            aliases = list(node.aliases)
            implicit_acronym = _implicit_acronym(node.title)
            if implicit_acronym and implicit_acronym not in aliases:
                aliases = [*aliases, implicit_acronym]

            # Word-boundary substring -> very strong signal.
            if _word_boundary_in(node.title, question) or any(
                _word_boundary_in(alias, question) for alias in aliases
            ):
                matches[node.id] = (100.0, node)
                continue

            # Acronym match (case-sensitive) -> strong signal.
            entity_label_candidates = {node.title, *aliases}
            for candidate in entity_label_candidates:
                if not candidate:
                    continue
                if candidate in question_acronyms:
                    matches[node.id] = (98.0, node)
                    break
                # Implicit acronym of a multi-word entity also counts.
                if candidate == node.title and implicit_acronym in question_acronyms:
                    matches[node.id] = (98.0, node)
                    break
            if node.id in matches:
                continue

            # Fallback: fuzzy ratio with the original threshold.
            ratio = max(
                fuzz.token_set_ratio(node.title, question),
                *([fuzz.token_set_ratio(alias, question) for alias in aliases] or [0]),
                int(node.title.lower() in lowered) * 100,
                int(any(alias.lower() in lowered for alias in aliases)) * 100,
                int(node.title.lower() in question_token_set_lower) * 100,
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


def _implicit_acronym(title: str) -> str:
    """Build the implicit uppercase acronym of a multi-word entity title.

    ``"Dense Passage Retrieval"`` → ``"DPR"``.
    ``"Fusion-in-Decoder"``       → ``"FID"``.
    ``"Self-RAG"``                → ``"SELFRAG"``.
    ``"RAG"`` (already an acronym) → ``"RAG"``.

    Used by :meth:`WikiGraphContextBuilder._match_entities` to widen
    entity matching when the question uses an acronym that the curated
    entity title spells out, or vice versa.
    """
    if not title:
        return ""
    import re as _re

    tokens = _re.findall(r"[A-Za-z0-9]+", title)
    if len(tokens) <= 1:
        # Single token: keep as-is (already an acronym or single word).
        return title.upper()
    letters = "".join(token[0] for token in tokens if token).upper()
    # Drop common connective words ("of", "the", "for", "in", "on", "a", "and")
    # by skipping leading-letter tokens that are connectives.
    connectives = {"of", "the", "for", "in", "on", "a", "and", "with"}
    keep = "".join(
        token[0] for token in tokens if token.lower() not in connectives
    ).upper()
    return keep or letters


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


def _reciprocal_rank_fusion(
    bundles: list[list[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists using reciprocal-rank fusion (Cormack et al. 2009).

    Each id earns ``1 / (k + rank)`` per bundle it appears in, summed
    across bundles. The output is ``(id, score)`` pairs sorted by
    score (highest first). Empty bundles are ignored. The ``k=60``
    default is the original paper value and is also what most modern
    RAG stacks use.
    """
    scores: dict[str, float] = {}
    for bundle in bundles:
        for rank, node_id in enumerate(bundle, start=1):
            scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


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
