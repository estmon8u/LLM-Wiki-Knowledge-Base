"""WikiGraphRAG retrieval modes."""

from __future__ import annotations

import re
from typing import Literal

from graphwiki_kb.services.config_service import WikiGraphRuntimeConfig
from graphwiki_kb.wikigraph.community_builder import WikiCommunity
from graphwiki_kb.wikigraph.context_builder import (
    contexts_from_lexical_hits,
    expand_contexts,
    merge_contexts,
    rank_with_pagerank,
)
from graphwiki_kb.wikigraph.entity_extractor import match_entities
from graphwiki_kb.wikigraph.graph_store import WikiGraphStore
from graphwiki_kb.wikigraph.index_builder import WikiGraphBuildResult
from graphwiki_kb.wikigraph.lexical_index import LexicalIndex
from graphwiki_kb.wikigraph.models import WikiGraphRetrievedContext

WikiGraphMethod = Literal["basic", "local", "global", "drift-lite", "auto"]


class WikiGraphQueryService:
    """Provider-free WikiGraphRAG retrieval."""

    def __init__(
        self,
        build: WikiGraphBuildResult,
        runtime: WikiGraphRuntimeConfig,
    ) -> None:
        self.build = build
        self.runtime = runtime
        self.node_by_id = {node.id: node for node in build.nodes}
        self.chunk_records = {chunk.chunk_id: chunk for chunk in build.chunks}
        self.chunk_nodes = {
            node.id: node for node in build.nodes if node.kind == "chunk"
        }
        self.entity_nodes = [node for node in build.nodes if node.kind == "entity"]
        self.community_nodes = [
            node for node in build.nodes if node.kind == "community"
        ]
        self.store = self._load_store()
        self.lexical = LexicalIndex(
            backend=build.snapshot.lexical_backend,
            chunks=build.chunks,
            index_dir=build.output_dir / "lexical",
        )

    def retrieve(
        self,
        question: str,
        *,
        method: WikiGraphMethod = "auto",
    ) -> tuple[list[WikiGraphRetrievedContext], list[dict[str, object]], list[str]]:
        """Retrieve contexts for a question."""
        resolved = self._resolve_method(question, method)
        warnings: list[str] = []
        trace: list[dict[str, object]] = [{"step": "method", "value": resolved}]
        if resolved == "basic":
            contexts = self._retrieve_basic(question)
        elif resolved == "local":
            contexts = self._retrieve_local(question)
        elif resolved == "global":
            contexts = self._retrieve_global(question)
        elif resolved == "drift-lite":
            contexts, drift_trace = self._retrieve_drift_lite(question)
            trace.extend(drift_trace)
        else:
            contexts = self._retrieve_basic(question)
            warnings.append(f"Unknown method fallback: {resolved}")
        contexts = contexts[: self.runtime.max_context_chunks]
        trace.append({"step": "context_count", "value": len(contexts)})
        return contexts, trace, warnings

    def _resolve_method(self, question: str, method: WikiGraphMethod) -> str:
        if method != "auto":
            return method
        lowered = question.lower()
        if any(
            token in lowered
            for token in ("across", "corpus", "themes", "patterns", "overall")
        ):
            return "global"
        if any(token in lowered for token in ("compare", "versus", " vs ", "differ")):
            return "drift-lite"
        if any(token in lowered for token in ("how does", "what is", "where", "who")):
            return "local"
        return "basic"

    def _retrieve_basic(self, question: str) -> list[WikiGraphRetrievedContext]:
        hits = self.lexical.search(question, limit=self.runtime.max_context_chunks)
        return contexts_from_lexical_hits(
            hits,
            chunk_nodes=self.chunk_nodes,
            chunk_records=self.chunk_records,
            limit=self.runtime.max_context_chunks,
        )

    def _retrieve_local(self, question: str) -> list[WikiGraphRetrievedContext]:
        lexical_contexts = self._retrieve_basic(question)
        matched = match_entities(question, self.entity_nodes)
        seed_ids = [entity.id for entity, _ in matched]
        expanded = expand_contexts(
            self.store,
            seed_ids,
            node_by_id=self.node_by_id,
            max_hops=self.runtime.max_hops,
            limit=self.runtime.max_context_chunks,
        )
        for index, (entity, score) in enumerate(matched):
            expanded.insert(
                min(index, len(expanded)),
                WikiGraphRetrievedContext(
                    node_id=entity.id,
                    node_kind="entity",
                    title=entity.title,
                    path=entity.path,
                    text=entity.text,
                    score=score,
                    source_ids=[
                        str(item) for item in entity.metadata.get("source_ids", [])
                    ],
                    trace=[f"entity-match:{entity.title}"],
                ),
            )
        merged = merge_contexts(
            lexical_contexts,
            expanded,
            limit=self.runtime.max_context_chunks,
        )
        return rank_with_pagerank(merged, self.store.pagerank_scores())

    def _retrieve_global(self, question: str) -> list[WikiGraphRetrievedContext]:
        community_hits = self._match_communities(question)
        contexts: list[WikiGraphRetrievedContext] = []
        for community, score in community_hits:
            contexts.append(
                WikiGraphRetrievedContext(
                    node_id=community.community_id,
                    node_kind="community",
                    title=community.title,
                    path=None,
                    text=community.summary,
                    score=score,
                    source_ids=[],
                    trace=[f"community:{community.community_id}"],
                )
            )
            for chunk_id in community.representative_chunks[:3]:
                node_id = (
                    chunk_id if chunk_id.startswith("chunk:") else f"chunk:{chunk_id}"
                )
                node = self.node_by_id.get(node_id)
                if node is None:
                    continue
                record = self.chunk_records.get(
                    chunk_id.replace("chunk:", "", 1)
                    if chunk_id.startswith("chunk:")
                    else chunk_id
                )
                source_ids = [record.source_id] if record and record.source_id else []
                contexts.append(
                    WikiGraphRetrievedContext(
                        node_id=node_id,
                        node_kind="chunk",
                        title=node.title,
                        path=node.path,
                        text=node.text,
                        score=score * 0.8,
                        source_ids=[sid for sid in source_ids if sid],
                        trace=[f"community-chunk:{chunk_id}"],
                    )
                )
        if not contexts:
            return self._retrieve_basic(question)
        return merge_contexts(contexts, limit=self.runtime.max_context_chunks)

    def _retrieve_drift_lite(
        self, question: str
    ) -> tuple[list[WikiGraphRetrievedContext], list[dict[str, object]]]:
        subquestions = self._generate_subquestions(question)
        trace = [{"step": "subquestions", "value": subquestions}]
        groups: list[list[WikiGraphRetrievedContext]] = []
        for subquestion in subquestions:
            groups.append(self._retrieve_local(subquestion))
        merged = merge_contexts(*groups, limit=self.runtime.max_context_chunks)
        return merged, trace

    def _generate_subquestions(self, question: str) -> list[str]:
        parts = re.split(r"\band\b|\bversus\b|\bvs\.?\b|,", question, flags=re.I)
        cleaned = [part.strip(" ?.") for part in parts if part.strip()]
        if len(cleaned) >= 2:
            return cleaned[:4]
        return [
            question,
            f"What evidence supports: {question}",
            f"What limitations apply to: {question}",
        ][:3]

    def _match_communities(self, question: str) -> list[tuple[WikiCommunity, float]]:
        communities = self.build.communities
        if not communities:
            return []
        query_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
        scored: list[tuple[WikiCommunity, float]] = []
        for community in communities:
            haystack = f"{community.title} {community.summary}".lower()
            overlap = sum(1 for token in query_tokens if token in haystack)
            if overlap:
                scored.append((community, overlap / max(len(query_tokens), 1)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:4]

    def _load_store(self) -> WikiGraphStore:
        store = WikiGraphStore()
        for node in self.build.nodes:
            store.add_node(node)
        for edge in self.build.edges:
            store.add_edge(edge)
        node_link = self.build.output_dir / "graph_node_link.json"
        if node_link.exists():
            store.load_node_link(node_link)
        return store
