"""Build LightRAG retrieval bundles and WikiGraph-compatible contexts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from graphwiki_kb.wikigraph.light_chunker import chunk_citation_ref
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    LightChunk,
    LightGraphIndex,
    LightQueryMethod,
    LightRetrievedBundle,
    QueryKeywords,
    RelationProfile,
)
from graphwiki_kb.wikigraph.light_vector_store import HybridRetriever, LightVectorStore
from graphwiki_kb.wikigraph.models import WikiGraphRetrievedContext


@dataclass(frozen=True)
class LightRetrievalConfig:
    """Token budgets and top-k limits."""

    top_k_entities: int = 12
    top_k_relations: int = 16
    top_k_chunks: int = 8
    max_entity_tokens: int = 6000
    max_relation_tokens: int = 8000
    max_chunk_tokens: int = 8000
    max_total_tokens: int = 24000
    rrf_weights: tuple[float, float, float] = (1.2, 1.2, 0.8)


def bundle_to_contexts(bundle: LightRetrievedBundle) -> list[WikiGraphRetrievedContext]:
    """Convert a LightRetrievedBundle into legacy WikiGraph contexts."""
    contexts: list[WikiGraphRetrievedContext] = []
    for rank, chunk in enumerate(bundle.chunks):
        contexts.append(
            WikiGraphRetrievedContext(
                node_id=chunk.id,
                node_kind="text_unit",
                title=chunk.metadata.get("source_title", chunk.source_slug),
                path=chunk.compiled_page_path or chunk.normalized_path,
                text=chunk.text,
                score=float(len(bundle.chunks) - rank),
                source_ids=[chunk.source_id],
                chunk_index=chunk.chunk_index,
                trace=["lightrag_chunk"],
                metadata={
                    "unit_index": chunk.chunk_index,
                    "normalized_path": chunk.normalized_path,
                    "citation_ref": chunk_citation_ref(chunk),
                },
            )
        )
    for rank, entity in enumerate(bundle.entities):
        contexts.append(
            WikiGraphRetrievedContext(
                node_id=entity.id,
                node_kind="entity",
                title=entity.canonical_name,
                path=None,
                text=entity.profile_text,
                score=float(len(bundle.entities) - rank),
                source_ids=list(entity.source_ids),
                trace=["lightrag_entity"],
                metadata={"entity_type": entity.type},
            )
        )
    return contexts


def fuse_and_budget(
    *,
    entities: list[EntityProfile],
    relations: list[RelationProfile],
    chunks: list[LightChunk],
    config: LightRetrievalConfig,
) -> LightRetrievedBundle:
    """Apply simple token budgets before answer synthesis."""
    limited_entities = entities[: config.top_k_entities]
    limited_relations = relations[: config.top_k_relations]
    limited_chunks = chunks[: config.top_k_chunks]
    return LightRetrievedBundle(
        question="",
        method="hybrid",
        entities=limited_entities,
        relations=limited_relations,
        chunks=limited_chunks,
        contexts=[],
    )


def rrf_fuse(
    ranked_lists: list[list[str]],
    *,
    weights: list[float] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion across ranked ID lists."""
    scores: dict[str, float] = {}
    for list_index, ranked in enumerate(ranked_lists):
        weight = 1.0 if weights is None else weights[list_index]
        for rank, item_id in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class LightContextBuilder:
    """Dual-level retrieval over entity/relation profiles and chunks."""

    def __init__(
        self,
        index: LightGraphIndex,
        *,
        entity_retriever: HybridRetriever,
        relation_retriever: HybridRetriever,
        chunk_retriever: HybridRetriever,
        config: LightRetrievalConfig | None = None,
    ) -> None:
        self.index = index
        self.entity_retriever = entity_retriever
        self.relation_retriever = relation_retriever
        self.chunk_retriever = chunk_retriever
        self.config = config or LightRetrievalConfig()
        self._entity_by_id = {entity.id: entity for entity in index.entities}
        self._relation_by_id = {relation.id: relation for relation in index.relations}
        self._chunk_by_id = {chunk.id: chunk for chunk in index.chunks}

    @classmethod
    def from_store(
        cls,
        index: LightGraphIndex,
        store_paths: Any,
        *,
        config: LightRetrievalConfig | None = None,
    ) -> LightContextBuilder:
        entity_retriever = _load_retriever(
            store_paths.entity_vectors_dir,
            index.entities,
            lambda item: item.embedding_text,
            lambda item: item.id,
        )
        relation_retriever = _load_retriever(
            store_paths.relation_vectors_dir,
            index.relations,
            lambda item: item.embedding_text,
            lambda item: item.id,
        )
        chunk_retriever = _load_retriever(
            store_paths.chunk_vectors_dir,
            index.chunks,
            lambda item: item.text,
            lambda item: item.id,
        )
        return cls(
            index,
            entity_retriever=entity_retriever,
            relation_retriever=relation_retriever,
            chunk_retriever=chunk_retriever,
            config=config,
        )

    def retrieve(
        self,
        question: str,
        *,
        method: LightQueryMethod,
        keywords: QueryKeywords,
        query_vector: list[float] | None = None,
    ) -> LightRetrievedBundle:
        trace: list[dict[str, Any]] = []
        entities: list[EntityProfile] = []
        relations: list[RelationProfile] = []
        chunks: list[LightChunk] = []

        local_query = " ".join(keywords.low_level_keywords) or question
        global_query = " ".join(keywords.high_level_keywords) or question

        if method in {"local", "hybrid", "drift-lite"}:
            entity_hits = self.entity_retriever.search(
                local_query, query_vector=query_vector, k=self.config.top_k_entities
            )
            entities = self._expand_entities(entity_hits)
            trace.append({"step": "local_entity_search", "hits": len(entity_hits)})

        if method in {"global", "hybrid", "drift-lite"}:
            relation_hits = self.relation_retriever.search(
                global_query, query_vector=query_vector, k=self.config.top_k_relations
            )
            relations = self._expand_relations(relation_hits)
            trace.append({"step": "global_relation_search", "hits": len(relation_hits)})

        if method in {"basic", "hybrid", "drift-lite", "local", "global"}:
            chunk_hits = self.chunk_retriever.search(
                question, query_vector=query_vector, k=self.config.top_k_chunks
            )
            chunks = [
                self._chunk_by_id[item_id]
                for item_id, _ in chunk_hits
                if item_id in self._chunk_by_id
            ]
            trace.append({"step": "chunk_search", "hits": len(chunk_hits)})

        for entity in entities:
            for chunk_id in entity.chunk_ids:
                chunk = self._chunk_by_id.get(chunk_id)
                if chunk is not None and chunk not in chunks:
                    chunks.append(chunk)
        for relation in relations:
            for chunk_id in relation.chunk_ids:
                chunk = self._chunk_by_id.get(chunk_id)
                if chunk is not None and chunk not in chunks:
                    chunks.append(chunk)

        bundle = fuse_and_budget(
            entities=entities,
            relations=relations,
            chunks=chunks,
            config=self.config,
        )
        bundle.question = question
        bundle.method = method
        bundle.low_level_keywords = list(keywords.low_level_keywords)
        bundle.high_level_keywords = list(keywords.high_level_keywords)
        bundle.trace = trace
        bundle.contexts = bundle_to_contexts(bundle)
        return bundle

    def _expand_entities(self, hits: list[tuple[str, float]]) -> list[EntityProfile]:
        expanded: list[EntityProfile] = []
        seen: set[str] = set()
        for entity_id, _score in hits:
            entity = self._entity_by_id.get(entity_id)
            if entity is None or entity.id in seen:
                continue
            seen.add(entity.id)
            expanded.append(entity)
            for relation_id in entity.relation_ids:
                relation = self._relation_by_id.get(relation_id)
                if relation is None:
                    continue
                for neighbor_id in (
                    relation.source_entity_id,
                    relation.target_entity_id,
                ):
                    neighbor = self._entity_by_id.get(neighbor_id)
                    if neighbor is not None and neighbor.id not in seen:
                        seen.add(neighbor.id)
                        expanded.append(neighbor)
        return expanded[: self.config.top_k_entities]

    def _expand_relations(self, hits: list[tuple[str, float]]) -> list[RelationProfile]:
        expanded: list[RelationProfile] = []
        seen: set[str] = set()
        for relation_id, _score in hits:
            relation = self._relation_by_id.get(relation_id)
            if relation is None or relation.id in seen:
                continue
            seen.add(relation.id)
            expanded.append(relation)
        return expanded[: self.config.top_k_relations]


def _load_retriever(
    vector_dir: Any,
    items: list[Any],
    text_fn: Any,
    id_fn: Any,
) -> HybridRetriever:
    store = LightVectorStore(vector_dir)
    vectors: list[list[float]] | None = None
    if store.exists():
        _meta, vectors = store.load()
    return HybridRetriever(
        ids=[id_fn(item) for item in items],
        texts=[text_fn(item) for item in items],
        vectors=vectors if vectors else None,
    )
