"""LightRAG dual-level retrieval and context assembly.

Implements LightRAG's retrieval shape: extract low/high-level query keywords,
match low-level keywords to entity vectors and high-level keywords to relation
vectors, gather one-hop neighbors for higher-order relatedness, and collect the
supporting source chunks that ground every answer. When no embeddings are
available the same shape runs over a BM25 fallback (clearly labeled).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from graphwiki_kb.providers.embedding_base import EmbeddingError, EmbeddingProvider
from graphwiki_kb.services.config_service import LightRagRuntimeConfig
from graphwiki_kb.wikigraph.lexical_index import LexicalDocument, LexicalIndex
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    LightChunk,
    LightQueryMethod,
    LightRetrievedBundle,
    QueryKeywords,
    RelationProfile,
)
from graphwiki_kb.wikigraph.light_tokenizer import Tokenizer, get_default_tokenizer
from graphwiki_kb.wikigraph.light_vector_store import LightVectorStore
from graphwiki_kb.wikigraph.models import WikiGraphRetrievedContext

_RRF_K = 60
_COMPARISON_HINTS = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    "differ",
    "difference",
    "tradeoff",
    "trade-off",
    "contrast",
    "relate",
    "relationship",
)
_THEME_HINTS = (
    "main theme",
    "themes",
    "overall",
    "across",
    "landscape",
    "main ideas",
    "trends",
    "overview",
)


@dataclass
class LightRetriever:
    """Provider-free-capable dual-level retriever over a LightRAG index."""

    entities: list[EntityProfile]
    relations: list[RelationProfile]
    chunks: list[LightChunk]
    config: LightRagRuntimeConfig
    entity_vectors: LightVectorStore | None = None
    relation_vectors: LightVectorStore | None = None
    embedding_provider: EmbeddingProvider | None = None
    tokenizer: Tokenizer | None = None
    diagnostics: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._tok = self.tokenizer or get_default_tokenizer()
        self._entities_by_id = {entity.id: entity for entity in self.entities}
        self._relations_by_id = {relation.id: relation for relation in self.relations}
        self._chunks_by_id = {chunk.id: chunk for chunk in self.chunks}
        self._relations_by_entity: dict[str, list[RelationProfile]] = defaultdict(list)
        for relation in self.relations:
            self._relations_by_entity[relation.source_entity_id].append(relation)
            self._relations_by_entity[relation.target_entity_id].append(relation)
        self._entity_lexical = self._build_lexical(
            [(entity.id, _entity_doc(entity)) for entity in self.entities]
        )
        self._relation_lexical = self._build_lexical(
            [(relation.id, _relation_doc(relation)) for relation in self.relations]
        )
        self._chunk_lexical = self._build_lexical(
            [(chunk.id, chunk.text) for chunk in self.chunks]
        )
        self._embeddings_ready = self._check_embeddings()

    # ------------------------------------------------------------------ #
    # Setup helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_lexical(docs: list[tuple[str, str]]) -> LexicalIndex:
        index = LexicalIndex()
        for doc_id, text in docs:
            index.add(LexicalDocument(doc_id=doc_id, text=text))
        index.fit()
        return index

    def _check_embeddings(self) -> bool:
        if self.embedding_provider is None or self.entity_vectors is None:
            return False
        ensure = getattr(self.embedding_provider, "ensure_available", None)
        if callable(ensure):
            try:
                ensure()
            except EmbeddingError:
                return False
        return True

    @property
    def using_embeddings(self) -> bool:
        """Whether vector retrieval (vs BM25 fallback) is active."""
        return self._embeddings_ready

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def route(self, question: str, keywords: QueryKeywords) -> LightQueryMethod:
        """Choose a method for ``method=auto`` based on query shape."""
        lowered = f" {question.casefold()} "
        if any(hint in lowered for hint in _COMPARISON_HINTS):
            return "hybrid"
        if any(hint in lowered for hint in _THEME_HINTS):
            return "global"
        if self._match_entities(keywords.low_level_keywords):
            return "local"
        return "hybrid"

    def retrieve(
        self,
        question: str,
        keywords: QueryKeywords,
        method: LightQueryMethod,
    ) -> LightRetrievedBundle:
        """Retrieve a structured bundle for ``question`` using ``method``."""
        resolved = self.route(question, keywords) if method == "auto" else method
        trace: list[dict] = [
            {
                "step": "retrieve",
                "method": resolved,
                "using_embeddings": self._embeddings_ready,
            }
        ]
        entities: list[EntityProfile] = []
        relations: list[RelationProfile] = []
        chunk_ranked: list[tuple[LightChunk, float]] = []

        if resolved in {"local", "hybrid", "drift-lite"}:
            local_entities, local_relations, local_chunks = self._local(keywords)
            entities = _merge_unique(entities, local_entities)
            relations = _merge_unique(relations, local_relations)
            chunk_ranked = _rrf_extend(chunk_ranked, local_chunks, weight=1.2)
        if resolved in {"global", "hybrid", "drift-lite"}:
            global_entities, global_relations, global_chunks = self._global(keywords)
            entities = _merge_unique(entities, global_entities)
            relations = _merge_unique(relations, global_relations)
            chunk_ranked = _rrf_extend(chunk_ranked, global_chunks, weight=1.2)
        if resolved in {"basic", "hybrid", "drift-lite"}:
            basic_chunks = self._search_chunks(question)
            chunk_ranked = _rrf_extend(chunk_ranked, basic_chunks, weight=0.8)

        entities = entities[: self.config.retrieval.top_k_entities]
        relations = relations[: self.config.retrieval.top_k_relations]
        chunks = self._budget_chunks([chunk for chunk, _ in chunk_ranked])
        scored = {chunk.id: score for chunk, score in chunk_ranked}

        contexts = [_to_context(chunk, scored.get(chunk.id, 0.0)) for chunk in chunks]
        return LightRetrievedBundle(
            question=question,
            method=resolved,
            low_level_keywords=keywords.low_level_keywords,
            high_level_keywords=keywords.high_level_keywords,
            entities=entities,
            relations=relations,
            chunks=chunks,
            contexts=contexts,
            trace=trace,
            diagnostics=list(self.diagnostics),
        )

    # ------------------------------------------------------------------ #
    # Retrieval primitives                                               #
    # ------------------------------------------------------------------ #

    def _local(
        self, keywords: QueryKeywords
    ) -> tuple[
        list[EntityProfile], list[RelationProfile], list[tuple[LightChunk, float]]
    ]:
        terms = keywords.low_level_keywords or keywords.high_level_keywords
        entity_hits = self._search_entities(terms)
        entities: list[EntityProfile] = []
        relations: list[RelationProfile] = []
        chunk_scores: dict[str, float] = {}
        for entity, score in entity_hits:
            entities.append(entity)
            for chunk_id in entity.chunk_ids:
                chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), score)
            for relation in self._relations_by_entity.get(entity.id, []):
                if relation not in relations:
                    relations.append(relation)
                neighbor_id = (
                    relation.target_entity_id
                    if relation.source_entity_id == entity.id
                    else relation.source_entity_id
                )
                neighbor = self._entities_by_id.get(neighbor_id)
                if neighbor is not None and neighbor not in entities:
                    entities.append(neighbor)
                for chunk_id in relation.chunk_ids:
                    chunk_scores.setdefault(chunk_id, score * 0.5)
        chunks = self._resolve_chunk_scores(chunk_scores)
        return entities, relations, chunks

    def _global(
        self, keywords: QueryKeywords
    ) -> tuple[
        list[EntityProfile], list[RelationProfile], list[tuple[LightChunk, float]]
    ]:
        terms = keywords.high_level_keywords or keywords.low_level_keywords
        relation_hits = self._search_relations(terms)
        entities: list[EntityProfile] = []
        relations: list[RelationProfile] = []
        chunk_scores: dict[str, float] = {}
        for relation, score in relation_hits:
            relations.append(relation)
            for endpoint_id in (relation.source_entity_id, relation.target_entity_id):
                endpoint = self._entities_by_id.get(endpoint_id)
                if endpoint is not None and endpoint not in entities:
                    entities.append(endpoint)
            for chunk_id in relation.chunk_ids:
                chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), score)
        chunks = self._resolve_chunk_scores(chunk_scores)
        return entities, relations, chunks

    def _resolve_chunk_scores(
        self, chunk_scores: dict[str, float]
    ) -> list[tuple[LightChunk, float]]:
        resolved = [
            (self._chunks_by_id[cid], score)
            for cid, score in chunk_scores.items()
            if cid in self._chunks_by_id
        ]
        resolved.sort(key=lambda item: item[1], reverse=True)
        return resolved

    def _match_entities(self, terms: list[str]) -> list[EntityProfile]:
        return [entity for entity, _ in self._search_entities(terms)]

    def _search_entities(self, terms: list[str]) -> list[tuple[EntityProfile, float]]:
        query = " ".join(terms).strip()
        if not query:
            return []
        top_k = self.config.retrieval.top_k_entities
        if self._embeddings_ready:
            hits = self._vector_search(self.entity_vectors, query, top_k)
            if hits is not None:
                return [
                    (self._entities_by_id[eid], score)
                    for eid, score in hits
                    if eid in self._entities_by_id
                ]
        return [
            (self._entities_by_id[hit.doc_id], hit.score)
            for hit in self._entity_lexical.search(query, limit=top_k)
            if hit.doc_id in self._entities_by_id
        ]

    def _search_relations(
        self, terms: list[str]
    ) -> list[tuple[RelationProfile, float]]:
        query = " ".join(terms).strip()
        if not query:
            return []
        top_k = self.config.retrieval.top_k_relations
        if self._embeddings_ready and self.relation_vectors is not None:
            hits = self._vector_search(self.relation_vectors, query, top_k)
            if hits is not None:
                return [
                    (self._relations_by_id[rid], score)
                    for rid, score in hits
                    if rid in self._relations_by_id
                ]
        return [
            (self._relations_by_id[hit.doc_id], hit.score)
            for hit in self._relation_lexical.search(query, limit=top_k)
            if hit.doc_id in self._relations_by_id
        ]

    def _search_chunks(self, question: str) -> list[tuple[LightChunk, float]]:
        top_k = self.config.retrieval.top_k_chunks
        return [
            (self._chunks_by_id[hit.doc_id], hit.score)
            for hit in self._chunk_lexical.search(question, limit=top_k)
            if hit.doc_id in self._chunks_by_id
        ]

    def _vector_search(
        self, store: LightVectorStore | None, query: str, top_k: int
    ) -> list[tuple[str, float]] | None:
        if store is None or self.embedding_provider is None:
            return None
        try:
            query_vector = self.embedding_provider.embed_texts([query])[0]
        except (EmbeddingError, IndexError) as exc:
            self.diagnostics.append(f"query embedding failed, using BM25: {exc}")
            self._embeddings_ready = False
            return None
        if not query_vector or (
            store.dimension > 0 and len(query_vector) != store.dimension
        ):
            self.diagnostics.append(
                "query embedding dimension mismatch, using BM25: "
                f"expected {store.dimension}, got {len(query_vector) if query_vector else 0}"
            )
            self._embeddings_ready = False
            return None
        return store.search(query_vector, top_k)

    def _budget_chunks(self, chunks: list[LightChunk]) -> list[LightChunk]:
        max_chunks = self.config.retrieval.top_k_chunks
        max_tokens = min(
            self.config.retrieval.max_chunk_tokens,
            self.config.retrieval.max_total_tokens,
        )
        out: list[LightChunk] = []
        used_tokens = 0
        for chunk in chunks[:max_chunks]:
            tokens = chunk.token_count or self._tok.count(chunk.text)
            if out and used_tokens + tokens > max_tokens:
                break
            out.append(chunk)
            used_tokens += tokens
        return out


def _entity_doc(entity: EntityProfile) -> str:
    return " ".join(
        [entity.canonical_name, *entity.aliases, entity.type, entity.embedding_text]
    )


def _relation_doc(relation: RelationProfile) -> str:
    return " ".join(
        [relation.relation_type, *relation.keywords, relation.embedding_text]
    )


def _merge_unique(
    base: list[EntityProfile] | list[RelationProfile],
    extra: list[EntityProfile] | list[RelationProfile],
) -> list:
    seen = {item.id for item in base}
    out = list(base)
    for item in extra:
        if item.id not in seen:
            out.append(item)
            seen.add(item.id)
    return out


def _rrf_extend(
    ranked: list[tuple[LightChunk, float]],
    additions: list[tuple[LightChunk, float]],
    *,
    weight: float,
) -> list[tuple[LightChunk, float]]:
    """Reciprocal-rank-fuse ``additions`` into the running chunk ranking."""
    scores: dict[str, float] = {}
    chunk_by_id: dict[str, LightChunk] = {}
    for chunk, score in ranked:
        scores[chunk.id] = score
        chunk_by_id[chunk.id] = chunk
    for rank, (chunk, _) in enumerate(additions, start=1):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + weight / (_RRF_K + rank)
        chunk_by_id.setdefault(chunk.id, chunk)
    fused = [(chunk_by_id[cid], score) for cid, score in scores.items()]
    fused.sort(key=lambda item: item[1], reverse=True)
    return fused


def _to_context(chunk: LightChunk, score: float) -> WikiGraphRetrievedContext:
    metadata: dict[str, object]
    if chunk.compiled_page_path:
        node_kind = "chunk"
        path = chunk.compiled_page_path
        metadata = {"source_id": chunk.source_id}
    else:
        node_kind = "text_unit"
        path = chunk.normalized_path
        metadata = {"source_id": chunk.source_id, "unit_index": chunk.chunk_index}
    return WikiGraphRetrievedContext(
        node_id=chunk.id,
        node_kind=node_kind,
        title=chunk.metadata.get("title", chunk.source_slug),
        path=path,
        text=chunk.text,
        score=score,
        source_ids=[chunk.source_id],
        section="",
        chunk_index=chunk.chunk_index,
        trace=["lightrag"],
        metadata=metadata,
    )
