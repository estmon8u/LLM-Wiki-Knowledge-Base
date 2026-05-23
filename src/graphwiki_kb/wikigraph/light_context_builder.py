"""Dual-level retrieval over a :class:`LightGraphIndex`.

Implements the LightRAG paper's local/global/hybrid retrieval, plus a
``basic`` BM25-style chunk-only fallback and an ``auto`` router that
selects a method based on the question's surface form.

The retriever is **provider-free**: it relies on the locally fit
embedding provider (BM25 by default) so it can run in tests, in CI, and
in the Cursor Cloud Agent VM without any API keys.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from graphwiki_kb.wikigraph.light_embeddings import (
    BM25SparseEmbeddingProvider,
    EmbeddingProvider,
)
from graphwiki_kb.wikigraph.light_keywords import (
    LightKeywordProvider,
    QueryKeywords,
    RuleBasedKeywordProvider,
)
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    LightChunk,
    LightGraphIndex,
    LightQueryMethod,
    LightRetrievedBundle,
    LightRetrievedContext,
    RelationProfile,
)
from graphwiki_kb.wikigraph.light_vector_store import LightVectorStore


@dataclass(frozen=True)
class LightContextBuilderConfig:
    """Tunable knobs for :class:`LightContextBuilder`."""

    top_k_entities: int = 12
    top_k_relations: int = 16
    top_k_chunks: int = 8
    max_entity_tokens: int = 6000
    max_relation_tokens: int = 8000
    max_chunk_tokens: int = 8000
    max_total_tokens: int = 24000
    rrf_k: int = 60
    weight_entity: float = 1.2
    weight_relation: float = 1.2
    weight_chunk: float = 0.8


_GLOBAL_KEYWORDS = (
    "main theme",
    "main themes",
    "main ideas",
    "across",
    "patterns",
    "landscape",
    "whole corpus",
    "ecosystem",
    "trends",
    "evolution",
    "tradeoff",
    "trade-off",
    "tradeoffs",
)
_HYBRID_KEYWORDS = (
    "compare",
    "comparison",
    "differ",
    "difference",
    "differences",
    "relate",
    "related to",
    "relationship",
    "relationship between",
    " versus ",
    " vs ",
    "contrast",
)


@dataclass
class LightContextBuilder:
    """Builds :class:`LightRetrievedBundle` for a question over a LightGraphIndex.

    The builder owns an internal vector store per content type. Callers
    can pass pre-fit vector stores via :meth:`from_index` (preferred)
    or :meth:`from_persisted_vectors` (when the store was loaded from
    disk and we want to skip re-embedding the corpus).
    """

    index: LightGraphIndex
    config: LightContextBuilderConfig = field(default_factory=LightContextBuilderConfig)
    embedding_provider: EmbeddingProvider | None = None
    keyword_provider: LightKeywordProvider | None = None
    precomputed_entity_vectors: list[tuple[str, list[float]]] | None = None
    precomputed_relation_vectors: list[tuple[str, list[float]]] | None = None
    precomputed_chunk_vectors: list[tuple[str, list[float]]] | None = None

    def __post_init__(self) -> None:
        if self.embedding_provider is None:
            self.embedding_provider = _fit_corpus_embedder(self.index)
        if self.keyword_provider is None:
            aliases: list[str] = []
            for ent in self.index.entities:
                aliases.append(ent.canonical_name)
                aliases.extend(ent.aliases)
            self.keyword_provider = RuleBasedKeywordProvider(
                known_aliases=tuple(dict.fromkeys(aliases))
            )

        self._entity_store = LightVectorStore()
        self._relation_store = LightVectorStore()
        self._chunk_store = LightVectorStore()
        self._entity_by_id = {e.id: e for e in self.index.entities}
        self._relation_by_id = {r.id: r for r in self.index.relations}
        self._chunk_by_id = {c.id: c for c in self.index.chunks}
        self._fit_vectors()

    def _fit_vectors(self) -> None:
        assert self.embedding_provider is not None
        # Persisted vectors short-circuit the embed step entirely. This
        # is the path the production query service takes after loading
        # the LightGraph index from disk — kills the per-call BM25
        # refit that made lightrag retrieval ~30x slower than classic.
        if self.precomputed_entity_vectors is not None:
            self._load_precomputed(self._entity_store, self.precomputed_entity_vectors)
        elif self.index.entities:
            ent_vectors = self.embedding_provider.embed_texts(
                [e.embedding_text or e.canonical_name for e in self.index.entities]
            )
            for entity, vec in zip(self.index.entities, ent_vectors, strict=True):
                self._entity_store.add(entity.id, vec)

        if self.precomputed_relation_vectors is not None:
            self._load_precomputed(
                self._relation_store, self.precomputed_relation_vectors
            )
        elif self.index.relations:
            rel_vectors = self.embedding_provider.embed_texts(
                [r.embedding_text or r.relation_type for r in self.index.relations]
            )
            for relation, vec in zip(self.index.relations, rel_vectors, strict=True):
                self._relation_store.add(relation.id, vec)

        if self.precomputed_chunk_vectors is not None:
            self._load_precomputed(self._chunk_store, self.precomputed_chunk_vectors)
        elif self.index.chunks:
            chunk_vectors = self.embedding_provider.embed_texts(
                [c.text for c in self.index.chunks]
            )
            for chunk, vec in zip(self.index.chunks, chunk_vectors, strict=True):
                self._chunk_store.add(chunk.id, vec)

    @staticmethod
    def _load_precomputed(
        store: LightVectorStore, vectors: list[tuple[str, list[float]]]
    ) -> None:
        """Bulk-load persisted ``(id, vector)`` pairs into ``store``."""
        for item_id, vector in vectors:
            if vector:
                store.add(item_id, vector)

    # ---------------------------------------------------------------- #
    # Routing                                                           #
    # ---------------------------------------------------------------- #

    def route(self, question: str, keywords: QueryKeywords) -> LightQueryMethod:
        """Return the :class:`LightQueryMethod` that ``auto`` should run."""
        normalized = f" {question.casefold()} "
        if any(kw in normalized for kw in _HYBRID_KEYWORDS):
            return "hybrid"
        if any(kw in normalized for kw in _GLOBAL_KEYWORDS):
            return "global"
        if keywords.low_level_keywords:
            return "local"
        if keywords.high_level_keywords:
            return "global"
        return "basic"

    # ---------------------------------------------------------------- #
    # Retrieval                                                         #
    # ---------------------------------------------------------------- #

    def retrieve(
        self, question: str, *, method: LightQueryMethod = "auto"
    ) -> LightRetrievedBundle:
        """Return the structured :class:`LightRetrievedBundle` for ``question``."""
        assert self.keyword_provider is not None
        assert self.embedding_provider is not None
        keywords = self.keyword_provider.extract(question)
        diagnostics: list[str] = []

        chosen: LightQueryMethod = method
        if chosen == "auto":
            chosen = self.route(question, keywords)
            diagnostics.append(f"auto-selected {chosen}")

        trace: list[dict] = [
            {
                "step": "keywords",
                "low_level_keywords": list(keywords.low_level_keywords),
                "high_level_keywords": list(keywords.high_level_keywords),
                "keyword_provider": getattr(self.keyword_provider, "name", "unknown"),
            }
        ]

        entities: list[EntityProfile] = []
        relations: list[RelationProfile] = []
        chunks: list[LightChunk] = []
        contexts: list[LightRetrievedContext] = []

        if chosen == "basic":
            chunks, chunk_contexts = self._chunk_search(question)
            contexts.extend(chunk_contexts)
            trace.append({"step": "basic_chunk_search", "chunks": len(chunk_contexts)})
        elif chosen == "local":
            entities, ent_contexts = self._local_entities(keywords, question)
            chunks, chunk_contexts = self._chunks_from_entities(entities)
            relations, rel_contexts = self._relations_from_entities(entities)
            contexts.extend(ent_contexts + rel_contexts + chunk_contexts)
            trace.append(
                {
                    "step": "local_search",
                    "entities": [e.id for e in entities],
                    "relations": [r.id for r in relations],
                    "chunks": len(chunk_contexts),
                }
            )
        elif chosen == "global":
            relations, rel_contexts = self._global_relations(keywords, question)
            entities, ent_contexts = self._entities_from_relations(relations)
            chunks, chunk_contexts = self._chunks_from_entities(entities)
            contexts.extend(rel_contexts + ent_contexts + chunk_contexts)
            trace.append(
                {
                    "step": "global_search",
                    "relations": [r.id for r in relations],
                    "entities": [e.id for e in entities],
                    "chunks": len(chunk_contexts),
                }
            )
        elif chosen == "hybrid":
            local_entities, local_ent_contexts = self._local_entities(
                keywords, question
            )
            global_relations, global_rel_contexts = self._global_relations(
                keywords, question
            )
            ent_from_rel, ent_from_rel_contexts = self._entities_from_relations(
                global_relations
            )
            rel_from_ent, rel_from_ent_contexts = self._relations_from_entities(
                local_entities
            )
            entities = _dedupe_profiles(local_entities + ent_from_rel)
            relations = _dedupe_profiles(global_relations + rel_from_ent)
            chunks, chunk_contexts = self._chunks_from_entities(entities)
            fused_contexts = _rrf_fuse(
                [
                    local_ent_contexts + ent_from_rel_contexts,
                    global_rel_contexts + rel_from_ent_contexts,
                    chunk_contexts,
                ],
                k=self.config.rrf_k,
                weights=[
                    self.config.weight_entity,
                    self.config.weight_relation,
                    self.config.weight_chunk,
                ],
            )
            contexts.extend(fused_contexts)
            trace.append(
                {
                    "step": "hybrid_search",
                    "entities": [e.id for e in entities],
                    "relations": [r.id for r in relations],
                    "chunks": len(chunk_contexts),
                }
            )
        elif chosen == "drift-lite":
            # Re-use hybrid retrieval but mark the trace so callers can
            # render it as the GraphWiki-specific extension.
            local_entities, local_ent_contexts = self._local_entities(
                keywords, question
            )
            global_relations, global_rel_contexts = self._global_relations(
                keywords, question
            )
            entities = local_entities
            relations = global_relations
            chunks, chunk_contexts = self._chunks_from_entities(entities)
            contexts.extend(local_ent_contexts + global_rel_contexts + chunk_contexts)
            trace.append(
                {
                    "step": "drift_lite",
                    "entities": [e.id for e in entities],
                    "relations": [r.id for r in relations],
                    "chunks": len(chunk_contexts),
                }
            )
        else:  # pragma: no cover - protected by Literal type
            raise ValueError(f"Unknown method: {chosen}")

        bundle = LightRetrievedBundle(
            question=question,
            method=chosen,
            low_level_keywords=list(keywords.low_level_keywords),
            high_level_keywords=list(keywords.high_level_keywords),
            entities=entities,
            relations=relations,
            chunks=chunks,
            contexts=_apply_token_budget(
                contexts, max_total_tokens=self.config.max_total_tokens
            ),
            trace=trace,
            diagnostics=diagnostics,
        )
        return bundle

    # ---------------------------------------------------------------- #
    # Internals                                                         #
    # ---------------------------------------------------------------- #

    def _vector_for_query(self, question: str, keywords_text: str = "") -> list[float]:
        assert self.embedding_provider is not None
        text = " ".join([question, keywords_text]).strip() or question
        try:
            vectors = self.embedding_provider.embed_texts([text])
        except RuntimeError:
            return []
        if not vectors:
            return []
        return vectors[0]

    def _query_vector_for_store(
        self, store: LightVectorStore, question: str, keywords_text: str = ""
    ) -> list[float]:
        """Return a query vector with the right dimension for ``store``.

        When the caller supplied precomputed vectors built by a
        different embedder (e.g. OpenAI 3072-dim vectors persisted on
        disk while the in-memory ``embedding_provider`` defaults to
        BM25), the query embedding can have a different dimension.
        Returning an empty vector tells :meth:`LightVectorStore.search`
        to short-circuit instead of raising.
        """
        if store.dimension == 0:
            return []
        vector = self._vector_for_query(question, keywords_text)
        if not vector or len(vector) != store.dimension:
            return []
        return vector

    def _local_entities(
        self, keywords: QueryKeywords, question: str
    ) -> tuple[list[EntityProfile], list[LightRetrievedContext]]:
        query_text = " ".join(keywords.low_level_keywords) or question
        vector = self._query_vector_for_store(self._entity_store, question, query_text)
        hits = self._entity_store.search(vector, top_k=self.config.top_k_entities)
        entities: list[EntityProfile] = []
        contexts: list[LightRetrievedContext] = []
        for hit in hits:
            entity = self._entity_by_id.get(hit.id)
            if entity is None:
                continue
            entities.append(entity)
            contexts.append(
                LightRetrievedContext(
                    kind="entity",
                    id=entity.id,
                    title=entity.canonical_name,
                    score=hit.score,
                    text=entity.profile_text,
                    path=None,
                    chunk_index=None,
                    source_ids=list(entity.source_ids),
                    trace=["entity_vector_search"],
                    metadata={"type": entity.type},
                )
            )
        return entities, contexts

    def _global_relations(
        self, keywords: QueryKeywords, question: str
    ) -> tuple[list[RelationProfile], list[LightRetrievedContext]]:
        query_text = " ".join(keywords.high_level_keywords) or question
        vector = self._query_vector_for_store(
            self._relation_store, question, query_text
        )
        hits = self._relation_store.search(vector, top_k=self.config.top_k_relations)
        relations: list[RelationProfile] = []
        contexts: list[LightRetrievedContext] = []
        for hit in hits:
            relation = self._relation_by_id.get(hit.id)
            if relation is None:
                continue
            relations.append(relation)
            contexts.append(
                LightRetrievedContext(
                    kind="relation",
                    id=relation.id,
                    title=(
                        f"{relation.source_entity_id} "
                        f"{relation.relation_type} {relation.target_entity_id}"
                    ),
                    score=hit.score,
                    text=relation.profile_text,
                    path=None,
                    chunk_index=None,
                    source_ids=list(relation.source_ids),
                    trace=["relation_vector_search"],
                    metadata={"relation_type": relation.relation_type},
                )
            )
        return relations, contexts

    def _chunks_from_entities(
        self, entities: list[EntityProfile]
    ) -> tuple[list[LightChunk], list[LightRetrievedContext]]:
        seen: set[str] = set()
        chunks: list[LightChunk] = []
        contexts: list[LightRetrievedContext] = []
        for entity in entities:
            for chunk_id in entity.chunk_ids:
                if chunk_id in seen:
                    continue
                chunk = self._chunk_by_id.get(chunk_id)
                if chunk is None:
                    continue
                seen.add(chunk_id)
                chunks.append(chunk)
                contexts.append(
                    LightRetrievedContext(
                        kind="chunk",
                        id=chunk.id,
                        title=chunk.source_title or chunk.source_slug,
                        score=0.5,
                        text=chunk.text[:1000],
                        path=chunk.compiled_page_path or chunk.normalized_path,
                        chunk_index=chunk.chunk_index,
                        source_ids=[chunk.source_id],
                        trace=["chunks_from_entities"],
                        metadata={"source_slug": chunk.source_slug},
                    )
                )
                if len(chunks) >= self.config.top_k_chunks:
                    return chunks, contexts
        return chunks, contexts

    def _entities_from_relations(
        self, relations: list[RelationProfile]
    ) -> tuple[list[EntityProfile], list[LightRetrievedContext]]:
        seen: set[str] = set()
        entities: list[EntityProfile] = []
        contexts: list[LightRetrievedContext] = []
        for relation in relations:
            for endpoint_id in (relation.source_entity_id, relation.target_entity_id):
                if endpoint_id in seen:
                    continue
                entity = self._entity_by_id.get(endpoint_id)
                if entity is None:
                    continue
                seen.add(endpoint_id)
                entities.append(entity)
                contexts.append(
                    LightRetrievedContext(
                        kind="entity",
                        id=entity.id,
                        title=entity.canonical_name,
                        score=0.4,
                        text=entity.profile_text,
                        path=None,
                        chunk_index=None,
                        source_ids=list(entity.source_ids),
                        trace=["entities_from_relations"],
                        metadata={"type": entity.type},
                    )
                )
        return entities, contexts

    def _relations_from_entities(
        self, entities: list[EntityProfile]
    ) -> tuple[list[RelationProfile], list[LightRetrievedContext]]:
        wanted_ids: set[str] = set()
        for entity in entities:
            wanted_ids.update(entity.relation_ids)
        relations = [
            self._relation_by_id[rid]
            for rid in wanted_ids
            if rid in self._relation_by_id
        ]
        relations.sort(key=lambda r: r.id)
        contexts = [
            LightRetrievedContext(
                kind="relation",
                id=r.id,
                title=(f"{r.source_entity_id} {r.relation_type} {r.target_entity_id}"),
                score=0.4,
                text=r.profile_text,
                path=None,
                chunk_index=None,
                source_ids=list(r.source_ids),
                trace=["relations_from_entities"],
                metadata={"relation_type": r.relation_type},
            )
            for r in relations
        ]
        return relations, contexts

    def _chunk_search(
        self, question: str
    ) -> tuple[list[LightChunk], list[LightRetrievedContext]]:
        vector = self._query_vector_for_store(self._chunk_store, question)
        hits = self._chunk_store.search(vector, top_k=self.config.top_k_chunks)
        chunks: list[LightChunk] = []
        contexts: list[LightRetrievedContext] = []
        for hit in hits:
            chunk = self._chunk_by_id.get(hit.id)
            if chunk is None:
                continue
            chunks.append(chunk)
            contexts.append(
                LightRetrievedContext(
                    kind="chunk",
                    id=chunk.id,
                    title=chunk.source_title or chunk.source_slug,
                    score=hit.score,
                    text=chunk.text[:1000],
                    path=chunk.compiled_page_path or chunk.normalized_path,
                    chunk_index=chunk.chunk_index,
                    source_ids=[chunk.source_id],
                    trace=["chunk_vector_search"],
                    metadata={"source_slug": chunk.source_slug},
                )
            )
        return chunks, contexts


def _dedupe_profiles(profiles: list):
    seen: set[str] = set()
    out: list = []
    for profile in profiles:
        if profile.id in seen:
            continue
        seen.add(profile.id)
        out.append(profile)
    return out


def _rrf_fuse(
    lists: list[list[LightRetrievedContext]],
    *,
    k: int,
    weights: list[float] | None = None,
) -> list[LightRetrievedContext]:
    """Reciprocal-rank fusion over multiple ranked context lists."""
    weights = weights or [1.0] * len(lists)
    scores: dict[str, float] = defaultdict(float)
    seen: dict[str, LightRetrievedContext] = {}
    for ranked, weight in zip(lists, weights, strict=False):
        for rank, ctx in enumerate(ranked):
            scores[ctx.id] += weight * (1.0 / (k + rank + 1))
            if ctx.id not in seen:
                seen[ctx.id] = ctx
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [
        seen[ctx_id].model_copy(update={"score": float(score)})
        for ctx_id, score in ordered
    ]


def _apply_token_budget(
    contexts: list[LightRetrievedContext], *, max_total_tokens: int
) -> list[LightRetrievedContext]:
    if max_total_tokens <= 0:
        return contexts
    budget = max_total_tokens
    kept: list[LightRetrievedContext] = []
    for ctx in contexts:
        approx_tokens = max(1, math.ceil(len(ctx.text) / 4))
        if budget - approx_tokens < 0:
            break
        budget -= approx_tokens
        kept.append(ctx)
    return kept


def _fit_corpus_embedder(index: LightGraphIndex) -> EmbeddingProvider:
    corpus: list[str] = []
    for entity in index.entities:
        corpus.append(entity.embedding_text or entity.canonical_name)
    for relation in index.relations:
        corpus.append(relation.embedding_text or relation.relation_type)
    for chunk in index.chunks:
        corpus.append(chunk.text)
    if not corpus:
        corpus = [""]
    provider = BM25SparseEmbeddingProvider()
    provider.fit(corpus)
    return provider
