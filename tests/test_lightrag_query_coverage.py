"""Branch-coverage tests for LightRAG retrieval/keywords/query engine."""

from __future__ import annotations

import dataclasses

from graphwiki_kb.providers.base import ProviderResponse, TextProvider
from graphwiki_kb.providers.embedding_base import (
    EmbeddingConfigurationError,
    EmbeddingExecutionError,
)
from graphwiki_kb.services.config_service import (
    DEFAULT_CONFIG,
    resolve_wikigraph_config,
)
from graphwiki_kb.wikigraph.light_context_builder import LightRetriever
from graphwiki_kb.wikigraph.light_graph_store import (
    LightGraphStore,
    LightGraphStorePaths,
)
from graphwiki_kb.wikigraph.light_keywords import extract_query_keywords
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    LightChunk,
    LightGraphIndex,
    QueryKeywords,
    RelationProfile,
)
from graphwiki_kb.wikigraph.light_query_service import LightQueryEngine
from graphwiki_kb.wikigraph.light_vector_store import LightVectorStore

_CFG = resolve_wikigraph_config(DEFAULT_CONFIG).lightrag


def _index() -> LightGraphIndex:
    chunks = [
        LightChunk(
            id="chunk-0",
            source_id="rag",
            source_slug="rag",
            normalized_path="raw/normalized/rag.md",
            chunk_index=0,
            token_count=12,
            text="RAG uses Dense Passage Retrieval.",
            content_hash="h0",
            metadata={"title": "RAG"},
        ),
        LightChunk(
            id="chunk-1",
            source_id="dpr",
            source_slug="dpr",
            normalized_path="raw/normalized/dpr.md",
            chunk_index=1,
            token_count=12,
            text="Dense Passage Retrieval dual encoder.",
            content_hash="h1",
            metadata={"title": "DPR"},
        ),
    ]
    entities = [
        EntityProfile(
            id="entity:rag",
            canonical_name="RAG",
            type="MODEL",
            chunk_ids=["chunk-0"],
            embedding_text="RAG retrieval augmentation",
        ),
        EntityProfile(
            id="entity:dpr",
            canonical_name="Dense Passage Retrieval",
            type="METHOD",
            aliases=["DPR"],
            chunk_ids=["chunk-0", "chunk-1"],
            embedding_text="Dense Passage Retrieval dense retriever",
        ),
    ]
    relations = [
        RelationProfile(
            id="relation:rag:uses:dpr",
            source_entity_id="entity:rag",
            target_entity_id="entity:dpr",
            relation_type="USES",
            keywords=["retrieval augmentation"],
            chunk_ids=["chunk-0"],
            embedding_text="RAG uses Dense Passage Retrieval retrieval augmentation",
        )
    ]
    return LightGraphIndex(
        built_at="t", chunks=chunks, entities=entities, relations=relations
    )


class _Embedder:
    name = "stub"
    model_name = "stub"
    dimension = 2

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_check_embeddings_ensure_failure_disables_vectors() -> None:
    idx = _index()

    class _EnsureBoom:
        name = "x"
        model_name = "x"
        dimension = 2

        def ensure_available(self) -> None:
            raise EmbeddingConfigurationError("no key")

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return []

    vectors = LightVectorStore.from_embeddings(
        ["entity:rag", "entity:dpr"], [[1.0, 0.0], [0.0, 1.0]], model="m", dimension=2
    )
    retriever = LightRetriever(
        entities=idx.entities,
        relations=idx.relations,
        chunks=idx.chunks,
        config=_CFG,
        entity_vectors=vectors,
        embedding_provider=_EnsureBoom(),
    )
    assert retriever.using_embeddings is False


def test_vector_search_failure_falls_back_to_bm25() -> None:
    idx = _index()

    class _BoomEmbedder:
        name = "x"
        model_name = "x"
        dimension = 2

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            raise EmbeddingExecutionError("boom")

    vectors = LightVectorStore.from_embeddings(
        ["entity:rag", "entity:dpr"], [[1.0, 0.0], [0.0, 1.0]], model="m", dimension=2
    )
    retriever = LightRetriever(
        entities=idx.entities,
        relations=idx.relations,
        chunks=idx.chunks,
        config=_CFG,
        entity_vectors=vectors,
        embedding_provider=_BoomEmbedder(),
    )
    assert retriever.using_embeddings is True
    bundle = retriever.retrieve(
        "dpr", QueryKeywords(low_level_keywords=["Dense Passage Retrieval"]), "local"
    )
    # Falls back; diagnostics record the failure.
    assert any("BM25" in d for d in bundle.diagnostics)
    assert bundle.entities


def test_vector_search_dimension_mismatch_falls_back_to_bm25() -> None:
    idx = _index()

    class _WrongDimEmbedder:
        name = "x"
        model_name = "x"
        dimension = 3

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0, 0.0] for _ in texts]

    vectors = LightVectorStore.from_embeddings(
        ["entity:rag", "entity:dpr"], [[1.0, 0.0], [0.0, 1.0]], model="m", dimension=2
    )
    retriever = LightRetriever(
        entities=idx.entities,
        relations=idx.relations,
        chunks=idx.chunks,
        config=_CFG,
        entity_vectors=vectors,
        embedding_provider=_WrongDimEmbedder(),
    )

    bundle = retriever.retrieve(
        "dpr", QueryKeywords(low_level_keywords=["Dense Passage Retrieval"]), "local"
    )

    assert any("dimension mismatch" in d for d in bundle.diagnostics)
    assert bundle.entities


def test_global_uses_relation_vectors() -> None:
    idx = _index()
    entity_vectors = LightVectorStore.from_embeddings(
        ["entity:rag", "entity:dpr"], [[1.0, 0.0], [0.0, 1.0]], model="m", dimension=2
    )
    relation_vectors = LightVectorStore.from_embeddings(
        ["relation:rag:uses:dpr"], [[1.0, 0.0]], model="m", dimension=2
    )
    retriever = LightRetriever(
        entities=idx.entities,
        relations=idx.relations,
        chunks=idx.chunks,
        config=_CFG,
        entity_vectors=entity_vectors,
        relation_vectors=relation_vectors,
        embedding_provider=_Embedder(),
    )
    bundle = retriever.retrieve(
        "retrieval augmentation",
        QueryKeywords(high_level_keywords=["retrieval augmentation"]),
        "global",
    )
    assert any(r.id == "relation:rag:uses:dpr" for r in bundle.relations)


def test_budget_breaks_on_token_cap() -> None:
    idx = _index()
    cfg = dataclasses.replace(
        _CFG,
        retrieval=dataclasses.replace(
            _CFG.retrieval, top_k_chunks=5, max_chunk_tokens=15, max_total_tokens=15
        ),
    )
    retriever = LightRetriever(
        entities=idx.entities, relations=idx.relations, chunks=idx.chunks, config=cfg
    )
    bundle = retriever.retrieve("dense retrieval", QueryKeywords(), "basic")
    # Each chunk is ~12 tokens; the 15-token cap admits only one.
    assert len(bundle.chunks) == 1


def test_keywords_provider_ensure_failure_falls_back() -> None:
    class _Unavail(TextProvider):
        name = "x"

        def ensure_available(self) -> None:
            raise RuntimeError("down")

        def generate(self, request):  # pragma: no cover
            raise AssertionError

    keywords = extract_query_keywords("REALM architecture", provider=_Unavail())
    assert keywords.low_level_keywords or keywords.high_level_keywords


def test_query_engine_from_store_roundtrip(tmp_path) -> None:
    store = LightGraphStore(LightGraphStorePaths(tmp_path / "lr"))
    store.save(_index())
    engine = LightQueryEngine.from_store(store, config=_CFG)
    assert engine is not None

    # keyword_provider override path.
    class _KwProvider(TextProvider):
        name = "kw"

        def generate(self, request):  # type: ignore[override]
            return ProviderResponse(
                text='{"low_level_keywords": ["DPR"], "high_level_keywords": []}',
                model_name="m",
            )

    bundle = engine.find("dpr", method="local", keyword_provider=_KwProvider())
    assert bundle.entities
