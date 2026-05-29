"""Tests for LightRAG keywords, dual-level retrieval, and the query engine."""

from __future__ import annotations

import dataclasses

from graphwiki_kb.providers.base import ProviderResponse, TextProvider
from graphwiki_kb.services.config_service import (
    DEFAULT_CONFIG,
    resolve_wikigraph_config,
)
from graphwiki_kb.wikigraph.light_context_builder import LightRetriever
from graphwiki_kb.wikigraph.light_keywords import extract_query_keywords
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    LightChunk,
    QueryKeywords,
    RelationProfile,
)
from graphwiki_kb.wikigraph.light_query_service import LightQueryEngine
from graphwiki_kb.wikigraph.light_vector_store import LightVectorStore

_CFG = dataclasses.replace(
    resolve_wikigraph_config(DEFAULT_CONFIG).lightrag,
    chunk_token_size=40,
)


# --------------------------------------------------------------------------- #
# Fixtures: a tiny hand-built index                                           #
# --------------------------------------------------------------------------- #


def _chunk(
    cid: str, source_id: str, text: str, *, compiled: str | None = None
) -> LightChunk:
    return LightChunk(
        id=cid,
        source_id=source_id,
        source_slug=source_id,
        normalized_path=f"raw/normalized/{source_id}.md",
        compiled_page_path=compiled,
        chunk_index=int(cid.split("-")[-1]),
        token_count=12,
        text=text,
        content_hash="h-" + cid,
        metadata={"title": source_id.upper()},
    )


def _build_index() -> (
    tuple[list[EntityProfile], list[RelationProfile], list[LightChunk]]
):
    chunks = [
        _chunk(
            "chunk-0",
            "rag",
            "RAG uses Dense Passage Retrieval for open domain QA.",
            compiled="wiki/sources/rag.md",
        ),
        _chunk(
            "chunk-1",
            "dpr",
            "Dense Passage Retrieval is a dual-encoder dense retriever.",
        ),
        _chunk(
            "chunk-2",
            "realm",
            "REALM trains retrieval and language modeling jointly across the corpus.",
        ),
    ]
    entities = [
        EntityProfile(
            id="entity:rag",
            canonical_name="Retrieval-Augmented Generation",
            type="MODEL",
            aliases=["RAG"],
            description="A retrieval-augmented generator.",
            keywords=["retrieval augmentation"],
            chunk_ids=["chunk-0"],
            source_ids=["rag"],
            embedding_text="Retrieval-Augmented Generation RAG MODEL retrieval augmentation",
        ),
        EntityProfile(
            id="entity:dpr",
            canonical_name="Dense Passage Retrieval",
            type="METHOD",
            aliases=["DPR"],
            description="A dense retriever.",
            keywords=["dense retriever"],
            chunk_ids=["chunk-0", "chunk-1"],
            source_ids=["rag", "dpr"],
            embedding_text="Dense Passage Retrieval DPR METHOD dense retriever dual encoder",
        ),
        EntityProfile(
            id="entity:realm",
            canonical_name="REALM",
            type="MODEL",
            description="Retrieval-augmented LM pretraining.",
            keywords=["retrieval pretraining"],
            chunk_ids=["chunk-2"],
            source_ids=["realm"],
            embedding_text="REALM MODEL retrieval augmented language model pretraining",
        ),
    ]
    relations = [
        RelationProfile(
            id="relation:rag:uses:dpr",
            source_entity_id="entity:rag",
            target_entity_id="entity:dpr",
            relation_type="USES",
            keywords=["retrieval augmentation", "dense retriever"],
            description="RAG uses DPR to retrieve passages.",
            chunk_ids=["chunk-0"],
            source_ids=["rag"],
            embedding_text="RAG uses Dense Passage Retrieval retrieval augmentation",
        )
    ]
    return entities, relations, chunks


# --------------------------------------------------------------------------- #
# Keywords                                                                    #
# --------------------------------------------------------------------------- #


def test_keyword_fallback_low_and_high() -> None:
    keywords = extract_query_keywords(
        "Compare REALM and RAG retrieval tradeoffs",
        known_aliases={"RAG", "REALM"},
    )
    assert (
        "RAG" in keywords.low_level_keywords or "REALM" in keywords.low_level_keywords
    )
    assert (
        "tradeoffs" in keywords.high_level_keywords
        or "compare" in keywords.high_level_keywords
    )


def test_keyword_provider_used_when_available() -> None:
    class _KwProvider(TextProvider):
        name = "kw"

        def generate(self, request):  # type: ignore[override]
            return ProviderResponse(
                text='{"low_level_keywords": ["DPR"], "high_level_keywords": ["retrieval"]}',
                model_name="m",
            )

    keywords = extract_query_keywords("how does dpr work", provider=_KwProvider())
    assert keywords.low_level_keywords == ["DPR"]
    assert keywords.high_level_keywords == ["retrieval"]


def test_keyword_provider_failure_falls_back() -> None:
    class _BoomProvider(TextProvider):
        name = "boom"

        def generate(self, request):  # type: ignore[override]
            raise RuntimeError("down")

    keywords = extract_query_keywords("REALM architecture", provider=_BoomProvider())
    assert keywords.low_level_keywords  # fallback produced something


# --------------------------------------------------------------------------- #
# Retrieval (BM25 fallback, no embeddings)                                    #
# --------------------------------------------------------------------------- #


def _retriever() -> LightRetriever:
    entities, relations, chunks = _build_index()
    return LightRetriever(
        entities=entities, relations=relations, chunks=chunks, config=_CFG
    )


def test_local_retrieves_entity_and_incident_relation() -> None:
    retriever = _retriever()
    keywords = QueryKeywords(low_level_keywords=["Dense Passage Retrieval", "DPR"])
    bundle = retriever.retrieve("How does DPR retrieve passages?", keywords, "local")
    assert bundle.method == "local"
    entity_ids = {e.id for e in bundle.entities}
    assert "entity:dpr" in entity_ids
    # Incident relation surfaced, plus the neighbor entity (RAG).
    assert any(r.id == "relation:rag:uses:dpr" for r in bundle.relations)
    assert "entity:rag" in entity_ids
    # Citations map to returned source chunks.
    assert bundle.contexts
    assert all(ctx.node_id in {c.id for c in bundle.chunks} for ctx in bundle.contexts)


def test_global_retrieves_relation_by_theme() -> None:
    retriever = _retriever()
    keywords = QueryKeywords(high_level_keywords=["retrieval augmentation"])
    bundle = retriever.retrieve("retrieval augmentation themes", keywords, "global")
    assert bundle.method == "global"
    assert any(r.relation_type == "USES" for r in bundle.relations)
    # Endpoint entities included.
    assert {"entity:rag", "entity:dpr"} <= {e.id for e in bundle.entities}


def test_hybrid_fuses_entities_relations_chunks() -> None:
    retriever = _retriever()
    keywords = QueryKeywords(
        low_level_keywords=["DPR"], high_level_keywords=["retrieval augmentation"]
    )
    bundle = retriever.retrieve("Compare RAG and DPR", keywords, "hybrid")
    assert bundle.method == "hybrid"
    assert bundle.entities
    assert bundle.relations
    assert bundle.chunks


def test_basic_chunk_only() -> None:
    retriever = _retriever()
    keywords = QueryKeywords()
    bundle = retriever.retrieve("dual encoder dense retriever", keywords, "basic")
    assert bundle.method == "basic"
    assert bundle.chunks
    assert bundle.entities == []


def test_auto_routes_comparison_to_hybrid() -> None:
    retriever = _retriever()
    keywords = QueryKeywords(low_level_keywords=["RAG", "REALM"])
    bundle = retriever.retrieve("Compare RAG versus REALM", keywords, "auto")
    assert bundle.method == "hybrid"


def test_auto_routes_theme_to_global() -> None:
    retriever = _retriever()
    keywords = QueryKeywords(high_level_keywords=["themes"])
    bundle = retriever.retrieve(
        "What are the main themes across the corpus?", keywords, "auto"
    )
    assert bundle.method == "global"


def test_auto_routes_entity_match_to_local() -> None:
    retriever = _retriever()
    keywords = QueryKeywords(low_level_keywords=["Dense Passage Retrieval"])
    bundle = retriever.retrieve("Dense Passage Retrieval", keywords, "auto")
    assert bundle.method == "local"


def test_budget_caps_chunks() -> None:
    entities, relations, chunks = _build_index()
    cfg = dataclasses.replace(
        _CFG,
        retrieval=dataclasses.replace(_CFG.retrieval, top_k_chunks=1),
    )
    retriever = LightRetriever(
        entities=entities, relations=relations, chunks=chunks, config=cfg
    )
    bundle = retriever.retrieve(
        "retrieval", QueryKeywords(low_level_keywords=["DPR"]), "basic"
    )
    assert len(bundle.chunks) <= 1


# --------------------------------------------------------------------------- #
# Retrieval with embeddings                                                   #
# --------------------------------------------------------------------------- #


class _StubEmbedder:
    """Maps a few known phrases to fixed 2-D vectors; everything else -> origin."""

    name = "stub"
    model_name = "stub"
    dimension = 2

    _TABLE = {
        "dense passage retrieval": [1.0, 0.0],
        "dpr": [1.0, 0.0],
        "retrieval augmentation": [0.0, 1.0],
    }

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            key = text.strip().casefold()
            out.append(self._TABLE.get(key, [0.5, 0.5]))
        return out


def test_retriever_uses_embeddings_when_available() -> None:
    entities, relations, chunks = _build_index()
    entity_vectors = LightVectorStore.from_embeddings(
        ["entity:rag", "entity:dpr", "entity:realm"],
        [[0.0, 1.0], [1.0, 0.0], [0.2, 0.2]],
        model="stub",
        dimension=2,
    )
    retriever = LightRetriever(
        entities=entities,
        relations=relations,
        chunks=chunks,
        config=_CFG,
        entity_vectors=entity_vectors,
        embedding_provider=_StubEmbedder(),
    )
    assert retriever.using_embeddings is True
    bundle = retriever.retrieve(
        "dpr", QueryKeywords(low_level_keywords=["Dense Passage Retrieval"]), "local"
    )
    # The DPR vector [1,0] is closest to query [1,0].
    assert bundle.entities[0].id == "entity:dpr"


# --------------------------------------------------------------------------- #
# Query engine                                                                #
# --------------------------------------------------------------------------- #


def test_query_engine_find_and_find_result() -> None:
    entities, relations, chunks = _build_index()
    from graphwiki_kb.wikigraph.light_models import LightGraphIndex

    index = LightGraphIndex(
        built_at="t", chunks=chunks, entities=entities, relations=relations
    )
    engine = LightQueryEngine(index=index, config=_CFG)
    bundle = engine.find("How does DPR work?", method="local")
    assert bundle.entities
    result = engine.find_result("Compare RAG and DPR", method="hybrid")
    assert result.method == "hybrid"
    assert result.contexts
    assert "Dense Passage Retrieval" in result.entities or result.entities


def test_query_engine_from_store_missing(tmp_path) -> None:
    from graphwiki_kb.wikigraph.light_graph_store import (
        LightGraphStore,
        LightGraphStorePaths,
    )

    store = LightGraphStore(LightGraphStorePaths(tmp_path / "lr"))
    assert LightQueryEngine.from_store(store, config=_CFG) is None
