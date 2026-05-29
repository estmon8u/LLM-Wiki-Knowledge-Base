"""Targeted branch-coverage tests for the LightRAG modules."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from graphwiki_kb.providers.embedding_base import EmbeddingConfigurationError
from graphwiki_kb.services.config_service import (
    DEFAULT_CONFIG,
    EmbeddingsRuntimeConfig,
    resolve_wikigraph_config,
)
from graphwiki_kb.wikigraph import light_vector_store as lvs
from graphwiki_kb.wikigraph.light_deduper import (
    DedupeConfig,
    EntityDeduper,
    RelationDeduper,
    _canonical_key,
)
from graphwiki_kb.wikigraph.light_extractor import (
    ExtractionCache,
    ExtractionConfig,
    _provider_available,
    deterministic_extract_chunk,
)
from graphwiki_kb.wikigraph.light_graph_store import (
    LightGraphStore,
    LightGraphStorePaths,
)
from graphwiki_kb.wikigraph.light_index_builder import (
    _build_source_contributions,
    build_lightgraph_index,
)
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    ExtractedEntity,
    ExtractedRelation,
    LightChunk,
    LightExtractionResult,
    RelationProfile,
)
from graphwiki_kb.wikigraph.light_profiler import profile_index
from graphwiki_kb.wikigraph.light_tokenizer import RegexWordTokenizer
from graphwiki_kb.wikigraph.light_vector_store import LightVectorStore

_LIGHTRAG = dataclasses.replace(
    resolve_wikigraph_config(DEFAULT_CONFIG).lightrag,
    chunk_token_size=40,
    chunk_overlap_tokens=5,
    entity_extract_max_gleaning=0,
)
_EMB_CONFIG = EmbeddingsRuntimeConfig(
    provider="openai", model="fake-model", dimension=4, api_key_env="X"
)


def _chunk(text: str, *, chunk_id: str = "c1", source_id: str = "s1") -> LightChunk:
    return LightChunk(
        id=chunk_id,
        source_id=source_id,
        source_slug="s",
        normalized_path="raw/normalized/s.md",
        chunk_index=0,
        token_count=10,
        text=text,
        content_hash="h-" + chunk_id,
    )


# --------------------------------------------------------------------------- #
# Vector store pure-python path                                               #
# --------------------------------------------------------------------------- #


def test_cosine_pure_python_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lvs, "_get_numpy", lambda: None)
    store = LightVectorStore.from_embeddings(
        ["a", "b"], [[1.0, 0.0], [0.0, 1.0]], model="m", dimension=2
    )
    hits = store.search([1.0, 0.0], k=2)
    assert hits[0][0] == "a"


# --------------------------------------------------------------------------- #
# Graph store error/branch paths                                              #
# --------------------------------------------------------------------------- #


def _store(tmp_path: Path) -> LightGraphStore:
    return LightGraphStore(LightGraphStorePaths(tmp_path / "lr"))


def test_store_paths_extraction_cache_dir(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.paths.extraction_cache_dir.name == "extraction_cache"


def test_store_load_invalid_entity_schema_returns_none(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.paths.root.mkdir(parents=True, exist_ok=True)
    store.paths.index_file.write_text('{"built_at": "x"}', encoding="utf-8")
    store.paths.chunks_file.write_text("[]", encoding="utf-8")
    # Entity missing required fields -> ValidationError -> None.
    store.paths.entities_file.write_text('[{"nope": 1}]', encoding="utf-8")
    store.paths.relations_file.write_text("[]", encoding="utf-8")
    assert store.load() is None


def test_store_manifest_and_contrib_corrupt(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.paths.root.mkdir(parents=True, exist_ok=True)
    assert store.load_build_manifest() is None
    assert store.load_source_contributions() == {}
    store.paths.build_manifest_file.write_text("{bad", encoding="utf-8")
    store.paths.source_contributions_file.write_text("{bad", encoding="utf-8")
    assert store.load_build_manifest() is None
    assert store.load_source_contributions() == {}


def test_store_saves_and_loads_chunk_vectors(tmp_path: Path) -> None:
    store = _store(tmp_path)
    from graphwiki_kb.wikigraph.light_models import LightGraphIndex

    chunk_vectors = LightVectorStore.from_embeddings(
        ["c1"], [[1.0, 0.0]], model="m", dimension=2
    )
    store.save(LightGraphIndex(built_at="t"), chunk_vectors=chunk_vectors)
    assert store.load_chunk_vectors() is not None


# --------------------------------------------------------------------------- #
# Index builder branches                                                      #
# --------------------------------------------------------------------------- #


def test_embed_ensure_unavailable_falls_back(tmp_path: Path) -> None:
    class _EnsureBoom:
        name = "openai"
        model_name = "m"
        dimension = 4

        def ensure_available(self) -> None:
            raise EmbeddingConfigurationError("no key")

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return []

    src_path = tmp_path / "raw/normalized/s.md"
    src_path.parent.mkdir(parents=True, exist_ok=True)
    src_path.write_text("REALM and DPR exist here.", encoding="utf-8")
    from graphwiki_kb.models.source_models import RawSourceRecord

    source = RawSourceRecord(
        source_id="s1",
        slug="s",
        title="S",
        origin="upload",
        source_type="pdf",
        raw_path="raw/sources/s.pdf",
        content_hash="h1",
        ingested_at="2026-01-01T00:00:00Z",
        normalized_path="raw/normalized/s.md",
    )
    store = LightGraphStore(LightGraphStorePaths(tmp_path / "lr"))
    report = build_lightgraph_index(
        tmp_path,
        [source],
        store=store,
        lightrag_config=_LIGHTRAG,
        embeddings_config=_EMB_CONFIG,
        provider=None,
        embedding_provider=_EnsureBoom(),
        tokenizer=RegexWordTokenizer(),
    )
    assert report.tier.endswith("+bm25")
    assert any("unavailable" in w for w in report.warnings)


def test_empty_corpus_with_embedder_makes_empty_vectors(tmp_path: Path) -> None:
    class _Embedder:
        name = "openai"
        model_name = "fake-model"
        dimension = 4

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    store = LightGraphStore(LightGraphStorePaths(tmp_path / "lr"))
    report = build_lightgraph_index(
        tmp_path,
        [],
        store=store,
        lightrag_config=_LIGHTRAG,
        embeddings_config=_EMB_CONFIG,
        provider=None,
        embedding_provider=_Embedder(),
        tokenizer=RegexWordTokenizer(),
    )
    assert report.tier == "fallback+embedded"
    entity_vectors = store.load_entity_vectors()
    assert entity_vectors is not None and len(entity_vectors) == 0


def test_build_source_contributions_skips_unknown_chunk() -> None:
    chunks = [_chunk("text", chunk_id="c1", source_id="s1")]
    entities = [
        EntityProfile(
            id="entity:a",
            canonical_name="A",
            type="MODEL",
            chunk_ids=["c1", "missing"],
        )
    ]
    relations = [
        RelationProfile(
            id="rel:a",
            source_entity_id="entity:a",
            target_entity_id="entity:b",
            relation_type="USES",
            chunk_ids=["c1", "missing"],
        )
    ]
    contrib = _build_source_contributions(chunks, entities, relations)
    assert contrib["s1"]["entity_contributions"]["entity:a"] == ["c1"]
    assert contrib["s1"]["relation_contributions"]["rel:a"] == ["c1"]


# --------------------------------------------------------------------------- #
# Extractor branches                                                          #
# --------------------------------------------------------------------------- #


def test_extraction_cache_corrupt_returns_none(tmp_path: Path) -> None:
    cache = ExtractionCache(tmp_path / "ec")
    (tmp_path / "ec").mkdir(parents=True, exist_ok=True)
    key = ExtractionCache.key("h", "p", "id")
    (tmp_path / "ec" / f"{key}.json").write_text("{bad", encoding="utf-8")
    assert cache.get(key) is None


def test_provider_available_handles_ensure_failure() -> None:
    class _Unavail:
        name = "x"

        def ensure_available(self) -> None:
            raise RuntimeError("down")

        def generate(self, request):  # pragma: no cover - never called
            raise AssertionError

    assert _provider_available(_Unavail()) is False
    assert _provider_available(None) is False


def test_deterministic_extraction_empty_text() -> None:
    result = deterministic_extract_chunk(
        _chunk("... --- ,,,"),
        ExtractionConfig(entity_types=("MODEL",), relation_types=("USES",)),
    )
    assert isinstance(result, LightExtractionResult)
    assert result.entities == []
    assert result.relations == []


# --------------------------------------------------------------------------- #
# Deduper branches                                                            #
# --------------------------------------------------------------------------- #


def test_canonical_key_empty_fallback() -> None:
    assert _canonical_key("!!!") == ""


def test_acronym_match_when_full_name_added_first() -> None:
    deduper = EntityDeduper(DedupeConfig())
    deduper.add(
        ExtractedEntity(name="Fusion in Decoder", type="METHOD", chunk_ids=["c1"])
    )
    deduper.add(ExtractedEntity(name="FID", type="METHOD", chunk_ids=["c2"]))
    profiles, _ = deduper.build()
    assert len(profiles) == 1
    assert profiles[0].canonical_name == "Fusion in Decoder"


def test_fuzzy_match_merges_near_duplicates() -> None:
    deduper = EntityDeduper(DedupeConfig(fuzzy_threshold=85))
    deduper.add(
        ExtractedEntity(name="Dense Passage Retrieval", type="METHOD", chunk_ids=["c1"])
    )
    deduper.add(
        ExtractedEntity(
            name="Dense Passage Retrievals", type="METHOD", chunk_ids=["c2"]
        )
    )
    profiles, _ = deduper.build()
    assert len(profiles) == 1


def test_relation_deduper_unresolved_and_self_loop() -> None:
    name_to_id = {
        _canonical_key("RAG"): "entity:rag",
        _canonical_key("DPR"): "entity:dpr",
    }
    deduper = RelationDeduper(name_to_id)
    # Unresolved endpoint.
    assert (
        deduper.add(
            ExtractedRelation(source="RAG", target="Unknown", relation_type="USES")
        )
        is None
    )
    # Self loop.
    assert (
        deduper.add(ExtractedRelation(source="RAG", target="RAG", relation_type="USES"))
        is None
    )
    assert deduper.build() == []


# --------------------------------------------------------------------------- #
# Profiler branches                                                           #
# --------------------------------------------------------------------------- #


def test_profiler_snippet_truncation_and_missing_chunk() -> None:
    long_text = "word " * 200
    chunks = [_chunk(long_text, chunk_id="c1", source_id="s1")]
    entities = [
        EntityProfile(
            id="entity:a",
            canonical_name="A",
            type="MODEL",
            chunk_ids=["c1", "missing-chunk"],
            description="desc",
        )
    ]
    profile_index(entities, [], chunks, updated_at="t")
    assert "..." in entities[0].profile_text
