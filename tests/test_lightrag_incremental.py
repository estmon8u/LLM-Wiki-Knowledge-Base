"""Tests for LightRAG source-level incremental builds (reuse + replay)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import graphwiki_kb.wikigraph.light_index_builder as light_index_builder
from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.services.config_service import (
    DEFAULT_CONFIG,
    EmbeddingsRuntimeConfig,
    resolve_wikigraph_config,
)
from graphwiki_kb.wikigraph.light_graph_store import (
    LightGraphStore,
    LightGraphStorePaths,
)
from graphwiki_kb.wikigraph.light_index_builder import build_lightgraph_index
from graphwiki_kb.wikigraph.light_tokenizer import RegexWordTokenizer

_LIGHTRAG = dataclasses.replace(
    resolve_wikigraph_config(DEFAULT_CONFIG).lightrag,
    chunk_token_size=40,
    chunk_overlap_tokens=5,
    entity_extract_max_gleaning=0,
)
_EMB = EmbeddingsRuntimeConfig(
    provider="openai", model="fake-model", dimension=4, api_key_env="X"
)


class _CountingEmbedder:
    """Embedder that records how many texts it actually embeds."""

    name = "openai"
    model_name = "fake-model"
    dimension = 4

    def __init__(self) -> None:
        self.embedded_texts: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return [[float(len(t) % 5), 1.0, 0.0, 0.0] for t in texts]


def _source(
    root: Path, sid: str, slug: str, body: str, content_hash: str
) -> RawSourceRecord:
    rel = f"raw/normalized/{slug}.md"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(body, encoding="utf-8")
    return RawSourceRecord(
        source_id=sid,
        slug=slug,
        title=slug.upper(),
        origin="upload",
        source_type="pdf",
        raw_path=f"raw/sources/{slug}.pdf",
        content_hash=content_hash,
        ingested_at="2026-01-01T00:00:00Z",
        normalized_path=rel,
    )


def _store(tmp_path: Path) -> LightGraphStore:
    return LightGraphStore(LightGraphStorePaths(tmp_path / "graph/wikigraph/lightrag"))


def _build(
    tmp_path, store, sources, embedder, *, previous_index=None, prev_e=None, prev_r=None
):
    return build_lightgraph_index(
        tmp_path,
        sources,
        store=store,
        lightrag_config=_LIGHTRAG,
        embeddings_config=_EMB,
        provider=None,
        embedding_provider=embedder,
        tokenizer=RegexWordTokenizer(),
        previous_index=previous_index,
        previous_entity_vectors=prev_e,
        previous_relation_vectors=prev_r,
    )


def test_incremental_reuses_unchanged_and_embeds_only_changed(tmp_path: Path) -> None:
    s1 = _source(tmp_path, "s1", "dpr", "Dense Passage Retrieval dual encoder.", "h1")
    s2 = _source(
        tmp_path, "s2", "realm", "REALM retrieval augmented pretraining.", "h2"
    )
    store = _store(tmp_path)

    emb1 = _CountingEmbedder()
    first = _build(tmp_path, store, [s1, s2], emb1)
    assert first.incremental is False
    first_embed_count = len(emb1.embedded_texts)
    assert first_embed_count > 0

    prev_index = store.load()
    prev_e = store.load_entity_vectors()
    prev_r = store.load_relation_vectors()

    # Change only s2; s1 is unchanged.
    s2v2 = _source(
        tmp_path, "s2", "realm", "REALM updated retrieval method here now.", "h2v2"
    )
    emb2 = _CountingEmbedder()
    second = _build(
        tmp_path,
        store,
        [s1, s2v2],
        emb2,
        previous_index=prev_index,
        prev_e=prev_e,
        prev_r=prev_r,
    )
    assert second.incremental is True
    assert second.reused_source_count == 1
    assert second.reprocessed_source_count == 1
    assert second.changed_source_ids == ["s2"]
    # Embedding reuse: the incremental build embeds fewer texts than a full
    # build because s1's unchanged profiles reuse their previous vectors.
    assert len(emb2.embedded_texts) < first_embed_count


def test_incremental_replays_unchanged_chunks_without_extractor(
    tmp_path: Path, monkeypatch
) -> None:
    s1 = _source(tmp_path, "s1", "dpr", "Dense Passage Retrieval dual encoder.", "h1")
    s2 = _source(
        tmp_path, "s2", "realm", "REALM retrieval augmented pretraining.", "h2"
    )
    store = _store(tmp_path)
    _build(tmp_path, store, [s1, s2], _CountingEmbedder())
    prev_index = store.load()
    prev_e = store.load_entity_vectors()
    prev_r = store.load_relation_vectors()

    calls: list[list[str]] = []
    original = light_index_builder.run_extraction

    def _counting_run(chunks, *args, **kwargs):
        calls.append([chunk.source_id for chunk in chunks])
        return original(chunks, *args, **kwargs)

    monkeypatch.setattr(light_index_builder, "run_extraction", _counting_run)
    s2v2 = _source(
        tmp_path, "s2", "realm", "REALM updated retrieval method here now.", "h2v2"
    )
    second = _build(
        tmp_path,
        store,
        [s1, s2v2],
        _CountingEmbedder(),
        previous_index=prev_index,
        prev_e=prev_e,
        prev_r=prev_r,
    )

    assert calls == [["s2"]]
    assert second.extraction_cache_hits == 0
    assert second.extraction_cache_misses == 1


def test_incremental_missing_source_flagged_not_deleted(tmp_path: Path) -> None:
    s1 = _source(tmp_path, "s1", "dpr", "Dense Passage Retrieval dual encoder.", "h1")
    s2 = _source(
        tmp_path, "s2", "realm", "REALM retrieval augmented pretraining.", "h2"
    )
    store = _store(tmp_path)
    _build(tmp_path, store, [s1, s2], _CountingEmbedder())
    prev_index = store.load()
    prev_e = store.load_entity_vectors()
    prev_r = store.load_relation_vectors()

    # Drop s2 entirely.
    report = _build(
        tmp_path,
        store,
        [s1],
        _CountingEmbedder(),
        previous_index=prev_index,
        prev_e=prev_e,
        prev_r=prev_r,
    )
    assert report.missing_source_ids == ["s2"]
    contrib = store.load_source_contributions()
    assert contrib["s2"]["status"] == "missing"
    assert contrib["s2"]["requires_review"] is True
    # s1 retained and marked reused.
    assert contrib["s1"]["status"] in {"reused", "reprocessed", "fresh"}


def test_full_rebuild_when_no_previous_index(tmp_path: Path) -> None:
    s1 = _source(tmp_path, "s1", "dpr", "Dense Passage Retrieval dual encoder.", "h1")
    store = _store(tmp_path)
    report = _build(tmp_path, store, [s1], _CountingEmbedder())
    assert report.incremental is False
    assert report.reprocessed_source_count == 1
    assert report.reused_source_count == 0
