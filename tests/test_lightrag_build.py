"""Tests for the LightRAG index builder (build, incremental, persistence)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.providers.base import ProviderResponse, TextProvider
from graphwiki_kb.services.config_service import (
    DEFAULT_CONFIG,
    EmbeddingsRuntimeConfig,
    resolve_wikigraph_config,
)
from graphwiki_kb.wikigraph.light_graph_store import (
    LightGraphStore,
    LightGraphStorePaths,
)
from graphwiki_kb.wikigraph.light_index_builder import (
    build_lightgraph_index,
    plan_lightgraph_update,
)
from graphwiki_kb.wikigraph.light_tokenizer import RegexWordTokenizer

_LIGHTRAG = dataclasses.replace(
    resolve_wikigraph_config(DEFAULT_CONFIG).lightrag,
    chunk_token_size=40,
    chunk_overlap_tokens=5,
    entity_extract_max_gleaning=0,
)
_EMB_CONFIG = EmbeddingsRuntimeConfig(
    provider="openai", model="fake-model", dimension=4, api_key_env="X"
)


class _FakeEmbedder:
    name = "openai"
    model_name = "fake-model"
    dimension = 4

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 7), 1.0, 0.0, 0.0] for t in texts]


class _FakeExtractProvider(TextProvider):
    name = "fake"

    def generate(self, request) -> ProviderResponse:  # type: ignore[override]
        payload = {
            "entities": [
                {
                    "name": "Dense Passage Retrieval",
                    "type": "METHOD",
                    "aliases": ["DPR"],
                    "description": "A dense retriever.",
                    "evidence_quote": "",
                },
                {
                    "name": "REALM",
                    "type": "MODEL",
                    "aliases": [],
                    "description": "A retrieval-augmented LM.",
                    "evidence_quote": "",
                },
            ],
            "relations": [
                {
                    "source": "REALM",
                    "target": "Dense Passage Retrieval",
                    "relation_type": "USES",
                    "keywords": ["retrieval"],
                    "description": "REALM uses DPR-like retrieval.",
                    "evidence_quote": "",
                }
            ],
        }
        return ProviderResponse(text=json.dumps(payload), model_name="fake-model")


def _make_source(
    root: Path,
    *,
    source_id: str,
    slug: str,
    body: str,
    content_hash: str | None = None,
) -> RawSourceRecord:
    normalized_rel = f"raw/normalized/{slug}.md"
    path = root / normalized_rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return RawSourceRecord(
        source_id=source_id,
        slug=slug,
        title=slug.upper(),
        origin="upload",
        source_type="pdf",
        raw_path=f"raw/sources/{slug}.pdf",
        content_hash=content_hash or f"hash-{source_id}",
        ingested_at="2026-01-01T00:00:00Z",
        normalized_path=normalized_rel,
    )


def _store(root: Path) -> LightGraphStore:
    return LightGraphStore(
        LightGraphStorePaths(root / "graph" / "wikigraph" / "lightrag")
    )


def test_build_empty_corpus(tmp_path: Path) -> None:
    store = _store(tmp_path)
    report = build_lightgraph_index(
        tmp_path,
        [],
        store=store,
        lightrag_config=_LIGHTRAG,
        embeddings_config=_EMB_CONFIG,
        provider=None,
        embedding_provider=None,
        tokenizer=RegexWordTokenizer(),
    )
    assert report.chunk_count == 0
    assert report.entity_count == 0
    assert store.exists()
    loaded = store.load()
    assert loaded is not None
    assert loaded.chunk_count == 0


def test_build_fallback_tier_no_provider(tmp_path: Path) -> None:
    src = _make_source(
        tmp_path,
        source_id="s1",
        slug="dpr",
        body="Dense Passage Retrieval and REALM improve Open Domain QA significantly.",
    )
    store = _store(tmp_path)
    report = build_lightgraph_index(
        tmp_path,
        [src],
        store=store,
        lightrag_config=_LIGHTRAG,
        embeddings_config=_EMB_CONFIG,
        provider=None,
        embedding_provider=None,
        tokenizer=RegexWordTokenizer(),
    )
    assert report.tier == "fallback+bm25"
    assert report.chunk_count >= 1
    assert report.entity_count >= 1
    # No vectors written in BM25 fallback.
    assert store.load_entity_vectors() is None
    # Source contributions recorded.
    contrib = store.load_source_contributions()
    assert "s1" in contrib
    assert contrib["s1"]["chunk_ids"]


def test_build_strict_tier_with_provider_and_embeddings(tmp_path: Path) -> None:
    src = _make_source(
        tmp_path,
        source_id="s1",
        slug="realm",
        body="REALM uses Dense Passage Retrieval to retrieve passages for QA.",
    )
    store = _store(tmp_path)
    report = build_lightgraph_index(
        tmp_path,
        [src],
        store=store,
        lightrag_config=_LIGHTRAG,
        embeddings_config=_EMB_CONFIG,
        provider=_FakeExtractProvider(),
        embedding_provider=_FakeEmbedder(),
        provider_identity="fake:fake-model",
        tokenizer=RegexWordTokenizer(),
    )
    assert report.tier == "provider+embedded"
    assert report.embedding_model == "fake-model"
    # Entity/relation vectors persisted and loadable.
    entity_vectors = store.load_entity_vectors()
    relation_vectors = store.load_relation_vectors()
    assert entity_vectors is not None and len(entity_vectors) >= 1
    assert relation_vectors is not None and len(relation_vectors) >= 1
    # Canonical merge: DPR alias folds into Dense Passage Retrieval.
    loaded = store.load()
    assert loaded is not None
    names = {e.canonical_name for e in loaded.entities}
    assert "Dense Passage Retrieval" in names
    assert any(r.relation_type == "USES" for r in loaded.relations)


def test_build_manifest_written(tmp_path: Path) -> None:
    src = _make_source(tmp_path, source_id="s1", slug="dpr", body="REALM and DPR.")
    store = _store(tmp_path)
    build_lightgraph_index(
        tmp_path,
        [src],
        store=store,
        lightrag_config=_LIGHTRAG,
        embeddings_config=_EMB_CONFIG,
        provider=None,
        embedding_provider=None,
        tokenizer=RegexWordTokenizer(),
    )
    manifest = store.load_build_manifest()
    assert manifest is not None
    assert manifest["source_hashes"] == {"s1": "hash-s1"}
    assert manifest["chunking"]["chunk_token_size"] == 40
    assert manifest["index_schema_version"] == 1


def test_incremental_then_changed_then_missing(tmp_path: Path) -> None:
    src1 = _make_source(
        tmp_path, source_id="s1", slug="dpr", body="Dense Passage Retrieval."
    )
    src2 = _make_source(tmp_path, source_id="s2", slug="realm", body="REALM model.")
    store = _store(tmp_path)
    kwargs: dict = {
        "store": store,
        "lightrag_config": _LIGHTRAG,
        "embeddings_config": _EMB_CONFIG,
        "provider": None,
        "embedding_provider": None,
        "tokenizer": RegexWordTokenizer(),
    }

    first = build_lightgraph_index(tmp_path, [src1, src2], **kwargs)
    assert first.incremental is False
    assert sorted(first.new_source_ids) == ["s1", "s2"]

    # Rebuild unchanged -> incremental, no new/changed.
    second = build_lightgraph_index(tmp_path, [src1, src2], **kwargs)
    assert second.incremental is True
    assert second.new_source_ids == []
    assert second.changed_source_ids == []

    # Change s2's content hash.
    src2_changed = _make_source(
        tmp_path,
        source_id="s2",
        slug="realm",
        body="REALM updated.",
        content_hash="hash-s2-v2",
    )
    third = build_lightgraph_index(tmp_path, [src1, src2_changed], **kwargs)
    assert third.changed_source_ids == ["s2"]

    # Drop s2 -> flagged missing, NOT deleted.
    fourth = build_lightgraph_index(tmp_path, [src1], **kwargs)
    assert fourth.missing_source_ids == ["s2"]
    manifest = store.load_build_manifest()
    assert manifest is not None
    assert manifest["missing_sources"][0]["source_id"] == "s2"
    assert manifest["missing_sources"][0]["requires_review"] is True
    assert any("flagged for review" in w for w in fourth.warnings)


def test_plan_update_detects_contract_change(tmp_path: Path) -> None:
    src = _make_source(tmp_path, source_id="s1", slug="dpr", body="x")
    previous = {
        "source_hashes": {"s1": "hash-s1"},
        "extraction_prompt_hash": "OLD",
        "embedding_identity": "bm25",
    }
    plan = plan_lightgraph_update(
        [src], previous, force=False, prompt_hash="NEW", embedding_identity="bm25"
    )
    # Prompt-hash mismatch breaks the contract -> not incremental.
    assert plan.incremental is False


def test_embedding_failure_falls_back_to_bm25(tmp_path: Path) -> None:
    class _BoomEmbedder:
        name = "openai"
        model_name = "fake-model"
        dimension = 4

        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            from graphwiki_kb.providers.embedding_base import EmbeddingExecutionError

            raise EmbeddingExecutionError("boom")

    src = _make_source(
        tmp_path, source_id="s1", slug="dpr", body="REALM and DPR exist."
    )
    store = _store(tmp_path)
    report = build_lightgraph_index(
        tmp_path,
        [src],
        store=store,
        lightrag_config=_LIGHTRAG,
        embeddings_config=_EMB_CONFIG,
        provider=None,
        embedding_provider=_BoomEmbedder(),
        tokenizer=RegexWordTokenizer(),
    )
    assert report.tier.endswith("+bm25")
    assert store.load_entity_vectors() is None
    assert any("BM25" in w for w in report.warnings)


def test_store_load_corrupt_returns_none(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.paths.root.mkdir(parents=True, exist_ok=True)
    store.paths.index_file.write_text("{bad", encoding="utf-8")
    store.paths.entities_file.write_text("[]", encoding="utf-8")
    assert store.load() is None


def test_store_load_missing_returns_none(tmp_path: Path) -> None:
    assert _store(tmp_path).load() is None
