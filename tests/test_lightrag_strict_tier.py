"""Strict-tier vs fallback evidence + citation-anchor + synthesis benchmark."""

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
from graphwiki_kb.wikigraph.light_index_builder import build_lightgraph_index
from graphwiki_kb.wikigraph.light_query_service import LightQueryEngine
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


class _FakeExtractProvider(TextProvider):
    name = "fake"

    def generate(self, request) -> ProviderResponse:  # type: ignore[override]
        payload = {
            "entities": [
                {
                    "name": "REALM",
                    "type": "MODEL",
                    "aliases": [],
                    "description": "A retrieval-augmented LM.",
                    "evidence_quote": "",
                },
                {
                    "name": "Dense Passage Retrieval",
                    "type": "METHOD",
                    "aliases": ["DPR"],
                    "description": "Dense retriever.",
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
                },
            ],
        }
        return ProviderResponse(text=json.dumps(payload), model_name="fake-model")


class _FakeEmbedder:
    name = "openai"
    model_name = "fake-model"
    dimension = 4

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 5), 1.0, 0.0, 0.0] for t in texts]


def _source(root: Path, *, compiled: bool) -> RawSourceRecord:
    rel = "raw/normalized/realm.md"
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(
        "REALM uses Dense Passage Retrieval to retrieve passages for open domain QA.",
        encoding="utf-8",
    )
    if compiled:
        wiki = root / "wiki" / "sources" / "realm.md"
        wiki.parent.mkdir(parents=True, exist_ok=True)
        wiki.write_text("# REALM\n", encoding="utf-8")
    return RawSourceRecord(
        source_id="realm",
        slug="realm",
        title="REALM",
        origin="upload",
        source_type="pdf",
        raw_path="raw/sources/realm.pdf",
        content_hash="h1",
        ingested_at="2026-01-01T00:00:00Z",
        normalized_path=rel,
    )


def _store(tmp_path: Path) -> LightGraphStore:
    return LightGraphStore(LightGraphStorePaths(tmp_path / "graph/wikigraph/lightrag"))


def test_strict_tier_has_provider_extraction_and_vectors(tmp_path: Path) -> None:
    store = _store(tmp_path)
    report = build_lightgraph_index(
        tmp_path,
        [_source(tmp_path, compiled=True)],
        store=store,
        lightrag_config=_LIGHTRAG,
        embeddings_config=_EMB,
        provider=_FakeExtractProvider(),
        embedding_provider=_FakeEmbedder(),
        provider_identity="fake:fake-model",
        tokenizer=RegexWordTokenizer(),
    )
    assert report.tier == "provider+embedded"
    assert report.embedding_tier == "strict"
    assert store.load_build_manifest()["embedding_tier"] == "strict"
    assert store.load_entity_vectors() is not None
    assert store.load_relation_vectors() is not None


def test_fallback_tier_is_deterministic_bm25(tmp_path: Path) -> None:
    store = _store(tmp_path)
    report = build_lightgraph_index(
        tmp_path,
        [_source(tmp_path, compiled=True)],
        store=store,
        lightrag_config=_LIGHTRAG,
        embeddings_config=_EMB,
        provider=None,
        embedding_provider=None,
        tokenizer=RegexWordTokenizer(),
    )
    assert report.tier == "fallback+bm25"
    assert report.embedding_tier == "fallback"
    assert store.load_build_manifest()["embedding_tier"] == "fallback"
    assert store.load_entity_vectors() is None


def test_citation_anchor_uses_chunk_for_compiled_pages(tmp_path: Path) -> None:
    """A retrieved chunk backed by a compiled wiki page cites `path#chunk-N`."""
    store = _store(tmp_path)
    build_lightgraph_index(
        tmp_path,
        [_source(tmp_path, compiled=True)],
        store=store,
        lightrag_config=_LIGHTRAG,
        embeddings_config=_EMB,
        provider=None,
        embedding_provider=None,
        tokenizer=RegexWordTokenizer(),
    )
    engine = LightQueryEngine.from_store(store, config=_LIGHTRAG)
    assert engine is not None
    bundle = engine.find("Dense Passage Retrieval retrieve passages", method="basic")
    assert bundle.contexts
    ctx = bundle.contexts[0]
    assert ctx.node_kind == "chunk"
    assert ctx.citation_ref == "wiki/sources/realm.md#chunk-0"


def test_synthesis_benchmark_is_multi_source() -> None:
    from scripts.rag_eval.dataset import load_benchmark

    questions = load_benchmark(Path("eval") / "benchmark_synthesis.yaml")
    assert len(questions) >= 8
    # Every synthesis question forces 3+ source coverage.
    assert all(len(q.expected_source_ids) >= 3 for q in questions)
