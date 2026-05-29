"""Tests for LightRAG citation-grounded answers and service dispatch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graphwiki_kb.providers import ProviderConfigurationError
from graphwiki_kb.providers.base import ProviderResponse, TextProvider
from graphwiki_kb.services.config_service import (
    DEFAULT_CONFIG,
    resolve_wikigraph_config,
)
from graphwiki_kb.wikigraph.light_answer_service import LightAnswerService
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    LightChunk,
    LightGraphIndex,
    RelationProfile,
)
from graphwiki_kb.wikigraph.light_query_service import LightQueryEngine

_CFG = resolve_wikigraph_config(DEFAULT_CONFIG).lightrag


def _engine() -> LightQueryEngine:
    chunks = [
        LightChunk(
            id="chunk-0",
            source_id="rag",
            source_slug="rag",
            normalized_path="raw/normalized/rag.md",
            compiled_page_path="wiki/sources/rag.md",
            chunk_index=0,
            token_count=12,
            text="RAG uses Dense Passage Retrieval for open domain QA.",
            content_hash="h0",
            metadata={"title": "RAG"},
        )
    ]
    entities = [
        EntityProfile(
            id="entity:dpr",
            canonical_name="Dense Passage Retrieval",
            type="METHOD",
            aliases=["DPR"],
            chunk_ids=["chunk-0"],
            embedding_text="Dense Passage Retrieval DPR retriever",
        )
    ]
    relations: list[RelationProfile] = []
    index = LightGraphIndex(
        built_at="t", chunks=chunks, entities=entities, relations=relations
    )
    return LightQueryEngine(index=index, config=_CFG)


def test_provider_free_answer_cites_chunks() -> None:
    service = LightAnswerService(engine=_engine(), provider=None)
    answer = service.ask("Dense Passage Retrieval open domain QA", method="basic")
    assert answer.contexts
    assert answer.citations
    assert "bm25-fallback" in answer.warnings
    assert answer.citations[0]["ref"] == "wiki/sources/rag.md#chunk-0"


def test_require_provider_without_provider_raises() -> None:
    service = LightAnswerService(engine=_engine(), provider=None)
    with pytest.raises(ProviderConfigurationError):
        service.ask("Dense Passage Retrieval", method="basic", require_provider=True)


def test_no_context_marks_insufficient() -> None:
    empty = LightQueryEngine(index=LightGraphIndex(built_at="t"), config=_CFG)
    service = LightAnswerService(engine=empty, provider=None)
    answer = service.ask("anything", method="basic")
    assert answer.insufficient_evidence is True
    assert "no_context" in answer.warnings


def test_provider_backed_valid_claim() -> None:
    class _P(TextProvider):
        name = "p"

        def generate(self, request):  # type: ignore[override]
            return ProviderResponse(
                text=json.dumps(
                    {
                        "answer": "DPR retrieves passages [C1].",
                        "claims": [
                            {
                                "text": "DPR retrieves passages.",
                                "citation_refs": ["[C1]"],
                            }
                        ],
                        "citations": [],
                        "insufficient_evidence": False,
                    }
                ),
                model_name="m",
            )

    service = LightAnswerService(engine=_engine(), provider=_P())
    answer = service.ask("Dense Passage Retrieval QA", method="basic")
    assert answer.insufficient_evidence is False
    assert answer.contexts
    assert answer.provider_status["mode"] == "provider"


def test_provider_backed_invalid_ref_drops_claim() -> None:
    class _P(TextProvider):
        name = "p"

        def generate(self, request):  # type: ignore[override]
            return ProviderResponse(
                text=json.dumps(
                    {
                        "answer": "Some claim.",
                        "claims": [
                            {"text": "Unsupported.", "citation_refs": ["nope.md#x"]}
                        ],
                        "citations": [],
                        "insufficient_evidence": False,
                    }
                ),
                model_name="m",
            )

    service = LightAnswerService(engine=_engine(), provider=_P())
    answer = service.ask("Dense Passage Retrieval QA", method="basic")
    assert answer.insufficient_evidence is True


def test_provider_error_falls_back() -> None:
    class _P(TextProvider):
        name = "p"

        def generate(self, request):  # type: ignore[override]
            raise RuntimeError("boom")

    service = LightAnswerService(engine=_engine(), provider=_P())
    answer = service.ask("Dense Passage Retrieval QA", method="basic")
    assert "provider-error" in answer.warnings
    assert answer.provider_status["mode"] == "provider-error"


def test_provider_parse_error_falls_back() -> None:
    class _P(TextProvider):
        name = "p"

        def generate(self, request):  # type: ignore[override]
            return ProviderResponse(text="not json at all", model_name="m")

    service = LightAnswerService(engine=_engine(), provider=_P())
    answer = service.ask("Dense Passage Retrieval QA", method="basic")
    assert "provider-parse-error" in answer.warnings


# --------------------------------------------------------------------------- #
# Service dispatch integration (provider-free / BM25 fallback)                #
# --------------------------------------------------------------------------- #


def _make_lightrag_project(tmp_path: Path):
    from graphwiki_kb.models.source_models import RawSourceRecord
    from graphwiki_kb.services.config_service import ConfigService
    from graphwiki_kb.services.manifest_service import ManifestService
    from graphwiki_kb.services.project_service import (
        ProjectService,
        build_project_paths,
    )

    paths = build_project_paths(tmp_path)
    ProjectService(paths).ensure_structure()
    ConfigService(paths).ensure_files()
    config = ConfigService(paths).load()
    config["wikigraph"]["mode"] = "lightrag"
    # Force the provider-free / BM25 path so the test never hits a network
    # embedding API even when OPENAI_API_KEY is present in the environment.
    config["embeddings"]["provider"] = "anthropic"

    manifest = ManifestService(paths)
    manifest.ensure_manifest()
    normalized_rel = "raw/normalized/dpr.md"
    (tmp_path / normalized_rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / normalized_rel).write_text(
        "Dense Passage Retrieval is a dual encoder dense retriever. "
        "REALM trains retrieval and language modeling jointly for open domain QA.",
        encoding="utf-8",
    )
    manifest.save_source(
        RawSourceRecord(
            source_id="src_dpr",
            slug="dpr",
            title="DPR",
            origin="/tmp/dpr.pdf",
            source_type="pdf",
            raw_path="raw/sources/dpr.pdf",
            normalized_path=normalized_rel,
            content_hash="abc",
            ingested_at="2026-01-01T00:00:00Z",
        )
    )
    return paths, config, manifest


def test_index_service_builds_lightrag_mode(tmp_path: Path) -> None:
    from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService

    paths, config, manifest = _make_lightrag_project(tmp_path)
    service = WikiGraphIndexService(
        paths=paths, config=config, manifest_service=manifest
    )
    report = service.build()
    # Adapted to the classic report shape; text_unit_count == lightrag chunk count.
    assert report.text_unit_count >= 1
    assert any("lightrag tier" in w for w in report.warnings)
    # The lightrag store exists.
    assert service.lightrag_store().exists()
    light_report = service.build_lightrag_report()
    assert light_report is not None
    assert light_report.tier.endswith("+bm25")


def test_query_service_dispatches_to_lightrag(tmp_path: Path) -> None:
    from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService
    from graphwiki_kb.services.wikigraph_query_service import WikiGraphQueryService

    paths, config, manifest = _make_lightrag_project(tmp_path)
    index_service = WikiGraphIndexService(
        paths=paths, config=config, manifest_service=manifest
    )
    index_service.build()

    query_service = WikiGraphQueryService(
        paths=paths, index_service=index_service, provider=None, config=config
    )
    find = query_service.find("dense retriever dual encoder", method="hybrid")
    assert find.method == "hybrid"
    assert find.contexts

    answer = query_service.ask("What is Dense Passage Retrieval?", method="hybrid")
    assert answer.method == "hybrid"
    assert answer.contexts
    assert "bm25-fallback" in answer.warnings


def test_query_service_lightrag_missing_index_raises(tmp_path: Path) -> None:
    from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService
    from graphwiki_kb.services.wikigraph_query_service import (
        WikiGraphQueryError,
        WikiGraphQueryService,
    )

    paths, config, manifest = _make_lightrag_project(tmp_path)
    index_service = WikiGraphIndexService(
        paths=paths, config=config, manifest_service=manifest
    )
    query_service = WikiGraphQueryService(
        paths=paths, index_service=index_service, provider=None, config=config
    )
    with pytest.raises(WikiGraphQueryError):
        query_service.find("anything", method="hybrid")
