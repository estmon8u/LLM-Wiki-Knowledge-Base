"""Tests for the LightRAG evaluation ablation runners (offline / BM25)."""

from __future__ import annotations

from pathlib import Path

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.services.config_service import ConfigService
from graphwiki_kb.services.manifest_service import ManifestService
from graphwiki_kb.services.project_service import ProjectService, build_project_paths
from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService
from scripts.backend_evaluation_lib import (
    LIGHT_BACKEND_SPECS,
    BenchmarkQuestion,
    LightWikiGraphRunner,
    answer_metrics,
    build_command_context,
    make_light_runner,
    retrieval_metrics,
)


def _build_lightrag_corpus(tmp_path: Path) -> Path:
    paths = build_project_paths(tmp_path)
    ProjectService(paths).ensure_structure()
    ConfigService(paths).ensure_files()
    config = ConfigService(paths).load()
    config["wikigraph"]["mode"] = "lightrag"
    config["embeddings"]["provider"] = "anthropic"  # force BM25, no network
    ConfigService(paths).save(config)
    manifest = ManifestService(paths)
    manifest.ensure_manifest()
    normalized_rel = "raw/normalized/dpr.md"
    (tmp_path / normalized_rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / normalized_rel).write_text(
        "Dense Passage Retrieval is a dual encoder dense retriever. REALM trains "
        "retrieval and language modeling jointly for open domain question answering.",
        encoding="utf-8",
    )
    manifest.save_source(
        RawSourceRecord(
            source_id="dpr",
            slug="dpr",
            title="Dense Passage Retrieval",
            origin="/tmp/dpr.pdf",
            source_type="pdf",
            raw_path="raw/sources/dpr.pdf",
            normalized_path=normalized_rel,
            content_hash="abc",
            ingested_at="2026-01-01T00:00:00Z",
        )
    )
    WikiGraphIndexService(paths=paths, config=config, manifest_service=manifest).build()
    return paths.root


def test_light_backend_specs_cover_ablation_matrix() -> None:
    assert set(LIGHT_BACKEND_SPECS) == {
        "wikigraph-light",
        "wikigraph-light-local",
        "wikigraph-light-global",
        "wikigraph-light-hybrid",
        "wikigraph-light-basic",
        "wikigraph-light-no-vectors-bm25",
    }


def test_light_runner_retrieves_and_answers(tmp_path: Path) -> None:
    root = _build_lightrag_corpus(tmp_path)
    context = build_command_context(root)
    runner = make_light_runner(context, "wikigraph-light-hybrid")
    assert isinstance(runner, LightWikiGraphRunner)
    assert runner.name == "wikigraph-light-hybrid"

    question = BenchmarkQuestion(
        id="q1",
        question="What is Dense Passage Retrieval?",
        expected_sources=("dpr",),
        expected_entities=("Dense Passage Retrieval",),
    )
    retrieval = runner.retrieve(question)
    assert retrieval.backend == "wikigraph-light-hybrid"
    assert retrieval.error is None
    assert retrieval.retrieved_titles
    rmetrics = retrieval_metrics(question, retrieval)
    assert rmetrics["retrieved_count"] >= 1

    answer = runner.answer(question)
    assert answer.backend == "wikigraph-light-hybrid"
    assert answer.error is None
    ametrics = answer_metrics(question, answer)
    assert "answer_quality_score" in ametrics


def test_no_vectors_runner_uses_bm25(tmp_path: Path) -> None:
    root = _build_lightrag_corpus(tmp_path)
    context = build_command_context(root)
    runner = make_light_runner(context, "wikigraph-light-no-vectors-bm25")
    # The runner config forces an unsupported embedding provider -> BM25.
    assert runner.query_service.config["embeddings"]["provider"] == "bm25"
    question = BenchmarkQuestion(id="q2", question="dense retriever dual encoder")
    retrieval = runner.retrieve(question)
    assert retrieval.error is None
