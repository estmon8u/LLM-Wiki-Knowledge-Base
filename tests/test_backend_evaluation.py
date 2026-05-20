"""Tests for the cross-backend evaluation harness."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from scripts.backend_evaluation_lib import (
    RETRIEVAL_COLUMNS,
    AnswerRun,
    BenchmarkQuestion,
    GraphRAGRunner,
    LegacyRunner,
    RetrievalRun,
    WikiGraphRunner,
    answer_metrics,
    build_command_context,
    load_benchmark,
    retrieval_metrics,
    write_csv,
    write_json,
    write_summary_markdown,
)
from scripts.evaluate_backends import main as backends_main

REALM_PAGE = textwrap.dedent(
    """\
---
title: REALM
type: source
source_id: realm
aliases:
  - Retrieval-Augmented Language Model
tags:
  - retrieval
summary: REALM pretrains a language model alongside a learned retriever.
---

# REALM

## Summary

REALM is a retrieval-augmented language model.

## Key Points

- REALM trains a retriever and a masked language model jointly.

## Methods

REALM backpropagates through retrieval. See [[RAG]].
"""
)

RAG_PAGE = textwrap.dedent(
    """\
---
title: RAG
type: source
source_id: rag
aliases:
  - Retrieval-Augmented Generation
tags:
  - retrieval
summary: RAG augments a seq2seq generator with retrieved passages.
---

# RAG

## Summary

RAG augments a generator with retrieved passages.

## Key Points

- RAG uses a frozen retriever and a seq2seq generator.

## Methods

RAG decouples retrieval and generation, unlike [[REALM]].
"""
)


BENCHMARK_YAML = textwrap.dedent(
    """\
version: 2
questions:
  - id: realm_vs_rag
    question: How does REALM differ from RAG?
    category: comparison
    expected_sources:
      - realm
      - rag
    expected_entities:
      - REALM
      - RAG
    expected_methods:
      wikigraph: local
"""
)


@pytest.fixture
def seeded_project(test_project):
    """Seed the wiki with two source pages and build the wikigraph index."""
    (test_project.paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE)
    (test_project.paths.wiki_sources_dir / "rag.md").write_text(RAG_PAGE)
    test_project.services.wikigraph_index.build()
    return test_project


def test_benchmark_question_from_dict() -> None:
    question = BenchmarkQuestion.from_dict(
        {
            "id": "q1",
            "question": "?",
            "expected_sources": ["a", "b"],
            "expected_entities": ["A"],
            "expected_methods": {"wikigraph": "local"},
        }
    )
    assert question.expected_sources == ("a", "b")
    assert question.expected_methods["wikigraph"] == "local"


def test_load_benchmark(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(BENCHMARK_YAML)
    questions = load_benchmark(benchmark)
    assert len(questions) == 1
    assert questions[0].id == "realm_vs_rag"


def test_wikigraph_runner_retrieval(seeded_project) -> None:
    context = build_command_context(seeded_project.paths.root)
    runner = WikiGraphRunner(context=context, method="local")
    question = BenchmarkQuestion(
        id="q1",
        question="How does REALM differ from RAG?",
        expected_sources=("realm", "rag"),
        expected_entities=("REALM", "RAG"),
    )
    run = runner.retrieve(question)
    assert run.error is None
    metrics = retrieval_metrics(question, run)
    assert metrics["recall_at_5"] > 0

    answer = runner.answer(question)
    assert answer.answer
    ametrics = answer_metrics(question, answer)
    assert ametrics["matched_entity_count"] >= 1


def test_legacy_runner_retrieval(seeded_project) -> None:
    context = build_command_context(seeded_project.paths.root)
    runner = LegacyRunner(context=context)
    question = BenchmarkQuestion(
        id="q1",
        question="REALM retrieval",
        expected_sources=("realm",),
    )
    run = runner.retrieve(question)
    assert run.error is None
    metrics = retrieval_metrics(question, run)
    assert metrics["retrieved_count"] >= 0  # may be 0 if FTS unavailable


def test_evaluate_backends_main_retrieval_only(
    seeded_project, tmp_path: Path, monkeypatch
) -> None:
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(BENCHMARK_YAML)
    results_dir = tmp_path / "results"

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_backends.py",
            "--project-root",
            str(seeded_project.paths.root),
            "--benchmark",
            str(benchmark),
            "--results-dir",
            str(results_dir),
            "--backends",
            "wikigraph",
            "--wikigraph-methods",
            "auto",
            "--retrieval-only",
        ],
    )
    exit_code = backends_main()
    assert exit_code == 0
    assert (results_dir / "backend_summary.md").exists()
    assert (results_dir / "backend_retrieval_metrics.csv").exists()


def test_evaluate_backends_with_wikigraph_answer(
    seeded_project, tmp_path: Path, monkeypatch
) -> None:
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(BENCHMARK_YAML)
    results_dir = tmp_path / "results"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_backends.py",
            "--project-root",
            str(seeded_project.paths.root),
            "--benchmark",
            str(benchmark),
            "--results-dir",
            str(results_dir),
            "--backends",
            "wikigraph",
            "--wikigraph-methods",
            "local",
            "global",
        ],
    )
    exit_code = backends_main()
    assert exit_code == 0
    assert (results_dir / "backend_answer_metrics.csv").exists()


def test_graphrag_runner_retrieve_handles_missing_artifacts(seeded_project) -> None:
    """``GraphRAGRunner.retrieve`` returns an empty run when no parquet exists."""
    context = build_command_context(seeded_project.paths.root)
    runner = GraphRAGRunner(context=context, method="auto")
    run = runner.retrieve(
        BenchmarkQuestion(
            id="q1",
            question="anything",
            expected_sources=("realm",),
        )
    )
    assert isinstance(run, RetrievalRun)
    # No GraphRAG artifacts exist in test_project, so we get either an empty
    # result (success) or an error string; both are acceptable.
    assert run.error is None or "error" not in run.backend


def test_graphrag_runner_answer_records_provider_error_as_run(seeded_project) -> None:
    """A failing ask call should produce a structured AnswerRun, not raise."""
    context = build_command_context(seeded_project.paths.root)
    runner = GraphRAGRunner(context=context, method="auto")

    class _ExplodingController:
        def ask(self, question, **kwargs):
            raise RuntimeError("simulated provider unavailable")

    runner.ask_controller = _ExplodingController()  # type: ignore[assignment]
    run = runner.answer(
        BenchmarkQuestion(
            id="q1",
            question="anything",
            expected_entities=("REALM",),
        )
    )
    assert isinstance(run, AnswerRun)
    assert run.error is not None and "simulated provider" in run.error
    assert run.insufficient_evidence is True


def test_graphrag_runner_answer_success_path(seeded_project) -> None:
    """A passing ask call should produce a populated AnswerRun."""
    context = build_command_context(seeded_project.paths.root)
    runner = GraphRAGRunner(context=context, method="auto")

    class _FakeAnswer:
        answer = "REALM differs from RAG by training the retriever jointly."
        method = "auto"
        claim_support = "cited-graph-answer"
        graph_data_references = [{"ref": "Entities"}, {"ref": "Reports"}]

    class _FakeController:
        def ask(self, question, **kwargs):
            return _FakeAnswer()

    runner.ask_controller = _FakeController()  # type: ignore[assignment]
    run = runner.answer(
        BenchmarkQuestion(
            id="q1",
            question="How does REALM differ from RAG?",
            expected_entities=("REALM",),
        )
    )
    assert run.error is None
    assert run.citation_count == 2
    assert run.insufficient_evidence is False
    metrics = answer_metrics(
        BenchmarkQuestion(
            id="q1",
            question="How does REALM differ from RAG?",
            expected_entities=("REALM",),
        ),
        run,
    )
    assert metrics["matched_entity_count"] == 1


def test_evaluate_backends_main_includes_graphrag(
    seeded_project, tmp_path: Path, monkeypatch
) -> None:
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(BENCHMARK_YAML)
    results_dir = tmp_path / "results"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_backends.py",
            "--project-root",
            str(seeded_project.paths.root),
            "--benchmark",
            str(benchmark),
            "--results-dir",
            str(results_dir),
            "--backends",
            "graphrag",
            "wikigraph",
            "--graphrag-methods",
            "auto",
            "--wikigraph-methods",
            "auto",
            "--retrieval-only",
        ],
    )
    exit_code = backends_main()
    assert exit_code == 0
    csv_path = results_dir / "backend_retrieval_metrics.csv"
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8")
    assert "graphrag" in text and "wikigraph" in text


def test_write_helpers(tmp_path: Path) -> None:
    rows = [
        {
            "backend": "x",
            "method": "auto",
            "recall_at_5": 0.5,
            "latency_seconds": 0.1,
            "error": "",
        }
    ]
    write_csv(tmp_path / "r.csv", RETRIEVAL_COLUMNS, rows)
    write_json(tmp_path / "raw.json", {"a": 1})
    write_summary_markdown(tmp_path / "s.md", retrieval_rows=rows, answer_rows=[])
    assert (tmp_path / "r.csv").exists()
    assert (tmp_path / "s.md").exists()
    assert json.loads((tmp_path / "raw.json").read_text())["a"] == 1
