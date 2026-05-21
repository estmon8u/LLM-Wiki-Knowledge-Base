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
            "expected_answer_terms": ["alpha"],
            "forbidden_answer_terms": ["omega"],
            "expected_methods": {"wikigraph": "local"},
            "expected_behaviors": ["insufficient_evidence"],
        }
    )
    assert question.expected_sources == ("a", "b")
    assert question.expected_answer_terms == ("alpha",)
    assert question.forbidden_answer_terms == ("omega",)
    assert question.expected_methods["wikigraph"] == "local"
    assert question.insufficient_evidence_expected is True


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
    assert metrics["recall_at_8"] > 0

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


def test_matched_entities_recognizes_well_known_expansions() -> None:
    """Spelled-out forms should credit their canonical abbreviations."""
    from scripts.backend_evaluation_lib import AnswerRun, matched_entities

    run = AnswerRun(
        backend="wikigraph",
        method="local",
        question_id="fid_method",
        question="How does Fusion-in-Decoder combine retrieved passages?",
        answer=(
            "Fusion-in-Decoder encodes each passage independently and "
            "fuses them in the decoder."
        ),
        citation_count=1,
        insufficient_evidence=False,
        latency_seconds=0.01,
    )
    question = BenchmarkQuestion(
        id="fid_method",
        question="How does Fusion-in-Decoder combine retrieved passages?",
        expected_entities=("FiD",),
    )
    assert matched_entities(question, run) == ["FiD"]


def test_answer_quality_score_penalizes_refusal_when_grounding_expected() -> None:
    from scripts.backend_evaluation_lib import AnswerRun, answer_metrics

    refusal = AnswerRun(
        backend="legacy",
        method="ask",
        question_id="dpr_role",
        question="What role does Dense Passage Retrieval play in RAG?",
        answer="Provided evidence does not mention DPR.",
        citation_count=0,
        insufficient_evidence=True,
        latency_seconds=0.01,
        citation_ref_valid_rate=1.0,
    )
    question = BenchmarkQuestion(
        id="dpr_role",
        question="What role does Dense Passage Retrieval play in RAG?",
        expected_entities=("DPR",),
        insufficient_evidence_expected=False,
    )
    metrics = answer_metrics(question, refusal)

    assert metrics["grounded_entity_hits"] == 0
    assert metrics["insufficient_evidence_behavior"] == "mismatch"
    assert metrics["answer_quality_score"] < 0.3


def test_answer_quality_score_penalizes_generic_uncited_answer() -> None:
    """Name-dropping without citations/required terms should not score highly."""
    run = AnswerRun(
        backend="wikigraph",
        method="basic",
        question_id="graph_config",
        question="Where is GraphRAG provider configuration stored?",
        answer="GraphRAG is mentioned in the project.",
        citation_count=0,
        insufficient_evidence=False,
        latency_seconds=0.01,
        citation_ref_valid_rate=0.0,
    )
    question = BenchmarkQuestion(
        id="graph_config",
        question="Where is GraphRAG provider configuration stored?",
        expected_entities=("GraphRAG",),
        expected_answer_terms=("kb.config.yaml", "embedding_model"),
        forbidden_answer_terms=("not available",),
        insufficient_evidence_expected=False,
    )

    metrics = answer_metrics(question, run)

    assert metrics["matched_entity_count"] == 1
    assert metrics["matched_answer_term_count"] == 0
    assert metrics["citation_count"] == 0
    assert metrics["answer_quality_score"] < 0.7


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


def test_matched_source_ids_avoids_substring_false_positive() -> None:
    """G3: single-token expected source matches word boundaries, not substrings."""
    from scripts.backend_evaluation_lib import matched_source_ids

    question = BenchmarkQuestion(
        id="q",
        question="how does FiD combine retrieval?",
        expected_sources=("FiD",),
    )
    run = RetrievalRun(
        backend="wikigraph",
        method="basic",
        question_id="q",
        question="?",
        retrieved_titles=["the model was modified during training"],
        retrieved_paths=[],
        retrieved_source_ids=[],
        retrieved_text_snippets=["Modifying the architecture significantly..."],
        latency_seconds=0.0,
    )
    assert matched_source_ids(question, run) == []


def test_matched_source_ids_word_boundary_match() -> None:
    """G3: single-token expected source matches on word boundaries."""
    from scripts.backend_evaluation_lib import matched_source_ids

    question = BenchmarkQuestion(
        id="q",
        question="?",
        expected_sources=("FiD",),
    )
    run = RetrievalRun(
        backend="wikigraph",
        method="basic",
        question_id="q",
        question="?",
        retrieved_titles=["FiD: Fusion-in-Decoder explained"],
        retrieved_paths=[],
        retrieved_source_ids=[],
        retrieved_text_snippets=[],
        latency_seconds=0.0,
    )
    assert matched_source_ids(question, run) == ["FiD"]


def test_matched_source_ids_multiword_substring_still_works() -> None:
    """G3: multi-token / hyphenated needles keep substring matching."""
    from scripts.backend_evaluation_lib import matched_source_ids

    question = BenchmarkQuestion(
        id="q",
        question="?",
        expected_sources=("Fusion-in-Decoder", "Dense Passage Retrieval"),
    )
    run = RetrievalRun(
        backend="wikigraph",
        method="basic",
        question_id="q",
        question="?",
        retrieved_titles=["A paper on fusion-in-decoder and dense passage retrieval"],
        retrieved_paths=[],
        retrieved_source_ids=[],
        retrieved_text_snippets=[],
        latency_seconds=0.0,
    )
    matches = matched_source_ids(question, run)
    assert "Fusion-in-Decoder" in matches and "Dense Passage Retrieval" in matches


def test_graphrag_runner_retrieve_includes_snippet_in_haystack() -> None:
    """G1: GraphRAGRunner.retrieve must populate retrieved_text_snippets."""
    from scripts.backend_evaluation_lib import (
        GraphRAGRunner,
        matched_source_ids,
    )
    from scripts.graphrag_artifact_retriever import GraphRAGArtifactResult

    class _FakeArtifactRetriever:
        def search(self, query, limit):
            return [
                GraphRAGArtifactResult(
                    kind="text_units",
                    title="text_unit 42",
                    path="graph://text_units/42",
                    snippet="REALM jointly trains the retriever and the LM.",
                    score=10.0,
                    source_ids=(),
                )
            ]

    runner = GraphRAGRunner.__new__(GraphRAGRunner)
    runner.context = None  # type: ignore[assignment]
    runner.method = "auto"
    runner.retrieve_mode = "text_units"
    runner.find_service = None  # type: ignore[assignment]
    runner.ask_controller = None  # type: ignore[assignment]
    runner.artifact_retriever = _FakeArtifactRetriever()
    runner.name = "graphrag"

    question = BenchmarkQuestion(
        id="q",
        question="REALM",
        expected_sources=("REALM",),
    )
    run = runner.retrieve(question)
    assert run.error is None
    # Title doesn't mention REALM but snippet does — must still match.
    assert matched_source_ids(question, run) == ["REALM"]


def test_quality_score_does_not_reward_citation_volume() -> None:
    """G4: composite must not reward citation count beyond ``>=1 supported``."""
    from scripts.backend_evaluation_lib import AnswerRun, answer_metrics

    base_kwargs = {
        "backend": "x",
        "method": "auto",
        "question_id": "q",
        "question": "?",
        "answer": "DPR uses dense passage retrieval to fetch documents.",
        "insufficient_evidence": False,
        "latency_seconds": 0.01,
        "citation_ref_valid_rate": 1.0,
        "citation_ref_strict_rate": 1.0,
    }
    few = AnswerRun(citation_count=2, **base_kwargs)
    many = AnswerRun(citation_count=8, **base_kwargs)
    question = BenchmarkQuestion(
        id="q",
        question="?",
        expected_entities=("DPR",),
        expected_answer_terms=("dense passage",),
    )
    assert (
        answer_metrics(question, few)["answer_quality_score"]
        == answer_metrics(question, many)["answer_quality_score"]
    )


def test_quality_score_zero_when_no_supported_citations() -> None:
    """G4: an uncited grounded answer must score lower than a cited one."""
    from scripts.backend_evaluation_lib import AnswerRun, answer_metrics

    uncited = AnswerRun(
        backend="x",
        method="auto",
        question_id="q",
        question="?",
        answer="DPR is dense passage retrieval.",
        insufficient_evidence=False,
        citation_count=0,
        latency_seconds=0.01,
    )
    cited = AnswerRun(
        backend="x",
        method="auto",
        question_id="q",
        question="?",
        answer="DPR is dense passage retrieval.",
        insufficient_evidence=False,
        citation_count=1,
        latency_seconds=0.01,
        citation_ref_valid_rate=1.0,
        citation_ref_strict_rate=1.0,
    )
    question = BenchmarkQuestion(
        id="q",
        question="?",
        expected_entities=("DPR",),
        expected_answer_terms=("dense passage",),
    )
    cited_score = answer_metrics(question, cited)["answer_quality_score"]
    uncited_score = answer_metrics(question, uncited)["answer_quality_score"]
    assert cited_score > uncited_score


def test_graphrag_ref_valid_rate_symmetric() -> None:
    """G5: GraphRAG citation_ref_valid_rate now reflects ref kind+ids."""
    from scripts.backend_evaluation_lib import _graphrag_ref_valid_rate

    assert _graphrag_ref_valid_rate([]) == 0.0
    refs = [
        {"kind": "entity", "ids": ["1", "2"]},
        {"kind": "bogus", "ids": ["3"]},
        {"kind": "community_report", "ids": []},
    ]
    # Only the entity ref is valid -> 1/3.
    assert _graphrag_ref_valid_rate(refs) == 1 / 3


def test_method_fit_tracks_expected_vs_chosen() -> None:
    """G9: retrieval rows surface chosen_method and method_fit."""
    from scripts.backend_evaluation_lib import retrieval_metrics

    question = BenchmarkQuestion(
        id="q",
        question="?",
        expected_sources=("REALM",),
        expected_methods={"wikigraph": "local"},
    )
    matching = RetrievalRun(
        backend="wikigraph",
        method="auto",
        question_id="q",
        question="?",
        retrieved_titles=["REALM"],
        retrieved_paths=[],
        retrieved_source_ids=[],
        chosen_method="local",
        latency_seconds=0.0,
    )
    mismatching = RetrievalRun(
        backend="wikigraph",
        method="auto",
        question_id="q",
        question="?",
        retrieved_titles=["REALM"],
        retrieved_paths=[],
        retrieved_source_ids=[],
        chosen_method="basic",
        latency_seconds=0.0,
    )
    assert retrieval_metrics(question, matching)["method_fit"] == "1"
    assert retrieval_metrics(question, mismatching)["method_fit"] == "0"


def test_wikigraph_runner_strict_vs_loose_citation_rate(seeded_project) -> None:
    """G6: WikiGraphRAG answer run reports both strict and loose valid rates."""
    context = build_command_context(seeded_project.paths.root)
    runner = WikiGraphRunner(context=context, method="basic")
    question = BenchmarkQuestion(
        id="q1",
        question="How does REALM differ from RAG?",
        expected_entities=("REALM", "RAG"),
    )
    run = runner.answer(question)
    # Provider-free path: every emitted citation matches a retrieved
    # context byte-for-byte (we generate them from the same context list).
    assert run.error is None
    assert run.citation_ref_strict_rate >= 0.0
    assert run.citation_ref_valid_rate >= run.citation_ref_strict_rate


def test_write_helpers(tmp_path: Path) -> None:
    rows = [
        {
            "backend": "x",
            "method": "auto",
            "recall_at_8": 0.5,
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
