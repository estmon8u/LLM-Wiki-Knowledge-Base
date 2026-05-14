"""Tests for test evaluation scripts.

This module belongs to `tests.test_evaluation_scripts` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""


from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts import evaluation_lib
from scripts.evaluation_lib import CommandRun, EvaluationConfig


def test_score_expected_source_coverage_matches_nested_payloads() -> None:
    """Verifies that score expected source coverage matches nested payloads."""
    score = evaluation_lib.score_expected_source_coverage(
        ["REALM", "RAG"],
        [
            {
                "title": "REALM paper",
                "path": "wiki/sources/realm.md",
                "snippet": "REALM augments language model pretraining.",
            },
            {
                "title": "RAG paper",
                "path": "wiki/sources/retrieval-augmented-generation.md",
                "snippet": "RAG conditions generation on retrieved passages.",
            },
        ],
    )

    assert score == {
        "expected_count": 2,
        "matched_count": 2,
        "recall": 1.0,
        "multi_source_coverage": 1,
    }


def test_benchmark_source_project_resolves_from_repo_sibling(tmp_path: Path) -> None:
    """Verifies that benchmark source project resolves from repo sibling.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    repo_root = tmp_path / "repo"
    benchmark_path = repo_root / "eval" / "benchmark.yaml"
    benchmark_path.parent.mkdir(parents=True)
    source_project = tmp_path / "real-project"
    source_project.mkdir()
    benchmark_path.write_text(
        """
version: 2
name: test-eval
description: Test benchmark
source_project: real-project
questions:
  - id: route
    question: What are the main themes across the corpus?
    expected_method: global
""".strip(),
        encoding="utf-8",
    )

    benchmark = evaluation_lib.load_benchmark(benchmark_path)

    assert benchmark.path == benchmark_path.resolve()
    assert evaluation_lib.resolve_project_root(benchmark, None) == source_project


def test_graph_modes_evaluation_writes_safe_baseline_outputs(
    tmp_path: Path, test_project, monkeypatch
) -> None:
    """Verifies that graph modes evaluation writes safe baseline outputs.

    Args:
        tmp_path: Tmp path value used by the operation.
        test_project: Test project value used by the operation.
        monkeypatch: Monkeypatch value used by the operation.
    """
    benchmark_path = tmp_path / "benchmark.yaml"
    benchmark_path.write_text(
        """
version: 2
name: test-eval
description: Test benchmark
questions:
  - id: realm_vs_rag
    question: How does REALM differ from RAG?
    expected_method: drift
    expected_sources:
      - REALM
      - RAG
  - id: unsupported_claim
    question: What does the corpus say about unknown retrieval?
    expected_method: basic
    expected_behaviors:
      - insufficient_evidence
""".strip(),
        encoding="utf-8",
    )

    def fake_run_command(command, *, cwd, timeout_seconds):
        """Fake run command.

        Args:
            command: Command value used by the operation.
            cwd: Cwd value used by the operation.
            timeout_seconds: Timeout seconds value used by the operation.
        """
        return CommandRun(
            command=tuple(command),
            returncode=0,
            stdout=json.dumps(
                {
                    "results": [
                        {
                            "title": "REALM and RAG",
                            "path": "wiki/sources/realm-rag.md",
                            "snippet": "REALM and RAG both use retrieval.",
                        }
                    ]
                }
            ),
            stderr="Deprecated: legacy path",
            latency_seconds=0.25,
        )

    monkeypatch.setattr(evaluation_lib, "run_command", fake_run_command)

    result = evaluation_lib.run_graph_modes_evaluation(
        EvaluationConfig(
            benchmark_path=benchmark_path,
            project_root=test_project.root,
            results_dir=tmp_path / "results",
            limit=5,
            allow_provider_calls=False,
            include_legacy_ask=True,
            graph_methods=("basic", "drift"),
            timeout_seconds=30,
        )
    )

    assert len(result.retrieval_rows) == 4
    assert len(result.answer_rows) == 6
    assert (tmp_path / "results" / "summary.md").exists()
    assert (tmp_path / "results" / "retrieval_metrics.csv").exists()
    assert (tmp_path / "results" / "answer_metrics.csv").exists()
    assert "provider-backed answer commands were skipped" in (
        tmp_path / "results" / "summary.md"
    ).read_text(encoding="utf-8")

    with (tmp_path / "results" / "retrieval_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    legacy_row = next(row for row in rows if row["retriever"] == "legacy-fts")
    assert legacy_row["status"] == "ok"
    assert legacy_row["recall_at_5"] == "1.0"
    assert legacy_row["multi_source_coverage"] == "1"
    assert all(row["status"] == "skipped_provider_call" for row in result.answer_rows)


def test_graph_method_provider_payload_scores_claim_support(
    tmp_path: Path, test_project, monkeypatch
) -> None:
    """Verifies that graph method provider payload scores claim support.

    Args:
        tmp_path: Tmp path value used by the operation.
        test_project: Test project value used by the operation.
        monkeypatch: Monkeypatch value used by the operation.
    """
    question = evaluation_lib.BenchmarkQuestion(
        id="realm_vs_rag",
        question="How does REALM differ from RAG?",
        intent="",
        category="comparison",
        expected_method="drift",
        expected_sources=("REALM", "RAG"),
        expected_behaviors=(),
    )

    def fake_run_command(command, *, cwd, timeout_seconds):
        """Fake run command.

        Args:
            command: Command value used by the operation.
            cwd: Cwd value used by the operation.
            timeout_seconds: Timeout seconds value used by the operation.
        """
        return CommandRun(
            command=tuple(command),
            returncode=0,
            stdout=json.dumps(
                {
                    "answer": "REALM and RAG are both retrieval systems.",
                    "claim_support": "graph-grounded",
                }
            ),
            stderr="",
            latency_seconds=1.5,
        )

    monkeypatch.setattr(evaluation_lib, "run_command", fake_run_command)

    row = evaluation_lib.evaluate_graph_method(
        question,
        method="drift",
        project_root=test_project.root,
        results_dir=tmp_path / "results",
        allow_provider_calls=True,
        timeout_seconds=30,
    )

    assert row["status"] == "ok"
    assert row["claim_support"] == "graph-grounded"
    assert row["claim_support_rate"] == 1.0
    assert row["diversity"] == 1.0
