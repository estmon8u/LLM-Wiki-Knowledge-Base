"""Tests for three-backend evaluation scripts."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.backend_evaluation_lib import (
    BackendEvaluationConfig,
    evaluate_wikigraph_ask,
    evaluate_wikigraph_find,
    run_backend_evaluation,
)
from scripts.evaluation_lib import BenchmarkQuestion, CommandRun


def test_evaluate_wikigraph_find_skipped_without_project() -> None:
    question = BenchmarkQuestion(
        id="x",
        question="test",
        intent="",
        category="",
        expected_method=None,
        expected_sources=(),
        expected_behaviors=(),
    )
    row = evaluate_wikigraph_find(
        question,
        method="basic",
        project_root=None,
        command_cwd=Path(),
        results_dir=Path("eval/results"),
        limit=5,
        timeout_seconds=30,
    )
    assert row["status"] == "skipped"


def test_evaluate_wikigraph_find_failure(tmp_path: Path) -> None:
    question = BenchmarkQuestion(
        id="x",
        question="test",
        intent="",
        category="",
        expected_method=None,
        expected_sources=(),
        expected_behaviors=(),
    )

    def fake_run_command(command, *, cwd, timeout_seconds):
        return CommandRun(
            command=tuple(command),
            returncode=1,
            stdout="",
            stderr="index missing",
            latency_seconds=0.01,
        )

    import scripts.backend_evaluation_lib as backend_lib

    backend_lib.run_command = fake_run_command
    row = evaluate_wikigraph_find(
        question,
        method="basic",
        project_root=tmp_path,
        command_cwd=tmp_path,
        results_dir=tmp_path / "results",
        limit=5,
        timeout_seconds=30,
    )
    assert row["status"] == "failed"


def test_evaluate_wikigraph_find_parses_payload(tmp_path: Path) -> None:
    question = BenchmarkQuestion(
        id="realm_vs_rag",
        question="How does REALM differ from RAG?",
        intent="comparison",
        category="comparison",
        expected_method="drift-lite",
        expected_sources=("REALM", "RAG"),
        expected_behaviors=(),
    )
    stdout = json.dumps(
        {
            "contexts": [
                {
                    "title": "REALM",
                    "path": "wiki/sources/realm.md",
                    "node_id": "chunk:wiki/sources/realm.md#overview",
                    "trace": ["lexical:wiki/sources/realm.md#overview"],
                    "source_ids": ["realm"],
                }
            ],
            "trace": [{"step": "method", "value": "local"}],
        }
    )

    def fake_run_command(command, *, cwd, timeout_seconds):
        return CommandRun(
            command=tuple(command),
            returncode=0,
            stdout=stdout,
            stderr="",
            latency_seconds=0.1,
        )

    import scripts.backend_evaluation_lib as backend_lib

    backend_lib.run_command = fake_run_command
    row = evaluate_wikigraph_find(
        question,
        method="local",
        project_root=tmp_path,
        command_cwd=tmp_path,
        results_dir=tmp_path / "results",
        limit=5,
        timeout_seconds=30,
    )
    assert row["status"] == "ok"
    assert row["retrieved_context_count"] == 1


def test_evaluate_wikigraph_ask_extractive(tmp_path: Path) -> None:
    question = BenchmarkQuestion(
        id="realm",
        question="What is REALM?",
        intent="local",
        category="local",
        expected_method="local",
        expected_sources=("REALM",),
        expected_behaviors=(),
    )

    def fake_run_command(command, *, cwd, timeout_seconds):
        return CommandRun(
            command=tuple(command),
            returncode=0,
            stdout='{"answer": "REALM uses retrieval.", "contexts": [{"source_ids": ["realm"]}]}',
            stderr="",
            latency_seconds=0.2,
        )

    import scripts.backend_evaluation_lib as backend_lib

    backend_lib.run_command = fake_run_command
    row = backend_lib.evaluate_wikigraph_ask(
        question,
        method="local",
        project_root=tmp_path,
        command_cwd=tmp_path,
        results_dir=tmp_path / "results",
        allow_provider_calls=False,
        timeout_seconds=30,
    )
    assert row["status"] == "ok"
    assert row["wikigraph_context_count"] == 1


def test_evaluate_wikigraph_ask_skipped_with_provider_flag(tmp_path: Path) -> None:
    question = BenchmarkQuestion(
        id="q",
        question="test",
        intent="",
        category="",
        expected_method="basic",
        expected_sources=(),
        expected_behaviors=(),
    )
    row = evaluate_wikigraph_ask(
        question,
        method="basic",
        project_root=tmp_path,
        command_cwd=tmp_path,
        results_dir=tmp_path / "results",
        allow_provider_calls=True,
        timeout_seconds=30,
    )
    assert row["status"] == "skipped"


def test_run_backend_evaluation_writes_outputs(
    tmp_path: Path, test_project, monkeypatch
) -> None:
    benchmark_path = tmp_path / "benchmark.yaml"
    benchmark_path.write_text(
        """
version: 2
name: backend-eval
description: test
questions:
  - id: realm_vs_rag
    question: How does REALM differ from RAG?
    expected_method: drift-lite
    expected_sources:
      - REALM
      - RAG
""".strip(),
        encoding="utf-8",
    )

    def fake_run_command(command, *, cwd, timeout_seconds):
        return CommandRun(
            command=tuple(command),
            returncode=0,
            stdout=json.dumps({"contexts": [], "trace": []}),
            stderr="",
            latency_seconds=0.05,
        )

    import scripts.backend_evaluation_lib as backend_lib
    import scripts.evaluation_lib as evaluation_lib

    monkeypatch.setattr(backend_lib, "run_command", fake_run_command)
    monkeypatch.setattr(evaluation_lib, "run_command", fake_run_command)

    result = run_backend_evaluation(
        BackendEvaluationConfig(
            benchmark_path=benchmark_path,
            project_root=test_project.root,
            results_dir=tmp_path / "results",
            limit=5,
            allow_provider_calls=False,
            include_legacy_ask=False,
            backends=("wikigraph",),
            graphrag_methods=("basic",),
            wikigraph_methods=("basic",),
            retrieval_only=True,
            timeout_seconds=30,
        )
    )
    assert result.retrieval_rows
    assert (tmp_path / "results" / "backend_summary.md").exists()
