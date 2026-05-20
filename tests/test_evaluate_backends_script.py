"""Tests for scripts/evaluate_backends.py entrypoint."""

from __future__ import annotations

from pathlib import Path


def test_evaluate_backends_main_retrieval_only(
    tmp_path: Path, test_project, monkeypatch
) -> None:
    benchmark_path = tmp_path / "benchmark.yaml"
    benchmark_path.write_text(
        """
version: 2
name: backend-cli
description: test
questions:
  - id: q1
    question: What is REALM?
    expected_sources:
      - REALM
""".strip(),
        encoding="utf-8",
    )
    results_dir = tmp_path / "results"

    from scripts import evaluate_backends

    exit_code = evaluate_backends.main(
        [
            "--benchmark",
            str(benchmark_path),
            "--project-root",
            str(test_project.root),
            "--results-dir",
            str(results_dir),
            "--backends",
            "wikigraph",
            "--wikigraph-methods",
            "basic",
            "--retrieval-only",
        ]
    )
    assert exit_code == 0
    assert (results_dir / "backend_summary.md").exists()
