"""Offline tests for aggregation + the retrieval-only harness end-to-end."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from scripts.rag_eval.aggregate import (
    bootstrap_ci,
    summarize,
    write_leaderboard_markdown,
)
from scripts.rag_eval.cli import main

REALM_PAGE = textwrap.dedent(
    """\
    ---
    title: REALM
    type: source
    source_id: realm
    summary: REALM pretrains a retriever with a masked language model.
    ---

    # REALM

    ## Summary

    REALM jointly trains a neural retriever with a masked language model.
    """
)
RAG_PAGE = textwrap.dedent(
    """\
    ---
    title: RAG
    type: source
    source_id: rag
    summary: RAG augments a seq2seq generator with retrieved passages.
    ---

    # RAG

    ## Summary

    RAG combines DPR retrieval with a seq2seq generator over retrieved documents.
    """
)

TINY_BENCHMARK = textwrap.dedent(
    """\
    version: 4
    name: tiny
    questions:
      - id: realm_q
        question: How does REALM train its retriever?
        category: local
        expected_sources: [REALM]
        reference_answer: REALM jointly trains a retriever with an MLM.
      - id: rag_q
        question: How does RAG use retrieved documents?
        category: local
        expected_sources: [RAG]
        reference_answer: RAG conditions a seq2seq generator on retrieved docs.
    """
)


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #


def test_bootstrap_ci_basic() -> None:
    mean, lo, hi = bootstrap_ci([1.0, 1.0, 1.0, 1.0], n_boot=100, seed=1)
    assert mean == 1.0 and lo == 1.0 and hi == 1.0
    mean, lo, hi = bootstrap_ci([0.0, 1.0], n_boot=200, seed=1)
    assert 0.0 <= lo <= mean <= hi <= 1.0
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)
    assert bootstrap_ci([0.5]) == (0.5, 0.5, 0.5)


def test_summarize_skips_none_and_groups_by_backend() -> None:
    rows = [
        {"backend": "a", "question_id": "1", "recall_at_k": 1.0},
        {"backend": "a", "question_id": "2", "recall_at_k": None},
        {"backend": "b", "question_id": "1", "recall_at_k": 0.0},
    ]
    summaries = summarize(rows, ["recall_at_k"], n_boot=50, seed=0)
    by_backend = {s.backend: s for s in summaries}
    assert by_backend["a"].n == 1 and by_backend["a"].mean == 1.0
    assert by_backend["b"].n == 1 and by_backend["b"].mean == 0.0


def test_leaderboard_markdown(tmp_path: Path) -> None:
    from scripts.rag_eval.aggregate import MetricSummary

    summaries = [
        MetricSummary("wikigraph", "recall_at_k", 5, 0.8, 0.6, 0.95),
        MetricSummary("legacy", "recall_at_k", 5, 0.5, 0.3, 0.7),
    ]
    path = tmp_path / "lb.md"
    write_leaderboard_markdown(
        path, summaries, metric_order=["recall_at_k"], notes=["caveat"]
    )
    text = path.read_text()
    assert "recall_at_k" in text and "wikigraph" in text and "caveat" in text


def test_evaluate_backends_wrapper_translates_backends(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.evaluate_backends as wrapper

    seen = {}

    def _fake_run_eval(args) -> int:
        seen["methods"] = args.methods
        seen["retrieval_only"] = args.retrieval_only
        seen["results_dir"] = args.results_dir
        return 0

    monkeypatch.setattr(wrapper, "run_eval", _fake_run_eval)

    code = wrapper.main(
        [
            "--backends",
            "wikigraph-classic",
            "wikigraph-lightrag",
            "--retrieval-only",
            "--results-dir",
            str(tmp_path),
        ]
    )

    assert code == 0
    assert seen == {
        "methods": ["wikigraph-classic", "wikigraph-lightrag"],
        "retrieval_only": True,
        "results_dir": tmp_path,
    }


def test_rag_eval_parser_accepts_gemini_ragas_options() -> None:
    from scripts.rag_eval.cli import build_parser

    args = build_parser().parse_args(
        [
            "--allow-provider-calls",
            "--ragas",
            "--ragas-provider",
            "gemini",
            "--ragas-model",
            "gemini-3.1-flash-lite-preview",
            "--ragas-embedding-model",
            "gemini-embedding-001",
            "--ragas-embedding-dimension",
            "768",
            "--ragas-api-key-env",
            "GEMINI_API_KEY",
        ]
    )

    assert args.ragas_provider == "gemini"
    assert args.ragas_model == "gemini-3.1-flash-lite-preview"
    assert args.ragas_embedding_model == "gemini-embedding-001"
    assert args.ragas_embedding_dimension == 768
    assert args.ragas_api_key_env == "GEMINI_API_KEY"


# --------------------------------------------------------------------------- #
# Retrieval-only end-to-end (offline; classic wikigraph + legacy)             #
# --------------------------------------------------------------------------- #


def _setup_project(tmp_path: Path) -> Path:
    from graphwiki_kb.services.config_service import ConfigService
    from graphwiki_kb.services.manifest_service import ManifestService
    from graphwiki_kb.services.project_service import (
        ProjectService,
        build_project_paths,
    )
    from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService

    paths = build_project_paths(tmp_path)
    ProjectService(paths).ensure_structure()
    ConfigService(paths).ensure_files()
    (paths.wiki_sources_dir / "realm.md").write_text(REALM_PAGE, encoding="utf-8")
    (paths.wiki_sources_dir / "rag.md").write_text(RAG_PAGE, encoding="utf-8")
    ManifestService(paths).ensure_manifest()
    config = ConfigService(paths).load()
    # Build the classic wikigraph index so retrieval returns contexts.
    WikiGraphIndexService(paths=paths, config=config).build()
    return paths.root


def test_retrieval_only_end_to_end(tmp_path: Path) -> None:
    root = _setup_project(tmp_path)
    benchmark = tmp_path / "tiny_benchmark.yaml"
    benchmark.write_text(TINY_BENCHMARK, encoding="utf-8")
    results_dir = tmp_path / "results"

    code = main(
        [
            "--project-root",
            str(root),
            "--benchmark",
            str(benchmark),
            "--results-dir",
            str(results_dir),
            "--methods",
            "wikigraph",
            "legacy",
            "--retrieval-only",
            "--bootstrap",
            "50",
        ]
    )
    assert code == 0
    leaderboard = results_dir / "rag_eval_leaderboard.md"
    rows_csv = results_dir / "rag_eval_rows.csv"
    summary_json = results_dir / "rag_eval_summary.json"
    assert leaderboard.exists() and rows_csv.exists() and summary_json.exists()
    payload = json.loads(summary_json.read_text())
    assert payload["retrieval_only"] is True
    backends = {s["backend"] for s in payload["summaries"]}
    assert "wikigraph" in backends
    # recall_at_k should be summarized for at least one backend.
    assert any(s["metric"] == "recall_at_k" for s in payload["summaries"])


def test_provider_gating_skips_provider_backends_without_flag(tmp_path: Path) -> None:
    root = _setup_project(tmp_path)
    benchmark = tmp_path / "tiny_benchmark.yaml"
    benchmark.write_text(TINY_BENCHMARK, encoding="utf-8")
    results_dir = tmp_path / "results2"
    # Full run (not retrieval-only) WITHOUT --allow-provider-calls: direct/legacy/
    # graphrag are skipped; wikigraph runs provider-free.
    code = main(
        [
            "--project-root",
            str(root),
            "--benchmark",
            str(benchmark),
            "--results-dir",
            str(results_dir),
            "--methods",
            "direct",
            "legacy",
            "graphrag",
            "wikigraph",
            "--bootstrap",
            "50",
        ]
    )
    assert code == 0
    payload = json.loads((results_dir / "rag_eval_summary.json").read_text())
    backends = {s["backend"] for s in payload["summaries"]}
    assert backends == {"wikigraph"}
