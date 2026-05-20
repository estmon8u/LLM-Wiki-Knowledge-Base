"""Evaluate legacy, GraphRAG, and WikiGraphRAG backends."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backend_evaluation_lib import (
    WIKIGRAPH_METHODS,
    BackendEvaluationConfig,
    run_backend_evaluation,
)
from scripts.evaluation_lib import GRAPH_QUERY_METHODS, RESULTS_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare deprecated legacy FTS, Microsoft GraphRAG, and custom "
            "WikiGraphRAG over the same benchmark."
        )
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("eval") / "benchmark.yaml",
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--backends", nargs="+", default=["legacy", "wikigraph"])
    parser.add_argument(
        "--graphrag-methods",
        nargs="+",
        choices=GRAPH_QUERY_METHODS,
        default=list(GRAPH_QUERY_METHODS),
    )
    parser.add_argument(
        "--wikigraph-methods",
        nargs="+",
        choices=WIKIGRAPH_METHODS,
        default=list(WIKIGRAPH_METHODS),
    )
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--allow-provider-calls", action="store_true")
    parser.add_argument("--include-legacy-ask", action="store_true")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BackendEvaluationConfig(
        benchmark_path=args.benchmark,
        project_root=args.project_root,
        results_dir=args.results_dir,
        limit=args.limit,
        allow_provider_calls=args.allow_provider_calls,
        include_legacy_ask=args.include_legacy_ask,
        backends=tuple(args.backends),
        graphrag_methods=tuple(args.graphrag_methods),
        wikigraph_methods=tuple(args.wikigraph_methods),
        retrieval_only=args.retrieval_only,
        timeout_seconds=args.timeout_seconds,
        label=args.label,
    )
    run_backend_evaluation(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
