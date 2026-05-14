"""Command-line script for evaluate retrieval.

This module belongs to `scripts.evaluate_retrieval` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""


from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation_lib import (
    EvaluationConfig,
    GRAPH_QUERY_METHODS,
    RESULTS_DIR,
    run_graph_modes_evaluation,
    write_csv,
    RETRIEVAL_COLUMNS,
)


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser.

    Returns:
        argparse.ArgumentParser produced by the operation.
    """
    parser = argparse.ArgumentParser(
        description="Run retrieval-focused Phase 8 metrics for legacy FTS and routing."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("eval") / "benchmark.yaml",
        help="Benchmark YAML file.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Knowledge-base project root.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for metrics output.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Top-k legacy retrieval results used for Recall@5 scoring.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=300,
        help="Per-command timeout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Runs the command-line entry point.

    Args:
        argv: Optional argument vector. Uses process arguments when omitted.

    Returns:
        int produced by the operation.
    """
    args = build_parser().parse_args(argv)
    result = run_graph_modes_evaluation(
        EvaluationConfig(
            benchmark_path=args.benchmark,
            project_root=args.project_root,
            results_dir=args.results_dir,
            limit=args.limit,
            allow_provider_calls=False,
            include_legacy_ask=False,
            graph_methods=tuple(GRAPH_QUERY_METHODS),
            timeout_seconds=args.timeout_seconds,
        )
    )
    write_csv(
        args.results_dir / "retrieval_metrics.csv",
        RETRIEVAL_COLUMNS,
        result.retrieval_rows,
    )
    print(f"Wrote {args.results_dir / 'retrieval_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
