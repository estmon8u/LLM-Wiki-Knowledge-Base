"""Command-line script for evaluate answers.

This module belongs to `scripts.evaluate_answers` and keeps related behavior
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
    ANSWER_COLUMNS,
    EvaluationConfig,
    GRAPH_QUERY_METHODS,
    RESULTS_DIR,
    run_graph_modes_evaluation,
    write_csv,
)


def build_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser.

    Returns:
        argparse.ArgumentParser produced by the operation.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run answer-focused Phase 8 metrics for GraphRAG modes and optional "
            "legacy ask comparison."
        )
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
        "--allow-provider-calls",
        action="store_true",
        help="Allow provider-backed GraphRAG query and legacy ask calls.",
    )
    parser.add_argument(
        "--include-legacy-ask",
        action="store_true",
        help="Include deprecated kb legacy ask rows.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=GRAPH_QUERY_METHODS,
        default=list(GRAPH_QUERY_METHODS),
        help="GraphRAG query methods to evaluate.",
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
            limit=5,
            allow_provider_calls=args.allow_provider_calls,
            include_legacy_ask=args.include_legacy_ask,
            graph_methods=tuple(args.methods),
            timeout_seconds=args.timeout_seconds,
        )
    )
    write_csv(
        args.results_dir / "answer_metrics.csv", ANSWER_COLUMNS, result.answer_rows
    )
    print(f"Wrote {args.results_dir / 'answer_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
