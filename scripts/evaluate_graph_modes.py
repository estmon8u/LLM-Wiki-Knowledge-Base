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
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate deprecated legacy FTS retrieval against GraphRAG Basic, "
            "Local, Global, and DRIFT query modes."
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
        help=(
            "Knowledge-base project root. Defaults to the benchmark source_project "
            "when present."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for summary, CSV metrics, and JSON artifacts.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Top-k legacy retrieval results used for Recall@5 scoring.",
    )
    parser.add_argument(
        "--allow-provider-calls",
        action="store_true",
        help=(
            "Allow provider-backed legacy ask and GraphRAG query calls. Without "
            "this flag, the evaluator only runs local-safe retrieval/router checks."
        ),
    )
    parser.add_argument(
        "--include-legacy-ask",
        action="store_true",
        help=(
            "Include deprecated provider-backed kb legacy ask rows. Requires "
            "--allow-provider-calls to execute instead of skip."
        ),
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
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_graph_modes_evaluation(
        EvaluationConfig(
            benchmark_path=args.benchmark,
            project_root=args.project_root,
            results_dir=args.results_dir,
            limit=args.limit,
            allow_provider_calls=args.allow_provider_calls,
            include_legacy_ask=args.include_legacy_ask,
            graph_methods=tuple(args.methods),
            timeout_seconds=args.timeout_seconds,
        )
    )
    print(f"Wrote {args.results_dir / 'summary.md'}")
    print(f"Retrieval rows: {len(result.retrieval_rows)}")
    print(f"Answer rows: {len(result.answer_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
