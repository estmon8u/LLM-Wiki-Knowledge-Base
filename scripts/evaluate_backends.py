"""Compare legacy/graphrag/wikigraph retrieval and answer quality.

This driver runs benchmark questions against any subset of:

* ``legacy``    -- the deprecated SQLite FTS5 path
* ``graphrag``  -- Microsoft GraphRAG through ``kb ask --engine graphrag``
* ``wikigraph`` -- the custom WikiGraphRAG backend

By default it compares WikiGraphRAG and GraphRAG retrieval. Provider-backed
answer generation is opt-in via ``--allow-provider-calls``; provider-free
WikiGraphRAG answer rows are marked as such and should not be treated as
like-for-like answer wins over skipped GraphRAG rows.

Sample invocations::

    poetry run python scripts/evaluate_backends.py --retrieval-only
    poetry run python scripts/evaluate_backends.py --retrieval-only \\
        --backends wikigraph graphrag legacy
    poetry run python scripts/evaluate_backends.py --backends wikigraph graphrag \\
        --allow-provider-calls
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.backend_evaluation_lib import (  # noqa: E402
    ANSWER_COLUMNS,
    ARTIFACTS_SUBDIR,
    DEFAULT_RESULTS_DIR,
    RETRIEVAL_COLUMNS,
    GraphRAGRunner,
    LegacyRunner,
    WikiGraphRunner,
    answer_metrics,
    build_command_context,
    load_benchmark,
    retrieval_metrics,
    write_csv,
    write_json,
    write_summary_markdown,
)

_DEFAULT_WIKIGRAPH_METHODS = ["auto"]
_DEFAULT_GRAPHRAG_METHODS = ["auto"]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run benchmark questions through WikiGraphRAG, Microsoft GraphRAG, "
            "and/or legacy FTS backends and write retrieval/answer metrics."
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
        help="Project root (defaults to the working directory).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for metrics output.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["legacy", "graphrag", "wikigraph"],
        default=["wikigraph", "graphrag"],
        help="Subset of backends to evaluate.",
    )
    parser.add_argument(
        "--wikigraph-methods",
        nargs="+",
        default=_DEFAULT_WIKIGRAPH_METHODS,
        help="Methods to evaluate against the wikigraph backend.",
    )
    parser.add_argument(
        "--graphrag-methods",
        nargs="+",
        default=_DEFAULT_GRAPHRAG_METHODS,
        help="Methods to evaluate against the Microsoft GraphRAG backend.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip provider-backed answer evaluation.",
    )
    parser.add_argument(
        "--allow-provider-calls",
        action="store_true",
        help="Permit provider-backed answer evaluation (cost!).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label appended to result filenames (e.g. ablation tag).",
    )
    return parser


def _result_paths(
    results_dir: Path, *, label: str | None
) -> tuple[Path, Path, Path, Path]:
    suffix = f"_{label}" if label else ""
    return (
        results_dir / f"backend_summary{suffix}.md",
        results_dir / f"backend_retrieval_metrics{suffix}.csv",
        results_dir / f"backend_answer_metrics{suffix}.csv",
        results_dir / ARTIFACTS_SUBDIR / f"backend_runs{suffix}.json",
    )


def main() -> int:
    """Entry point for ``python scripts/evaluate_backends.py``."""
    args = build_parser().parse_args()
    project_root = (args.project_root or Path.cwd()).resolve()
    context = build_command_context(project_root)
    questions = load_benchmark(args.benchmark)

    if not questions:
        print(f"No benchmark questions found in {args.benchmark}.")
        return 1

    summary_path, retrieval_csv, answer_csv, artifacts_path = _result_paths(
        args.results_dir, label=args.label
    )

    retrieval_rows: list[dict] = []
    answer_rows: list[dict] = []
    raw_payload: dict = {
        "project_root": str(project_root),
        "benchmark": str(args.benchmark),
        "backends": list(args.backends),
        "wikigraph_methods": list(args.wikigraph_methods),
        "graphrag_methods": list(args.graphrag_methods),
        "allow_provider_calls": bool(args.allow_provider_calls),
        "retrieval_only": bool(args.retrieval_only),
        "results": {"retrieval": [], "answers": []},
    }

    backends = []
    if "wikigraph" in args.backends:
        for method in args.wikigraph_methods:
            backends.append(WikiGraphRunner(context=context, method=method))
    if "graphrag" in args.backends:
        for method in args.graphrag_methods:
            backends.append(GraphRAGRunner(context=context, method=method))
    if "legacy" in args.backends:
        backends.append(LegacyRunner(context=context))

    for question in questions:
        for runner in backends:
            retrieval_run = runner.retrieve(question)
            metrics = retrieval_metrics(question, retrieval_run)
            row = {
                "question_id": question.id,
                "question": question.question,
                "backend": runner.name,
                "method": runner.method,
                **metrics,
            }
            retrieval_rows.append(row)
            raw_payload["results"]["retrieval"].append(
                {
                    "question": question.question,
                    "run": retrieval_run.__dict__,
                    "metrics": metrics,
                }
            )

            if args.retrieval_only:
                continue
            if runner.name in {"legacy", "graphrag"} and not args.allow_provider_calls:
                # Both require a real provider to produce an answer; skip
                # unless the user opts in to provider-backed evaluation.
                continue
            if runner.name == "wikigraph" and not args.allow_provider_calls:
                # WikiGraphRAG falls back to provider-free synthesis, so it
                # is safe to evaluate the answer path without a provider.
                pass
            answer_run = runner.answer(question)
            ametrics = answer_metrics(question, answer_run)
            arow = {
                "question_id": question.id,
                "question": question.question,
                "backend": runner.name,
                "method": runner.method,
                **ametrics,
            }
            answer_rows.append(arow)
            raw_payload["results"]["answers"].append(
                {
                    "question": question.question,
                    "run": answer_run.__dict__,
                    "metrics": ametrics,
                }
            )

    write_csv(retrieval_csv, RETRIEVAL_COLUMNS, retrieval_rows)
    if answer_rows:
        write_csv(answer_csv, ANSWER_COLUMNS, answer_rows)
    write_summary_markdown(
        summary_path,
        retrieval_rows=retrieval_rows,
        answer_rows=answer_rows,
    )
    write_json(artifacts_path, raw_payload)
    print(f"Wrote retrieval metrics to {retrieval_csv}")
    if answer_rows:
        print(f"Wrote answer metrics to {answer_csv}")
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
