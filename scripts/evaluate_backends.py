"""Backward-compatible wrapper for the RAG evaluation harness.

Historically the cross-backend evaluator was invoked as
``scripts/evaluate_backends.py --backends ...``. The implementation now lives in
``scripts.rag_eval``; this wrapper preserves the documented command surface while
delegating to the shared harness.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.rag_eval.cli import run_eval

_BACKEND_CHOICES = [
    "direct",
    "legacy",
    "graphrag",
    "wikigraph",
    "wikigraph-classic",
    "wikigraph-lightrag",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the compatibility CLI parser."""
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper around scripts/evaluate_rag.py."
    )
    parser.add_argument(
        "--benchmark", type=Path, default=Path("eval") / "benchmark.yaml"
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("eval") / "rag_eval")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["wikigraph", "graphrag", "legacy"],
        choices=_BACKEND_CHOICES,
    )
    parser.add_argument("--wikigraph-method", default="auto")
    parser.add_argument("--graphrag-method", default="auto")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--allow-provider-calls", action="store_true")
    parser.add_argument("--ragas", action="store_true")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--ragas-provider",
        default="openai",
        choices=["openai", "gemini"],
    )
    parser.add_argument("--ragas-model", default="gpt-5.4-nano")
    parser.add_argument("--ragas-embedding-model", default="text-embedding-3-large")
    parser.add_argument("--ragas-embedding-dimension", type=int, default=0)
    parser.add_argument("--ragas-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--label", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the shared RAG evaluator using legacy ``--backends`` naming."""
    args = build_parser().parse_args(argv)
    args.methods = list(args.backends)
    return run_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
