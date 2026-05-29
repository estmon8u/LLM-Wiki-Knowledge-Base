"""Benchmark dataset loading for the RAG evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EvalQuestion:
    """A single benchmark question with ground truth for fair scoring."""

    id: str
    question: str
    category: str = "unspecified"
    expected_source_ids: tuple[str, ...] = ()
    expected_entities: tuple[str, ...] = ()
    reference_answer: str | None = None
    insufficient_expected: bool = False
    expected_methods: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalQuestion:
        """Build a question from a benchmark YAML mapping."""
        expected_sources = (
            payload.get("expected_source_ids") or payload.get("expected_sources") or []
        )
        behaviors = payload.get("expected_behaviors") or []
        insufficient = bool(
            payload.get("insufficient_evidence_expected", False)
            or "insufficient_evidence" in behaviors
            or payload.get("id") == "unsupported_claim"
        )
        reference = payload.get("reference_answer")
        return cls(
            id=str(payload["id"]),
            question=str(payload["question"]),
            category=str(payload.get("category", "unspecified")),
            expected_source_ids=tuple(str(s) for s in expected_sources),
            expected_entities=tuple(
                str(s) for s in (payload.get("expected_entities") or [])
            ),
            reference_answer=str(reference) if reference else None,
            insufficient_expected=insufficient,
            expected_methods={
                str(k): str(v)
                for k, v in (payload.get("expected_methods") or {}).items()
            },
        )


def load_benchmark(path: Path) -> list[EvalQuestion]:
    """Load benchmark questions from ``path`` (YAML)."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    questions = payload.get("questions") or []
    return [EvalQuestion.from_dict(item) for item in questions]
