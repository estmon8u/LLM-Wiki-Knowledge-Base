"""Reusable evaluator primitives for comparing retrieval/answer backends.

This module supports the ``scripts/evaluate_backends.py`` driver. It is
deliberately lightweight and provider-free by default; provider-backed
answer generation is opt-in via ``--allow-provider-calls``.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from graphwiki_kb.models.command_models import CommandContext
from graphwiki_kb.services import build_services
from graphwiki_kb.services.config_service import ConfigService
from graphwiki_kb.services.project_service import build_project_paths
from graphwiki_kb.wikigraph.models import WikiGraphAnswer, WikiGraphFindResult

DEFAULT_RESULTS_DIR = Path("eval") / "results"
ARTIFACTS_SUBDIR = "artifacts"


@dataclass(frozen=True)
class BenchmarkQuestion:
    """Lightweight benchmark question for backend comparison."""

    id: str
    question: str
    category: str = "unspecified"
    expected_sources: tuple[str, ...] = ()
    expected_entities: tuple[str, ...] = ()
    expected_methods: dict[str, str] = field(default_factory=dict)
    insufficient_evidence_expected: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BenchmarkQuestion:
        """Build a question from a YAML mapping."""
        expected_methods = payload.get("expected_methods") or {}
        return cls(
            id=str(payload["id"]),
            question=str(payload["question"]),
            category=str(payload.get("category", "unspecified")),
            expected_sources=tuple(
                str(s) for s in payload.get("expected_sources", []) or []
            ),
            expected_entities=tuple(
                str(s) for s in payload.get("expected_entities", []) or []
            ),
            expected_methods={str(k): str(v) for k, v in expected_methods.items()},
            insufficient_evidence_expected=bool(
                payload.get("insufficient_evidence_expected", False)
            ),
        )


@dataclass
class RetrievalRun:
    """Outcome of a single retrieval invocation."""

    backend: str
    method: str
    question_id: str
    question: str
    retrieved_titles: list[str]
    retrieved_paths: list[str]
    retrieved_source_ids: list[str]
    latency_seconds: float
    artifact_path: str | None = None
    error: str | None = None


@dataclass
class AnswerRun:
    """Outcome of a single answer invocation."""

    backend: str
    method: str
    question_id: str
    question: str
    answer: str
    insufficient_evidence: bool
    citation_count: int
    latency_seconds: float
    artifact_path: str | None = None
    error: str | None = None


def load_benchmark(path: Path) -> list[BenchmarkQuestion]:
    """Load benchmark questions from ``path``."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    questions = payload.get("questions") or []
    return [BenchmarkQuestion.from_dict(item) for item in questions]


def build_command_context(project_root: Path) -> CommandContext:
    """Construct a CommandContext for evaluator runs."""
    paths = build_project_paths(project_root)
    config_service = ConfigService(paths)
    config = config_service.load()
    schema_text = config_service.load_schema()
    services = build_services(paths, config)
    return CommandContext(
        project_root=paths.root,
        cwd=paths.root,
        config=config,
        schema_text=schema_text,
        services=services,
        verbose=False,
    )


# --------------------------------------------------------------------------- #
# Backend runners                                                             #
# --------------------------------------------------------------------------- #


class WikiGraphRunner:
    """Backend runner for the custom WikiGraphRAG pipeline."""

    name = "wikigraph"

    def __init__(self, context: CommandContext, *, method: str = "auto") -> None:
        self.context = context
        self.method = method
        self.query_service = context.services.wikigraph_query

    def retrieve(self, question: BenchmarkQuestion) -> RetrievalRun:
        """Execute provider-free retrieval."""
        start = time.perf_counter()
        try:
            result = self.query_service.find(question.question, method=self.method)
            elapsed = time.perf_counter() - start
            return RetrievalRun(
                backend=self.name,
                method=self.method,
                question_id=question.id,
                question=question.question,
                retrieved_titles=[ctx.title for ctx in result.contexts],
                retrieved_paths=[ctx.path or "" for ctx in result.contexts],
                retrieved_source_ids=_flatten_source_ids(result),
                latency_seconds=elapsed,
            )
        except Exception as exc:
            return RetrievalRun(
                backend=self.name,
                method=self.method,
                question_id=question.id,
                question=question.question,
                retrieved_titles=[],
                retrieved_paths=[],
                retrieved_source_ids=[],
                latency_seconds=time.perf_counter() - start,
                error=str(exc),
            )

    def answer(self, question: BenchmarkQuestion) -> AnswerRun:
        """Run a full WikiGraphRAG answer."""
        start = time.perf_counter()
        try:
            ans: WikiGraphAnswer = self.query_service.ask(
                question.question, method=self.method
            )
            elapsed = time.perf_counter() - start
            return AnswerRun(
                backend=self.name,
                method=self.method,
                question_id=question.id,
                question=question.question,
                answer=ans.answer,
                insufficient_evidence=ans.insufficient_evidence,
                citation_count=len(ans.citations),
                latency_seconds=elapsed,
            )
        except Exception as exc:
            return AnswerRun(
                backend=self.name,
                method=self.method,
                question_id=question.id,
                question=question.question,
                answer="",
                insufficient_evidence=True,
                citation_count=0,
                latency_seconds=time.perf_counter() - start,
                error=str(exc),
            )


class GraphRAGRunner:
    """Backend runner for Microsoft GraphRAG (kb ask / kb find paths)."""

    name = "graphrag"

    def __init__(self, context: CommandContext, *, method: str = "auto") -> None:
        self.context = context
        self.method = method
        self.find_service = context.services.graphrag_find
        self.ask_controller = context.services.graph_ask_controller

    def retrieve(self, question: BenchmarkQuestion) -> RetrievalRun:
        """Execute provider-free retrieval via the GraphRAG find service."""
        start = time.perf_counter()
        try:
            results = self.find_service.search(question.question, limit=8)
            elapsed = time.perf_counter() - start
            return RetrievalRun(
                backend=self.name,
                method="find",
                question_id=question.id,
                question=question.question,
                retrieved_titles=[r.title for r in results],
                retrieved_paths=[str(r.path) for r in results],
                retrieved_source_ids=[],
                latency_seconds=elapsed,
            )
        except Exception as exc:
            return RetrievalRun(
                backend=self.name,
                method="find",
                question_id=question.id,
                question=question.question,
                retrieved_titles=[],
                retrieved_paths=[],
                retrieved_source_ids=[],
                latency_seconds=time.perf_counter() - start,
                error=str(exc),
            )

    def answer(self, question: BenchmarkQuestion) -> AnswerRun:
        """Run the GraphRAG-aware ask controller for a benchmark question."""
        start = time.perf_counter()
        try:
            answer = self.ask_controller.ask(question.question, method=self.method)
            elapsed = time.perf_counter() - start
            insufficient = (answer.claim_support or "").lower() in {
                "no-answer",
                "insufficient-evidence",
                "stale-index",
            }
            return AnswerRun(
                backend=self.name,
                method=answer.method or self.method,
                question_id=question.id,
                question=question.question,
                answer=answer.answer or "",
                insufficient_evidence=insufficient,
                citation_count=len(answer.graph_data_references or []),
                latency_seconds=elapsed,
            )
        except Exception as exc:
            return AnswerRun(
                backend=self.name,
                method=self.method,
                question_id=question.id,
                question=question.question,
                answer="",
                insufficient_evidence=True,
                citation_count=0,
                latency_seconds=time.perf_counter() - start,
                error=str(exc),
            )


class LegacyRunner:
    """Backend runner for the deprecated SQLite FTS5 wiki retrieval path."""

    name = "legacy"

    def __init__(self, context: CommandContext) -> None:
        self.context = context
        self.method = "ask"

    def retrieve(self, question: BenchmarkQuestion) -> RetrievalRun:
        """Run a legacy FTS retrieval."""
        start = time.perf_counter()
        search = self.context.services.search
        try:
            results = search.search(
                question.question,
                limit=8,
                include_analysis=False,
                page_types={"source"},
            )
            elapsed = time.perf_counter() - start
            return RetrievalRun(
                backend=self.name,
                method="find",
                question_id=question.id,
                question=question.question,
                retrieved_titles=[result.title for result in results],
                retrieved_paths=[result.path for result in results],
                retrieved_source_ids=[],
                latency_seconds=elapsed,
            )
        except Exception as exc:
            return RetrievalRun(
                backend=self.name,
                method="find",
                question_id=question.id,
                question=question.question,
                retrieved_titles=[],
                retrieved_paths=[],
                retrieved_source_ids=[],
                latency_seconds=time.perf_counter() - start,
                error=str(exc),
            )

    def answer(self, question: BenchmarkQuestion) -> AnswerRun:
        """Run a legacy FTS answer (provider-backed)."""
        start = time.perf_counter()
        try:
            answer = self.context.services.query.answer_question(question.question)
            elapsed = time.perf_counter() - start
            return AnswerRun(
                backend=self.name,
                method="ask",
                question_id=question.id,
                question=question.question,
                answer=answer.answer,
                insufficient_evidence=answer.insufficient_evidence,
                citation_count=len(answer.citations),
                latency_seconds=elapsed,
            )
        except Exception as exc:
            return AnswerRun(
                backend=self.name,
                method="ask",
                question_id=question.id,
                question=question.question,
                answer="",
                insufficient_evidence=True,
                citation_count=0,
                latency_seconds=time.perf_counter() - start,
                error=str(exc),
            )


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #


def _flatten_source_ids(result: WikiGraphFindResult) -> list[str]:
    out: list[str] = []
    for ctx in result.contexts:
        for sid in ctx.source_ids:
            if sid not in out:
                out.append(sid)
    return out


def matched_source_ids(question: BenchmarkQuestion, run: RetrievalRun) -> list[str]:
    """Return the expected source ids that appear in ``run``."""
    if not question.expected_sources:
        return []
    haystack = " ".join(
        [
            *run.retrieved_titles,
            *run.retrieved_paths,
            *run.retrieved_source_ids,
        ]
    ).lower()
    return [
        expected
        for expected in question.expected_sources
        if expected.lower() in haystack
    ]


def matched_entities(question: BenchmarkQuestion, run: AnswerRun) -> list[str]:
    """Return the expected entities that appear in ``run.answer``."""
    if not question.expected_entities:
        return []
    text = run.answer.lower()
    return [entity for entity in question.expected_entities if entity.lower() in text]


def retrieval_metrics(question: BenchmarkQuestion, run: RetrievalRun) -> dict[str, Any]:
    """Compute simple retrieval metrics for one (question, run) pair."""
    matched = matched_source_ids(question, run)
    expected = len(question.expected_sources)
    recall = (len(matched) / expected) if expected else 0.0
    return {
        "matched_source_count": len(matched),
        "expected_source_count": expected,
        "recall_at_5": recall,
        "retrieved_count": len(run.retrieved_titles),
        "latency_seconds": run.latency_seconds,
        "error": run.error,
    }


def answer_metrics(question: BenchmarkQuestion, run: AnswerRun) -> dict[str, Any]:
    """Compute simple answer metrics for one (question, run) pair."""
    entity_hits = matched_entities(question, run)
    expected_insufficient = question.insufficient_evidence_expected
    behavior = (
        "matches_expectation"
        if run.insufficient_evidence == expected_insufficient
        else "mismatch"
    )
    return {
        "answer_length": len(run.answer or ""),
        "citation_count": run.citation_count,
        "matched_entity_count": len(entity_hits),
        "expected_entity_count": len(question.expected_entities),
        "insufficient_evidence_expected": expected_insufficient,
        "insufficient_evidence_observed": run.insufficient_evidence,
        "insufficient_evidence_behavior": behavior,
        "latency_seconds": run.latency_seconds,
        "error": run.error,
    }


# --------------------------------------------------------------------------- #
# IO helpers                                                                  #
# --------------------------------------------------------------------------- #


RETRIEVAL_COLUMNS = (
    "question_id",
    "question",
    "backend",
    "method",
    "matched_source_count",
    "expected_source_count",
    "recall_at_5",
    "retrieved_count",
    "latency_seconds",
    "error",
)

ANSWER_COLUMNS = (
    "question_id",
    "question",
    "backend",
    "method",
    "answer_length",
    "citation_count",
    "matched_entity_count",
    "expected_entity_count",
    "insufficient_evidence_expected",
    "insufficient_evidence_observed",
    "insufficient_evidence_behavior",
    "latency_seconds",
    "error",
)


def write_csv(
    path: Path,
    columns: Iterable[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    """Write ``rows`` to ``path`` with ``columns`` as header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_summary_markdown(
    path: Path,
    *,
    retrieval_rows: list[dict[str, Any]],
    answer_rows: list[dict[str, Any]],
) -> None:
    """Write a small human-readable backend comparison summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Backend evaluation summary", ""]
    if retrieval_rows:
        lines.append("## Retrieval metrics (per backend, averaged)")
        lines.append("")
        lines.append("| Backend | Method | Avg Recall@5 | Avg Latency (s) | Errors |")
        lines.append("|---|---|---|---|---|")
        per_backend: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in retrieval_rows:
            per_backend.setdefault((row["backend"], row["method"]), []).append(row)
        for (backend, method), rows in sorted(per_backend.items()):
            recall = sum(float(row.get("recall_at_5", 0) or 0) for row in rows) / len(
                rows
            )
            latency = sum(
                float(row.get("latency_seconds", 0) or 0) for row in rows
            ) / len(rows)
            errors = sum(1 for row in rows if row.get("error"))
            lines.append(
                f"| {backend} | {method} | {recall:.3f} | {latency:.3f} | {errors} |"
            )
        lines.append("")
    if answer_rows:
        lines.append("## Answer metrics (per backend, averaged)")
        lines.append("")
        lines.append(
            "| Backend | Method | Avg Entity Hits | Avg Citation Count | "
            "Insufficient-Evidence Match Rate |"
        )
        lines.append("|---|---|---|---|---|")
        per_backend: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in answer_rows:
            per_backend.setdefault((row["backend"], row["method"]), []).append(row)
        for (backend, method), rows in sorted(per_backend.items()):
            entity_hits = sum(
                int(row.get("matched_entity_count", 0) or 0) for row in rows
            ) / len(rows)
            citations = sum(
                int(row.get("citation_count", 0) or 0) for row in rows
            ) / len(rows)
            match_rate = sum(
                1
                for row in rows
                if row.get("insufficient_evidence_behavior") == "matches_expectation"
            ) / len(rows)
            lines.append(
                f"| {backend} | {method} | {entity_hits:.2f} | {citations:.2f} | {match_rate:.2f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
