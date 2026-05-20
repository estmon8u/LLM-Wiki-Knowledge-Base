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
    # Short body snippets from each retrieved context so the
    # source-coverage matcher can also see body content (not just the
    # heading/title). Critical for paper-body-only matches like ORQA.
    retrieved_text_snippets: list[str] = field(default_factory=list)
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
    # Per-kind context counts populated for the WikiGraphRAG backend so the
    # evaluator can distinguish source-derived TextUnits, curated wiki
    # chunks, claims, and community-summary contexts.
    text_unit_context_count: int = 0
    wiki_chunk_context_count: int = 0
    claim_context_count: int = 0
    community_context_count: int = 0
    unique_source_id_count: int = 0
    citation_ref_valid_rate: float = 0.0
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
                retrieved_text_snippets=[
                    (ctx.text or "")[:600] for ctx in result.contexts
                ],
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
            kind_counts = _count_context_kinds(ans.contexts)
            unique_sources = {sid for ctx in ans.contexts for sid in ctx.source_ids}
            cited_refs = {citation.get("ref") for citation in ans.citations}
            known_refs = {ctx.citation_ref for ctx in ans.contexts}
            valid_cited = sum(1 for ref in cited_refs if ref in known_refs)
            citation_ref_valid_rate = (
                valid_cited / len(cited_refs) if cited_refs else 0.0
            )
            return AnswerRun(
                backend=self.name,
                method=self.method,
                question_id=question.id,
                question=question.question,
                answer=ans.answer,
                insufficient_evidence=ans.insufficient_evidence,
                citation_count=len(ans.citations),
                latency_seconds=elapsed,
                text_unit_context_count=kind_counts.get("text_unit", 0),
                wiki_chunk_context_count=kind_counts.get("chunk", 0),
                claim_context_count=kind_counts.get("claim", 0),
                community_context_count=kind_counts.get("community", 0),
                unique_source_id_count=len(unique_sources),
                citation_ref_valid_rate=citation_ref_valid_rate,
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
                retrieved_text_snippets=[
                    (result.snippet or "")[:600] for result in results
                ],
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


def _count_context_kinds(contexts: list[Any]) -> dict[str, int]:
    """Return ``{node_kind: count}`` for a list of retrieved contexts.

    Works for both :class:`WikiGraphRetrievedContext` (has ``node_kind``)
    and dict-shaped payloads, so the helper can be reused from the
    evaluator JSON layer.
    """
    counts: dict[str, int] = {}
    for ctx in contexts:
        kind = getattr(ctx, "node_kind", None)
        if kind is None and isinstance(ctx, dict):
            kind = ctx.get("node_kind")
        if not isinstance(kind, str):
            continue
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def matched_source_ids(question: BenchmarkQuestion, run: RetrievalRun) -> list[str]:
    """Return the expected source ids that appear in ``run``.

    The haystack now includes the short body snippets of retrieved
    contexts so that backends which surface paper-body content (notably
    the TextUnit layer in WikiGraphRAG) get credit when the body
    mentions an expected source name even if the paper's *slug* does
    not (e.g. the ORQA paper whose slug is ``latent-retrieval-...``).
    """
    if not question.expected_sources:
        return []
    haystack = " ".join(
        [
            *run.retrieved_titles,
            *run.retrieved_paths,
            *run.retrieved_source_ids,
            *run.retrieved_text_snippets,
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
    """Compute retrieval metrics for one (question, run) pair.

    The previous formula returned ``0.0`` when ``expected_sources`` was
    empty (synthesis / out-of-scope questions), which dragged every
    backend's average toward zero. We now emit:

    * ``recall_at_5`` — empty string when ``expected_sources`` is empty
      so the column averages over only the questions that have ground
      truth.
    * ``effective_recall_at_5`` — same value, kept under a distinct
      name so summary tooling can compute the fair average without
      ambiguity.
    * ``has_ground_truth`` — 1 / 0 so downstream tools can weight per
      question.
    """
    matched = matched_source_ids(question, run)
    expected = len(question.expected_sources)
    if expected:
        recall = len(matched) / expected
        return {
            "matched_source_count": len(matched),
            "expected_source_count": expected,
            "recall_at_5": recall,
            "effective_recall_at_5": recall,
            "has_ground_truth": 1,
            "retrieved_count": len(run.retrieved_titles),
            "latency_seconds": run.latency_seconds,
            "error": run.error,
        }
    return {
        "matched_source_count": 0,
        "expected_source_count": 0,
        "recall_at_5": "",
        "effective_recall_at_5": "",
        "has_ground_truth": 0,
        "retrieved_count": len(run.retrieved_titles),
        "latency_seconds": run.latency_seconds,
        "error": run.error,
    }


def answer_metrics(question: BenchmarkQuestion, run: AnswerRun) -> dict[str, Any]:
    """Compute answer metrics for one (question, run) pair.

    We now distinguish:

    * ``matched_entity_count`` — raw count of expected entities that
      appear in the answer text (gameable by refusals that name-drop
      the entity, kept for backwards compatibility).
    * ``grounded_entity_hits`` — only counts entity matches when the
      backend produced a grounded answer (``insufficient_evidence ==
      False``). Refusals contribute 0. This is a much fairer score.
    * ``answer_quality_score`` — composite in ``[0, 1]`` averaging:
        - grounded_entity_rate (proportion of expected entities cited
          in a grounded answer; questions with no expected entities
          fall back to 1.0 when the run grounded, 0.0 when refused
          incorrectly),
        - normalized citation count (clipped to 5),
        - insufficient-evidence behavior (1 when matches_expectation,
          0 otherwise),
        - citation_ref_valid_rate (defaults to 1 when there are no
          citations, since vacuously valid).
    """
    entity_hits = matched_entities(question, run)
    expected_insufficient = question.insufficient_evidence_expected
    behavior_match = run.insufficient_evidence == expected_insufficient
    behavior = "matches_expectation" if behavior_match else "mismatch"

    grounded_entity_hits = 0 if run.insufficient_evidence else len(entity_hits)
    expected_entity_count = len(question.expected_entities)
    if expected_entity_count:
        grounded_entity_rate = grounded_entity_hits / expected_entity_count
    else:
        # Entity-free questions (corpus_themes, missing_topic) score on
        # whether the backend behaved as expected: grounded when the
        # question wanted an answer, refused when it didn't.
        grounded_entity_rate = 1.0 if behavior_match else 0.0

    normalized_citations = min(run.citation_count, 5) / 5.0
    insufficient_score = 1.0 if behavior_match else 0.0
    ref_valid = (
        run.citation_ref_valid_rate if run.citation_count else 1.0
    )
    answer_quality_score = round(
        (
            grounded_entity_rate
            + normalized_citations
            + insufficient_score
            + ref_valid
        )
        / 4.0,
        4,
    )

    return {
        "answer_length": len(run.answer or ""),
        "citation_count": run.citation_count,
        "matched_entity_count": len(entity_hits),
        "expected_entity_count": expected_entity_count,
        "grounded_entity_hits": grounded_entity_hits,
        "grounded_entity_rate": round(grounded_entity_rate, 4),
        "insufficient_evidence_expected": expected_insufficient,
        "insufficient_evidence_observed": run.insufficient_evidence,
        "insufficient_evidence_behavior": behavior,
        "answer_quality_score": answer_quality_score,
        "text_unit_context_count": run.text_unit_context_count,
        "wiki_chunk_context_count": run.wiki_chunk_context_count,
        "claim_context_count": run.claim_context_count,
        "community_context_count": run.community_context_count,
        "unique_source_id_count": run.unique_source_id_count,
        "citation_ref_valid_rate": run.citation_ref_valid_rate,
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
    "effective_recall_at_5",
    "has_ground_truth",
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
    "grounded_entity_hits",
    "grounded_entity_rate",
    "insufficient_evidence_expected",
    "insufficient_evidence_observed",
    "insufficient_evidence_behavior",
    "answer_quality_score",
    "text_unit_context_count",
    "wiki_chunk_context_count",
    "claim_context_count",
    "community_context_count",
    "unique_source_id_count",
    "citation_ref_valid_rate",
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
    """Write a human-readable backend comparison summary.

    Averages are computed fairly:

    * Retrieval ``Effective Recall@5`` only counts questions with
      ground truth (``has_ground_truth == 1``); synthesis and
      out-of-scope questions no longer drag the headline number to
      zero.
    * Answer ``Quality Score`` is the composite from
      :func:`answer_metrics` (grounded-entity rate + normalized
      citations + insufficient-evidence behavior + citation_ref_valid
      rate), averaged across all questions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# Backend evaluation summary", ""]
    if retrieval_rows:
        lines.append("## Retrieval metrics (per backend, averaged)")
        lines.append("")
        lines.append(
            "| Backend | Method | Effective Recall@5 | "
            "Questions w/ Ground Truth | Avg Latency (s) | Errors |"
        )
        lines.append("|---|---|---|---|---|---|")
        per_backend: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in retrieval_rows:
            per_backend.setdefault((row["backend"], row["method"]), []).append(row)
        for (backend, method), rows in sorted(per_backend.items()):
            grounded_rows = [
                row
                for row in rows
                if int(row.get("has_ground_truth", 0) or 0) == 1
            ]
            if grounded_rows:
                effective = sum(
                    float(row.get("effective_recall_at_5", 0) or 0)
                    for row in grounded_rows
                ) / len(grounded_rows)
                effective_str = f"{effective:.3f}"
            else:
                effective_str = "n/a"
            latency = sum(
                float(row.get("latency_seconds", 0) or 0) for row in rows
            ) / len(rows)
            errors = sum(1 for row in rows if row.get("error"))
            lines.append(
                f"| {backend} | {method} | {effective_str} | "
                f"{len(grounded_rows)}/{len(rows)} | "
                f"{latency:.3f} | {errors} |"
            )
        lines.append("")
    if answer_rows:
        lines.append("## Answer metrics (per backend, averaged)")
        lines.append("")
        lines.append(
            "| Backend | Method | Quality Score | Grounded Entity Rate | "
            "Avg Citations | Insufficient-Evidence Match | "
            "Citation Ref Valid Rate |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        per_backend: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in answer_rows:
            per_backend.setdefault((row["backend"], row["method"]), []).append(row)
        for (backend, method), rows in sorted(per_backend.items()):
            quality = sum(
                float(row.get("answer_quality_score", 0) or 0) for row in rows
            ) / len(rows)
            grounded_entity_rate = sum(
                float(row.get("grounded_entity_rate", 0) or 0) for row in rows
            ) / len(rows)
            citations = sum(
                int(row.get("citation_count", 0) or 0) for row in rows
            ) / len(rows)
            match_rate = sum(
                1
                for row in rows
                if row.get("insufficient_evidence_behavior") == "matches_expectation"
            ) / len(rows)
            ref_valid = sum(
                float(row.get("citation_ref_valid_rate", 0) or 0) for row in rows
            ) / len(rows)
            lines.append(
                f"| {backend} | {method} | **{quality:.3f}** | "
                f"{grounded_entity_rate:.3f} | "
                f"{citations:.2f} | {match_rate:.2f} | "
                f"{ref_valid:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
