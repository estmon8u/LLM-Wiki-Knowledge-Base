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
    expected_answer_terms: tuple[str, ...] = ()
    forbidden_answer_terms: tuple[str, ...] = ()
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
            expected_answer_terms=tuple(
                str(s) for s in payload.get("expected_answer_terms", []) or []
            ),
            forbidden_answer_terms=tuple(
                str(s) for s in payload.get("forbidden_answer_terms", []) or []
            ),
            expected_methods={str(k): str(v) for k, v in expected_methods.items()},
            insufficient_evidence_expected=bool(
                payload.get("insufficient_evidence_expected", False)
                or "insufficient_evidence" in (payload.get("expected_behaviors") or [])
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
    # The actual retrieval *method* chosen by the backend (e.g.
    # ``local`` / ``global`` / ``drift-lite``). For WikiGraphRAG we
    # pass the WikiGraphFindResult.method here; for GraphRAG we record
    # the artifact-search or text-unit retrieval mode label. Used to
    # score ``method_fit`` against the benchmark ``expected_methods``
    # (G9 fix).
    chosen_method: str = ""
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
    # Loose validity: an LLM citation ref counts when it shares the
    # *path* of any retrieved context (WikiGraphRAG normalizes neighbor
    # anchors). Defined symmetrically for GraphRAG too (G5 fix).
    citation_ref_valid_rate: float = 0.0
    # Strict validity (G5): only counts when the citation ref equals a
    # retrieved-context citation_ref byte-for-byte. Reported alongside
    # the loose rate for transparency without changing the composite.
    citation_ref_strict_rate: float = 0.0
    provider_mode: str = ""
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
                # Record what the auto router actually chose so the
                # evaluator can compute method_fit (G9 fix).
                chosen_method=result.method or self.method,
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
            # Strict validity: byte-for-byte match against a retrieved
            # context's citation_ref. Loose validity (G6): any ref
            # sharing the path of a retrieved context counts because
            # the LLM was reasoning over the same source document.
            known_paths = {ctx.citation_ref.split("#", 1)[0] for ctx in ans.contexts}
            valid_strict = sum(1 for ref in cited_refs if ref in known_refs)
            valid_loose = sum(
                1
                for ref in cited_refs
                if ref in known_refs or str(ref or "").split("#", 1)[0] in known_paths
            )
            citation_ref_strict_rate = (
                valid_strict / len(cited_refs) if cited_refs else 0.0
            )
            citation_ref_valid_rate = (
                valid_loose / len(cited_refs) if cited_refs else 0.0
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
                citation_ref_strict_rate=citation_ref_strict_rate,
                provider_mode=str(ans.provider_status.get("mode", "")),
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

    def __init__(
        self,
        context: CommandContext,
        *,
        method: str = "auto",
        retrieve_mode: str = "text_units",
    ) -> None:
        self.context = context
        self.method = method
        self.retrieve_mode = retrieve_mode
        self.find_service = context.services.graphrag_find
        self.ask_controller = context.services.graph_ask_controller
        # Lazy import to keep WikiGraphRunner-only callers fast.
        from scripts.graphrag_artifact_retriever import GraphRAGArtifactRetriever

        self.artifact_retriever = GraphRAGArtifactRetriever(
            context.services.graphrag_status,
            mode=retrieve_mode,
        )

    def retrieve(self, question: BenchmarkQuestion) -> RetrievalRun:
        """Execute provider-free retrieval over GraphRAG output parquets.

        Phase 3 (apples-to-apples): switches from the entity/relationship
        artifact-directory search to the full GraphRAG retrieval surface
        (text_units + community_reports + entities + relationships).
        """
        start = time.perf_counter()
        try:
            results = self.artifact_retriever.search(question.question, limit=8)
            elapsed = time.perf_counter() - start
            return RetrievalRun(
                backend=self.name,
                method=self.retrieve_mode,
                question_id=question.id,
                question=question.question,
                retrieved_titles=[r.title for r in results],
                retrieved_paths=[r.path for r in results],
                retrieved_source_ids=[sid for r in results for sid in r.source_ids],
                # G1 fix: include the body text of each retrieved
                # artifact so source-name matching is symmetric with
                # WikiGraphRAG.
                retrieved_text_snippets=[r.snippet[:600] for r in results],
                chosen_method=self.retrieve_mode,
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
            refs = list(answer.graph_data_references or [])
            valid_rate = _graphrag_ref_valid_rate(refs)
            return AnswerRun(
                backend=self.name,
                method=answer.method or self.method,
                question_id=question.id,
                question=question.question,
                answer=answer.answer or "",
                insufficient_evidence=insufficient,
                citation_count=len(refs),
                latency_seconds=elapsed,
                # G5 fix: compute a real fraction of inline [Data: ...]
                # parts whose `kind` is a known GraphRAG reference kind
                # and whose `ids` parsed cleanly. Both rates equal here
                # because GraphRAG has no "loose" anchor concept; the
                # strict rate is the same as the loose rate.
                citation_ref_valid_rate=valid_rate,
                citation_ref_strict_rate=valid_rate,
                provider_mode="provider-backed",
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


_KNOWN_GRAPHRAG_REF_KINDS = frozenset(
    {
        "source",
        "document",
        "text_unit",
        "entity",
        "relationship",
        "community",
        "community_report",
    }
)


def _graphrag_ref_valid_rate(refs: list[dict]) -> float:
    """Return the fraction of GraphRAG inline ``[Data: ...]`` refs that resolve.

    A reference is considered *valid* when it has a known kind (one of
    the labels GraphRAG emits, e.g. ``Entities``, ``Reports``, ``Text
    Units``) AND a non-empty parsed id list. This is the symmetric
    counterpart to WikiGraphRAG's citation_ref validity calculation
    (G5 fix).
    """
    if not refs:
        return 0.0
    total = len(refs)
    valid = 0
    for ref in refs:
        kind = str(ref.get("kind", "")).strip().lower()
        ids = ref.get("ids", []) or []
        if kind in _KNOWN_GRAPHRAG_REF_KINDS and ids:
            valid += 1
    return valid / total


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

    Matching is now case-insensitive *word-boundary* for single-token
    expected sources (G3 fix). Substring matching is retained for
    multi-token expected sources like ``"Dense Passage Retrieval"`` or
    ``"Fusion-in-Decoder"`` because tokenising those would split the
    surface form. This prevents bug-tier false positives like ``"fid"``
    matching ``"modified"`` or ``"rag"`` matching ``"fragment"``, which
    asymmetrically inflated long-body backends (WikiGraphRAG).
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
    )
    haystack_lower = haystack.lower()
    matched: list[str] = []
    for expected in question.expected_sources:
        if _expected_source_in_haystack(expected, haystack, haystack_lower):
            matched.append(expected)
    return matched


def _expected_source_in_haystack(
    needle: str, haystack: str, haystack_lower: str
) -> bool:
    """Return True when ``needle`` appears in ``haystack``.

    Single-token needles use a word-boundary regex (case-insensitive);
    multi-token / hyphenated / whitespace-containing needles fall back
    to a case-insensitive substring check.
    """
    import re as _re

    lowered = needle.lower().strip()
    if not lowered:
        return False
    # Multi-token / hyphenated -> substring is the only reliable form
    # because a word-boundary regex would punish curly punctuation or
    # missing connectives. We accept the small false-positive risk; it
    # affects both backends equally.
    if any(ch in lowered for ch in (" ", "\t", "-", "/", ".")):
        return lowered in haystack_lower
    pattern = r"(?<![A-Za-z0-9_])" + _re.escape(lowered) + r"(?![A-Za-z0-9_])"
    return bool(_re.search(pattern, haystack, _re.IGNORECASE))


# Small alias table so an expected entity like ``FiD`` is credited
# when the answer uses the spelled-out form ``Fusion-in-Decoder``, and
# vice versa. Keep this conservative: only well-known one-to-one
# expansions, and only over the entities that appear in our benchmarks.
_ENTITY_ALIAS_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "fid": ("fusion-in-decoder", "fusion in decoder"),
    "dpr": ("dense passage retrieval", "dense-passage-retrieval"),
    "rag": ("retrieval-augmented generation", "retrieval augmented generation"),
    "realm": ("retrieval-augmented language model",),
    "ralm": ("retrieval augmented language model", "in-context retrieval-augmented"),
    "atlas": ("few-shot learning with retrieval-augmented",),
    "self-rag": ("self rag", "self-reflective retrieval-augmented"),
    "replug": ("retrieval-augmented black-box",),
    "orqa": ("open retrieval question answering",),
    "mips": ("maximum inner product search",),
    "bert": ("bidirectional encoder representations from transformers",),
}


def matched_entities(question: BenchmarkQuestion, run: AnswerRun) -> list[str]:
    """Return the expected entities that appear in ``run.answer``.

    Also accepts well-known spelled-out forms (e.g. an expected entity
    ``FiD`` is credited when the answer uses ``Fusion-in-Decoder``).
    """
    if not question.expected_entities:
        return []
    text = run.answer.lower()
    hits: list[str] = []
    for entity in question.expected_entities:
        lowered = entity.lower()
        if lowered in text:
            hits.append(entity)
            continue
        for alias in _ENTITY_ALIAS_EXPANSIONS.get(lowered, ()):
            if alias in text:
                hits.append(entity)
                break
    return hits


def matched_answer_terms(question: BenchmarkQuestion, run: AnswerRun) -> list[str]:
    """Return expected answer terms that appear in a grounded answer."""
    if not question.expected_answer_terms or run.insufficient_evidence:
        return []
    text = run.answer.lower()
    return [term for term in question.expected_answer_terms if term.lower() in text]


def forbidden_answer_hits(question: BenchmarkQuestion, run: AnswerRun) -> list[str]:
    """Return forbidden answer terms that appear in the answer text."""
    if not question.forbidden_answer_terms:
        return []
    text = run.answer.lower()
    return [term for term in question.forbidden_answer_terms if term.lower() in text]


def retrieval_metrics(question: BenchmarkQuestion, run: RetrievalRun) -> dict[str, Any]:
    """Compute retrieval metrics for one (question, run) pair.

    The previous formula returned ``0.0`` when ``expected_sources`` was
    empty (synthesis / out-of-scope questions), which dragged every
    backend's average toward zero. We now emit:

    * ``recall_at_8`` — empty string when ``expected_sources`` is empty
      so the column averages over only the questions that have ground
      truth. Renamed from ``recall_at_5`` because both backends
      actually retrieve up to 8 contexts (limit=8 / max_context_chunks=8).
    * ``effective_recall_at_8`` — same value, kept under a distinct
      name so summary tooling can compute the fair average without
      ambiguity.
    * ``has_ground_truth`` — 1 / 0 so downstream tools can weight per
      question.
    * ``expected_method`` / ``chosen_method`` / ``method_fit`` — G9
      fix: surface whether the backend's auto router picked the
      expected query method.
    """
    matched = matched_source_ids(question, run)
    expected = len(question.expected_sources)
    expected_method = (
        question.expected_methods.get(run.backend, "")
        if question.expected_methods
        else ""
    )
    chosen_method = run.chosen_method or run.method or ""
    method_fit = _method_fit(expected_method, chosen_method)
    common = {
        "expected_method": expected_method,
        "chosen_method": chosen_method,
        "method_fit": method_fit,
    }
    if expected:
        recall = len(matched) / expected
        return {
            "matched_source_count": len(matched),
            "expected_source_count": expected,
            "recall_at_8": recall,
            "effective_recall_at_8": recall,
            "has_ground_truth": 1,
            "retrieved_count": len(run.retrieved_titles),
            "latency_seconds": run.latency_seconds,
            "error": run.error,
            **common,
        }
    return {
        "matched_source_count": 0,
        "expected_source_count": 0,
        "recall_at_8": "",
        "effective_recall_at_8": "",
        "has_ground_truth": 0,
        "retrieved_count": len(run.retrieved_titles),
        "latency_seconds": run.latency_seconds,
        "error": run.error,
        **common,
    }


def _method_fit(expected_method: str, chosen_method: str) -> str:
    """Return 1/0/'' for whether the chosen method matches the expectation.

    Empty string when no expectation was declared so averages skip
    those rows. Comparison folds case and ``-``/``_`` separators so
    ``drift`` matches ``drift-lite`` is *not* a match (separate
    families), but ``community_report`` matches ``community-report``
    is.
    """
    if not expected_method:
        return ""
    if not chosen_method:
        return "0"
    norm_expected = expected_method.strip().lower().replace("_", "-")
    norm_chosen = chosen_method.strip().lower().replace("_", "-")
    return "1" if norm_expected == norm_chosen else "0"


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
    answer_term_hits = matched_answer_terms(question, run)
    forbidden_hits = forbidden_answer_hits(question, run)
    expected_insufficient = question.insufficient_evidence_expected
    behavior_match = run.insufficient_evidence == expected_insufficient
    behavior = "matches_expectation" if behavior_match else "mismatch"

    grounded_entity_hits = 0 if run.insufficient_evidence else len(entity_hits)
    expected_entity_count = len(question.expected_entities)
    if expected_entity_count:
        grounded_entity_rate = grounded_entity_hits / expected_entity_count
    else:
        grounded_entity_rate = 1.0 if behavior_match else 0.0

    grounded_answer_term_hits = len(answer_term_hits)
    expected_answer_term_count = len(question.expected_answer_terms)
    if expected_answer_term_count:
        grounded_answer_term_rate = (
            grounded_answer_term_hits / expected_answer_term_count
        )
    else:
        grounded_answer_term_rate = 1.0 if behavior_match else 0.0

    # G4 fix: replace the gameable ``min(citations, 5)/5`` term with a
    # binary ``has_supported_citations`` signal so the composite no
    # longer rewards verbose citation lists. Refusals on intentionally
    # unsupported questions still score full marks.
    if expected_insufficient and run.insufficient_evidence:
        has_supported_citations = 1.0
        ref_valid = 1.0
    else:
        ref_valid = run.citation_ref_valid_rate if run.citation_count else 0.0
        has_supported_citations = (
            1.0 if (run.citation_count > 0 and ref_valid >= 0.5) else 0.0
        )
    insufficient_score = 1.0 if behavior_match else 0.0
    forbidden_score = 1.0 if not forbidden_hits else 0.0
    content_rate = (grounded_entity_rate + grounded_answer_term_rate) / 2.0
    answer_quality_score = round(
        (
            content_rate
            + has_supported_citations
            + insufficient_score
            + ref_valid
            + forbidden_score
        )
        / 5.0,
        4,
    )

    return {
        "answer_length": len(run.answer or ""),
        "citation_count": run.citation_count,
        "has_supported_citations": has_supported_citations,
        "matched_entity_count": len(entity_hits),
        "expected_entity_count": expected_entity_count,
        "grounded_entity_hits": grounded_entity_hits,
        "grounded_entity_rate": round(grounded_entity_rate, 4),
        "matched_answer_term_count": grounded_answer_term_hits,
        "expected_answer_term_count": expected_answer_term_count,
        "grounded_answer_term_rate": round(grounded_answer_term_rate, 4),
        "forbidden_answer_hit_count": len(forbidden_hits),
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
        "citation_ref_strict_rate": run.citation_ref_strict_rate,
        "provider_mode": run.provider_mode,
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
    "recall_at_8",
    "effective_recall_at_8",
    "has_ground_truth",
    "retrieved_count",
    "expected_method",
    "chosen_method",
    "method_fit",
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
    "has_supported_citations",
    "matched_entity_count",
    "expected_entity_count",
    "grounded_entity_hits",
    "grounded_entity_rate",
    "matched_answer_term_count",
    "expected_answer_term_count",
    "grounded_answer_term_rate",
    "forbidden_answer_hit_count",
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
    "citation_ref_strict_rate",
    "provider_mode",
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
            "| Backend | Method | Effective Recall@8 | "
            "Questions w/ Ground Truth | Method Fit | "
            "Avg Latency (s) | Errors |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        per_backend: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in retrieval_rows:
            per_backend.setdefault((row["backend"], row["method"]), []).append(row)
        for (backend, method), rows in sorted(per_backend.items()):
            grounded_rows = [
                row for row in rows if int(row.get("has_ground_truth", 0) or 0) == 1
            ]
            if grounded_rows:
                effective = sum(
                    float(row.get("effective_recall_at_8", 0) or 0)
                    for row in grounded_rows
                ) / len(grounded_rows)
                effective_str = f"{effective:.3f}"
            else:
                effective_str = "n/a"
            latency = sum(
                float(row.get("latency_seconds", 0) or 0) for row in rows
            ) / len(rows)
            errors = sum(1 for row in rows if row.get("error"))
            method_rows = [row for row in rows if str(row.get("method_fit", ""))]
            if method_rows:
                fit_rate = sum(
                    1 for row in method_rows if str(row["method_fit"]) == "1"
                ) / len(method_rows)
                fit_str = f"{fit_rate:.2f} ({len(method_rows)})"
            else:
                fit_str = "n/a"
            lines.append(
                f"| {backend} | {method} | {effective_str} | "
                f"{len(grounded_rows)}/{len(rows)} | {fit_str} | "
                f"{latency:.3f} | {errors} |"
            )
        lines.append("")
    if answer_rows:
        lines.append("## Answer metrics (per backend, averaged)")
        lines.append("")
        lines.append(
            "| Backend | Method | Provider Modes | Quality Score | "
            "Grounded Entity Rate | Grounded Term Rate | "
            "Has Supported Citations | Avg Citations | "
            "Insufficient-Evidence Match | Citation Ref Valid (loose) | "
            "Citation Ref Valid (strict) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
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
            grounded_term_rate = sum(
                float(row.get("grounded_answer_term_rate", 0) or 0) for row in rows
            ) / len(rows)
            citations = sum(
                int(row.get("citation_count", 0) or 0) for row in rows
            ) / len(rows)
            has_supported = sum(
                float(row.get("has_supported_citations", 0) or 0) for row in rows
            ) / len(rows)
            match_rate = sum(
                1
                for row in rows
                if row.get("insufficient_evidence_behavior") == "matches_expectation"
            ) / len(rows)
            ref_valid = sum(
                float(row.get("citation_ref_valid_rate", 0) or 0) for row in rows
            ) / len(rows)
            ref_strict = sum(
                float(row.get("citation_ref_strict_rate", 0) or 0) for row in rows
            ) / len(rows)
            provider_modes = ", ".join(
                sorted({str(row.get("provider_mode") or "unknown") for row in rows})
            )
            lines.append(
                f"| {backend} | {method} | {provider_modes} | "
                f"**{quality:.3f}** | "
                f"{grounded_entity_rate:.3f} | {grounded_term_rate:.3f} | "
                f"{has_supported:.2f} | "
                f"{citations:.2f} | {match_rate:.2f} | "
                f"{ref_valid:.3f} | {ref_strict:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
