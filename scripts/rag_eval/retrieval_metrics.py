"""Deterministic, rank-aware retrieval metrics (no LLM).

Scored against ground-truth ``expected_source_ids`` using binary relevance.
Relevance of a retrieved context is decided by matching an expected source
fragment (e.g. ``"REALM"``, ``"Dense Passage Retrieval"``) against the
context's source id / citation ref / body text -- single-token fragments use a
case-insensitive word-boundary match (so ``"rag"`` does not match
``"fragment"``), multi-token fragments use substring matching.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from scripts.rag_eval.types import RetrievedContext

_TEXT_CAP = 4000


def _fragment_matches(fragment: str, haystack: str, haystack_lower: str) -> bool:
    lowered = fragment.lower().strip()
    if not lowered:
        return False
    if any(ch in lowered for ch in (" ", "\t", "-", "/", ".")):
        return lowered in haystack_lower
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(lowered) + r"(?![A-Za-z0-9_])"
    return bool(re.search(pattern, haystack, re.IGNORECASE))


def context_relevance_flags(
    expected_sources: tuple[str, ...],
    contexts: list[RetrievedContext],
) -> list[bool]:
    """Return a per-context boolean relevance flag in rank order."""
    if not expected_sources:
        return [False] * len(contexts)
    flags: list[bool] = []
    for ctx in contexts:
        haystack = " ".join([ctx.source_id, ctx.ref, ctx.text[:_TEXT_CAP]])
        haystack_lower = haystack.lower()
        flags.append(
            any(
                _fragment_matches(src, haystack, haystack_lower)
                for src in expected_sources
            )
        )
    return flags


def _matched_expected(
    expected_sources: tuple[str, ...],
    contexts: list[RetrievedContext],
    k: int,
) -> set[str]:
    matched: set[str] = set()
    for ctx in contexts[:k]:
        haystack = " ".join([ctx.source_id, ctx.ref, ctx.text[:_TEXT_CAP]])
        haystack_lower = haystack.lower()
        for src in expected_sources:
            if _fragment_matches(src, haystack, haystack_lower):
                matched.add(src)
    return matched


@dataclass(frozen=True)
class RetrievalScores:
    """Rank-aware retrieval scores for one (question, backend) pair."""

    has_ground_truth: bool
    k: int
    recall_at_k: float | None
    precision_at_k: float | None
    hit_at_k: float | None
    mrr: float | None
    ndcg_at_k: float | None
    retrieved_count: int


def _ndcg_at_k(flags: list[bool], k: int) -> float:
    gains = [1.0 if rel else 0.0 for rel in flags[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def score_retrieval(
    expected_sources: tuple[str, ...],
    contexts: list[RetrievedContext],
    *,
    k: int = 8,
) -> RetrievalScores:
    """Compute recall@k / precision@k / hit@k / MRR / nDCG@k.

    Returns ``None`` metric values (and ``has_ground_truth=False``) when the
    question declares no expected sources, so aggregation can fairly average
    over only the ground-truth questions.
    """
    if not expected_sources:
        return RetrievalScores(
            has_ground_truth=False,
            k=k,
            recall_at_k=None,
            precision_at_k=None,
            hit_at_k=None,
            mrr=None,
            ndcg_at_k=None,
            retrieved_count=len(contexts),
        )
    flags = context_relevance_flags(expected_sources, contexts)
    top = flags[:k]
    relevant_in_top = sum(1 for rel in top if rel)
    matched_sources = _matched_expected(expected_sources, contexts, k)
    recall = len(matched_sources) / len(expected_sources)
    precision = relevant_in_top / k if k else 0.0
    hit = 1.0 if relevant_in_top > 0 else 0.0
    first_rel = next((i for i, rel in enumerate(flags) if rel), None)
    mrr = 1.0 / (first_rel + 1) if first_rel is not None else 0.0
    return RetrievalScores(
        has_ground_truth=True,
        k=k,
        recall_at_k=recall,
        precision_at_k=precision,
        hit_at_k=hit,
        mrr=mrr,
        ndcg_at_k=_ndcg_at_k(flags, k),
        retrieved_count=len(contexts),
    )
