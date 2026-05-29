"""Deterministic, anti-gaming generation metrics (no LLM).

These complement the RAGAS LLM-judge metrics with cheap, reproducible signals
that are hard to game: citation validity against *returned* contexts, grounded
behavior, refusal correctness on insufficient-evidence questions, lexical
overlap (token-F1 / ROUGE-L) vs the reference answer, and answer length (a
verbosity signal, never rewarded).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scripts.rag_eval.dataset import EvalQuestion
from scripts.rag_eval.types import RagSample

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _token_f1(prediction: str, reference: str) -> float:
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred or not ref:
        return 0.0
    from collections import Counter

    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _rouge_l(prediction: str, reference: str) -> float:
    pred = _tokens(prediction)
    ref = _tokens(reference)
    if not pred or not ref:
        return 0.0
    # LCS length via DP.
    n, m = len(pred), len(ref)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[n][m]
    if lcs == 0:
        return 0.0
    precision = lcs / n
    recall = lcs / m
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class GenerationScores:
    """Deterministic generation/answer scores for one sample."""

    citation_validity: float
    citation_count: int
    grounded: float
    refusal_correct: float
    entity_coverage: float | None
    token_f1: float | None
    rouge_l: float | None
    answer_token_length: int
    insufficient_evidence: bool


def _valid_citation_count(sample: RagSample) -> int:
    known_refs = {ctx.ref for ctx in sample.retrieved_contexts if ctx.ref}
    known_paths = {ref.split("#", 1)[0] for ref in known_refs}
    valid = 0
    for ref in sample.citations:
        if ref in known_refs or ref.split("#", 1)[0] in known_paths:
            valid += 1
    return valid


def score_generation(question: EvalQuestion, sample: RagSample) -> GenerationScores:
    """Compute deterministic generation metrics for ``sample``."""
    citation_count = len(sample.citations)
    valid = _valid_citation_count(sample)
    citation_validity = (
        valid / citation_count
        if citation_count
        else (
            1.0
            if question.insufficient_expected and sample.insufficient_evidence
            else 0.0
        )
    )
    grounded = (
        1.0
        if (not sample.insufficient_evidence and valid > 0)
        else (
            1.0
            if question.insufficient_expected and sample.insufficient_evidence
            else 0.0
        )
    )
    refusal_correct = (
        1.0 if sample.insufficient_evidence == question.insufficient_expected else 0.0
    )
    entity_coverage: float | None = None
    if question.expected_entities and not sample.insufficient_evidence:
        text = sample.answer.lower()
        hits = sum(1 for e in question.expected_entities if e.lower() in text)
        entity_coverage = hits / len(question.expected_entities)
    token_f1: float | None = None
    rouge_l: float | None = None
    if question.reference_answer and not sample.insufficient_evidence:
        token_f1 = _token_f1(sample.answer, question.reference_answer)
        rouge_l = _rouge_l(sample.answer, question.reference_answer)
    return GenerationScores(
        citation_validity=citation_validity,
        citation_count=citation_count,
        grounded=grounded,
        refusal_correct=refusal_correct,
        entity_coverage=entity_coverage,
        token_f1=token_f1,
        rouge_l=rouge_l,
        answer_token_length=len(_tokens(sample.answer)),
        insufficient_evidence=sample.insufficient_evidence,
    )
