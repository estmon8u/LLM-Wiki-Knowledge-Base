"""Tests for the deterministic RAG-eval layers (dataset, retrieval, generation)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.rag_eval.dataset import EvalQuestion, load_benchmark
from scripts.rag_eval.generation_metrics import score_generation
from scripts.rag_eval.retrieval_metrics import (
    context_relevance_flags,
    score_retrieval,
)
from scripts.rag_eval.types import RagSample, RetrievedContext


def _ctx(text: str, source_id: str = "", ref: str = "") -> RetrievedContext:
    return RetrievedContext(text=text, source_id=source_id, ref=ref)


# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #


def test_load_real_benchmark_v4_has_reference_answers() -> None:
    questions = load_benchmark(Path("eval") / "benchmark.yaml")
    assert len(questions) == 14
    assert all(q.reference_answer for q in questions)
    unsupported = next(q for q in questions if q.id == "unsupported_claim")
    assert unsupported.insufficient_expected is True
    realm = next(q for q in questions if q.id == "realm_vs_rag")
    assert "REALM" in realm.expected_source_ids


# --------------------------------------------------------------------------- #
# Retrieval metrics                                                           #
# --------------------------------------------------------------------------- #


def test_relevance_flags_word_boundary() -> None:
    contexts = [_ctx("REALM jointly trains a retriever"), _ctx("a fragment of text")]
    # "rag" must NOT match "fragment"; "REALM" matches.
    flags = context_relevance_flags(("REALM", "RAG"), contexts)
    assert flags == [True, False]


def test_score_retrieval_basic() -> None:
    contexts = [
        _ctx("REALM is a retrieval-augmented LM"),
        _ctx("unrelated content"),
        _ctx("RAG generates from retrieved docs"),
    ]
    scores = score_retrieval(("REALM", "RAG"), contexts, k=8)
    assert scores.has_ground_truth is True
    assert scores.recall_at_k == pytest.approx(1.0)
    assert scores.precision_at_k == pytest.approx(2 / 8)
    assert scores.hit_at_k == 1.0
    assert scores.mrr == pytest.approx(1.0)  # first context is relevant
    assert 0.9 < scores.ndcg_at_k <= 1.0


def test_score_retrieval_mrr_when_first_relevant_is_second() -> None:
    contexts = [_ctx("noise"), _ctx("DPR dense passage retrieval")]
    scores = score_retrieval(("DPR",), contexts, k=8)
    assert scores.mrr == pytest.approx(0.5)
    assert scores.hit_at_k == 1.0


def test_score_retrieval_no_ground_truth() -> None:
    scores = score_retrieval((), [_ctx("anything")], k=8)
    assert scores.has_ground_truth is False
    assert scores.recall_at_k is None
    assert scores.ndcg_at_k is None


def test_score_retrieval_miss() -> None:
    scores = score_retrieval(("REALM",), [_ctx("nothing relevant here")], k=8)
    assert scores.recall_at_k == 0.0
    assert scores.hit_at_k == 0.0
    assert scores.mrr == 0.0


# --------------------------------------------------------------------------- #
# Generation metrics                                                          #
# --------------------------------------------------------------------------- #


def _question(**kw) -> EvalQuestion:
    base: dict = {
        "id": "q",
        "question": "Q?",
        "expected_source_ids": (),
        "reference_answer": None,
    }
    base.update(kw)
    return EvalQuestion(**base)


def test_generation_citation_validity_and_grounded() -> None:
    sample = RagSample(
        question_id="q",
        question="Q?",
        backend="wikigraph",
        method="hybrid",
        retrieved_contexts=[_ctx("body", "s1", "wiki/sources/a.md#chunk-1")],
        answer="grounded answer [C1]",
        citations=["wiki/sources/a.md#chunk-1", "bogus.md#x"],
        insufficient_evidence=False,
    )
    scores = score_generation(_question(), sample)
    assert scores.citation_validity == pytest.approx(0.5)  # 1 of 2 valid
    assert scores.grounded == 1.0
    assert scores.citation_count == 2


def test_generation_path_only_citation_counts_valid() -> None:
    sample = RagSample(
        question_id="q",
        question="Q?",
        backend="x",
        method="m",
        retrieved_contexts=[_ctx("b", "s1", "wiki/sources/a.md#chunk-3")],
        answer="ans",
        citations=["wiki/sources/a.md#chunk-99"],  # same path, different anchor
        insufficient_evidence=False,
    )
    scores = score_generation(_question(), sample)
    assert scores.citation_validity == pytest.approx(1.0)


def test_generation_refusal_correct_on_unsupported() -> None:
    q = _question(insufficient_expected=True)
    refused = RagSample(
        question_id="q",
        question="Q?",
        backend="x",
        method="m",
        answer="I cannot answer; insufficient evidence.",
        insufficient_evidence=True,
    )
    answered = RagSample(
        question_id="q",
        question="Q?",
        backend="x",
        method="m",
        answer="fabricated",
        insufficient_evidence=False,
    )
    assert score_generation(q, refused).refusal_correct == 1.0
    assert score_generation(q, refused).grounded == 1.0  # correct refusal is grounded
    assert score_generation(q, answered).refusal_correct == 0.0


def test_generation_reference_overlap_and_entities() -> None:
    q = _question(
        reference_answer="DPR uses a dual encoder for dense passage retrieval",
        expected_entities=("DPR", "dual encoder"),
    )
    sample = RagSample(
        question_id="q",
        question="Q?",
        backend="x",
        method="m",
        retrieved_contexts=[_ctx("b", "s1", "r#chunk-0")],
        answer="DPR uses a dual encoder for dense retrieval",
        citations=["r#chunk-0"],
        insufficient_evidence=False,
    )
    scores = score_generation(q, sample)
    assert scores.token_f1 is not None and scores.token_f1 > 0.5
    assert scores.rouge_l is not None and scores.rouge_l > 0.4
    assert scores.entity_coverage == pytest.approx(1.0)
    assert scores.answer_token_length > 0
