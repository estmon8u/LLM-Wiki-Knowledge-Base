"""Offline tests for the bias-mitigated judge and the RAGAS scorer wrapper."""

from __future__ import annotations

import json

from graphwiki_kb.providers.base import ProviderResponse, TextProvider
from scripts.rag_eval.judge import LLMJudge
from scripts.rag_eval.ragas_metrics import RagasItem, RagasScorer, select_metrics

# --------------------------------------------------------------------------- #
# Judge                                                                       #
# --------------------------------------------------------------------------- #


class _RubricProvider(TextProvider):
    name = "fake-judge"

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def generate(self, request) -> ProviderResponse:  # type: ignore[override]
        return ProviderResponse(text=json.dumps(self._payload), model_name="fake")


def test_judge_score_answer_normalizes() -> None:
    judge = LLMJudge(
        provider=_RubricProvider({"correctness": 5, "groundedness": 4, "relevance": 3})
    )
    scores = judge.score_answer(
        question="q", answer="a", reference="ref", contexts=["c"]
    )
    assert scores.correctness == 1.0
    assert scores.groundedness == 0.8
    assert scores.relevance == 0.6
    assert scores.overall == (1.0 + 0.8 + 0.6) / 3.0


def test_judge_neutral_on_provider_failure() -> None:
    class _Boom(TextProvider):
        name = "boom"

        def generate(self, request):  # type: ignore[override]
            raise RuntimeError("down")

    scores = LLMJudge(provider=_Boom()).score_answer(
        question="q", answer="a", reference=None, contexts=[]
    )
    assert (scores.correctness, scores.groundedness, scores.relevance) == (
        0.0,
        0.0,
        0.0,
    )


class _PositionBiasedProvider(TextProvider):
    """Always prefers whichever answer is presented FIRST (pure position bias)."""

    name = "biased"

    def generate(self, request) -> ProviderResponse:  # type: ignore[override]
        return ProviderResponse(text='{"winner": "first"}', model_name="fake")


def test_pairwise_order_swap_neutralizes_position_bias() -> None:
    judge = LLMJudge(provider=_PositionBiasedProvider())
    # Biased judge says "first" both orders -> disagreement -> tie.
    assert judge.pairwise(question="q", answer_a="A", answer_b="B") == "tie"


class _ConsistentProvider(TextProvider):
    """Consistently prefers the answer containing the word 'good'."""

    name = "consistent"

    def generate(self, request) -> ProviderResponse:  # type: ignore[override]
        prompt = request.prompt
        first = prompt.split("Answer FIRST:", 1)[1].split("Answer SECOND:", 1)[0]
        winner = "first" if "good" in first else "second"
        return ProviderResponse(text=json.dumps({"winner": winner}), model_name="f")


def test_pairwise_consistent_judge_picks_winner() -> None:
    judge = LLMJudge(provider=_ConsistentProvider())
    assert judge.pairwise(question="q", answer_a="good answer", answer_b="bad") == "A"
    assert judge.pairwise(question="q", answer_a="bad", answer_b="good answer") == "B"


# --------------------------------------------------------------------------- #
# RAGAS scorer (injected evaluate_fn; no network)                             #
# --------------------------------------------------------------------------- #


def test_select_metrics_full_when_contexts_and_reference() -> None:
    items = [RagasItem("q1", "Q", "A", ["ctx"], "ref")]
    selected = select_metrics(
        items,
        ("faithfulness", "answer_relevancy", "context_precision", "context_recall"),
    )
    assert selected == [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]


def test_select_metrics_direct_backend_only_answer_relevancy() -> None:
    # No contexts (direct backend) + reference present -> only answer_relevancy.
    items = [RagasItem("q1", "Q", "A", [], "ref")]
    selected = select_metrics(
        items,
        ("faithfulness", "answer_relevancy", "context_precision", "context_recall"),
    )
    assert selected == ["answer_relevancy"]


def test_select_metrics_no_reference_drops_context_metrics() -> None:
    items = [RagasItem("q1", "Q", "A", ["ctx"], None)]
    selected = select_metrics(
        items,
        ("faithfulness", "answer_relevancy", "context_precision", "context_recall"),
    )
    assert selected == ["faithfulness", "answer_relevancy"]


def test_ragas_scorer_uses_injected_evaluate_fn() -> None:
    captured = {}

    def fake_eval(items, metrics):
        captured["metrics"] = metrics
        return {it.question_id: dict.fromkeys(metrics, 0.9) for it in items}

    scorer = RagasScorer(evaluate_fn=fake_eval)
    items = [RagasItem("q1", "Q", "A", ["ctx"], "ref")]
    out = scorer.score(items)
    assert out["q1"]["faithfulness"] == 0.9
    assert set(captured["metrics"]) == {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    }


def test_ragas_scorer_empty_items() -> None:
    assert RagasScorer(evaluate_fn=lambda i, m: {}).score([]) == {}


def test_ragas_scorer_no_applicable_metrics() -> None:
    # No contexts and no reference -> only answer_relevancy applies, which DOES
    # apply; force the empty case by requesting only context metrics.
    scorer = RagasScorer(
        config=RagasScorer().config.__class__(metrics=("faithfulness",)),
        evaluate_fn=lambda i, m: {"x": {}},
    )
    items = [RagasItem("q1", "Q", "A", [], None)]
    out = scorer.score(items)
    assert out == {"q1": {}}
