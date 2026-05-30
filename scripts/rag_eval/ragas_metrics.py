"""RAGAS metric scoring via the real ``ragas`` library (provider-backed).

Computes faithfulness, answer relevancy, context precision, and context recall
using ``ragas`` (0.2.x ``evaluate`` API) with a LangChain provider. Metric
selection is automatic:

* ``answer_relevancy`` — always (needs question + answer + embeddings).
* ``faithfulness`` / ``context_precision`` — only when every item has retrieved
  contexts (so the no-retrieval ``direct`` backend isn't unfairly scored 0).
* ``context_recall`` / ``context_precision`` — only when references are present.

The actual ``ragas.evaluate`` call is injectable (``evaluate_fn``) so the
orchestration is unit-testable offline without any network/LLM calls.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import scripts.rag_eval._compat  # noqa: F401 - install ragas import shims first

EvaluateFn = Callable[[list["RagasItem"], tuple[str, ...]], dict[str, dict[str, float]]]


@dataclass(frozen=True)
class RagasItem:
    """One row scored by RAGAS."""

    question_id: str
    question: str
    answer: str
    contexts: list[str] = field(default_factory=list)
    reference: str | None = None


@dataclass(frozen=True)
class RagasConfig:
    """RAGAS LLM/embedding configuration."""

    provider: str = "openai"
    model: str = "gpt-5.4-nano"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimension: int = 0
    api_key_env: str = "OPENAI_API_KEY"
    metrics: tuple[str, ...] = (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    )


def select_metrics(items: list[RagasItem], requested: tuple[str, ...]) -> list[str]:
    """Choose which RAGAS metrics are applicable to ``items`` (fair selection)."""
    all_have_contexts = bool(items) and all(it.contexts for it in items)
    all_have_reference = bool(items) and all(it.reference for it in items)
    selected: list[str] = []
    context_metrics = {"faithfulness", "context_precision", "context_recall"}
    reference_metrics = {"context_precision", "context_recall"}
    for metric in requested:
        if metric in context_metrics and not all_have_contexts:
            continue
        if metric in reference_metrics and not all_have_reference:
            continue
        selected.append(metric)
    return selected


@dataclass
class RagasScorer:
    """Scores :class:`RagasItem`s with RAGAS (or an injected ``evaluate_fn``)."""

    config: RagasConfig = field(default_factory=RagasConfig)
    evaluate_fn: EvaluateFn | None = None

    def score(self, items: list[RagasItem]) -> dict[str, dict[str, float]]:
        """Return ``{question_id: {metric: score}}`` for ``items``."""
        if not items:
            return {}
        metrics = select_metrics(items, self.config.metrics)
        if not metrics:
            return {item.question_id: {} for item in items}
        if self.evaluate_fn is not None:
            return self.evaluate_fn(items, tuple(metrics))
        return self._score_with_ragas(items, tuple(metrics))

    def _score_with_ragas(
        self, items: list[RagasItem], metrics: tuple[str, ...]
    ) -> dict[str, dict[str, float]]:  # pragma: no cover - exercised in the real run
        import math

        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        metric_objects = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        api_key = os.environ.get(self.config.api_key_env, "")
        llm, embeddings = self._build_langchain_models(api_key)
        selected = [metric_objects[name] for name in metrics]
        samples = [
            SingleTurnSample(
                user_input=item.question,
                response=item.answer,
                retrieved_contexts=item.contexts or [""],
                reference=item.reference or "",
            )
            for item in items
        ]
        dataset = EvaluationDataset(samples=samples)
        result = evaluate(dataset, metrics=selected, llm=llm, embeddings=embeddings)
        frame = result.to_pandas()
        out: dict[str, dict[str, float]] = {}
        for row_index, item in enumerate(items):
            row = frame.iloc[row_index]
            scores: dict[str, float] = {}
            for name in metrics:
                if name in frame.columns:
                    value = row[name]
                    if value is not None and not (
                        isinstance(value, float) and math.isnan(value)
                    ):
                        scores[name] = float(value)
            out[item.question_id] = scores
        return out

    def _build_langchain_models(self, api_key: str):
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        provider = self.config.provider.strip().lower()
        if provider == "openai":
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings

            llm = ChatOpenAI(model=self.config.model, api_key=api_key, temperature=0)
            embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                api_key=api_key,
            )
            return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)

        if provider == "gemini":
            from langchain_google_genai import (
                ChatGoogleGenerativeAI,
                GoogleGenerativeAIEmbeddings,
            )

            llm = ChatGoogleGenerativeAI(
                model=self.config.model,
                api_key=api_key,
                temperature=0,
            )
            embedding_kwargs: dict[str, Any] = {
                "model": self.config.embedding_model,
                "api_key": api_key,
            }
            if self.config.embedding_dimension > 0:
                embedding_kwargs["output_dimensionality"] = (
                    self.config.embedding_dimension
                )
            embeddings = GoogleGenerativeAIEmbeddings(**embedding_kwargs)
            return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)

        raise ValueError(f"Unsupported RAGAS provider: {self.config.provider}")
