"""Bias-mitigated LLM-as-judge for the RAG evaluation harness.

Uses the project's own ``TextProvider`` (no langchain) so it works with any
configured provider. Bias-mitigation techniques (from the LLM-as-judge
literature):

* **temperature 0** + strict JSON schema (score stability).
* **blinded system identity** — the judge never sees which backend produced an
  answer ("the system", not "wikigraph").
* **anchored rubric** — concrete 1-5 anchors for each axis, with explicit
  instructions to penalize verbosity and unsupported claims (anti-gaming).
* **order-swap** for pairwise comparisons — run both orders and only count a
  win on agreement, otherwise a tie (neutralizes position bias).
* **cross-family judge** — callers should pass a judge provider from a different
  model family than the generator when possible.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from graphwiki_kb.providers.base import ProviderRequest, TextProvider
from graphwiki_kb.providers.structured import (
    StructuredOutputError,
    parse_model_payload,
)

_RUBRIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "correctness": {"type": "integer", "minimum": 1, "maximum": 5},
        "groundedness": {"type": "integer", "minimum": 1, "maximum": 5},
        "relevance": {"type": "integer", "minimum": 1, "maximum": 5},
    },
    "required": ["correctness", "groundedness", "relevance"],
}

_SYSTEM = (
    "You are a strict, impartial evaluator of an answer produced by 'the "
    "system' (identity hidden). Score on three axes, each an integer 1-5:\n"
    "- correctness: factual agreement with the reference answer (5=fully "
    "correct, 1=wrong/contradicted). If no reference is given, judge against "
    "the source excerpts.\n"
    "- groundedness: is every claim supported by the source excerpts? "
    "(5=fully supported, 1=hallucinated). Unsupported claims must lower this.\n"
    "- relevance: does the answer address the question concisely? Penalize "
    "padding/verbosity that adds no information.\n"
    "Return ONLY JSON matching the schema. Do not reward length."
)


class _RubricResult(BaseModel):
    correctness: int = 3
    groundedness: int = 3
    relevance: int = 3


@dataclass(frozen=True)
class JudgeScores:
    """Normalized [0,1] rubric scores from the LLM judge."""

    correctness: float
    groundedness: float
    relevance: float

    @property
    def overall(self) -> float:
        """Mean of the three axes."""
        return (self.correctness + self.groundedness + self.relevance) / 3.0


@dataclass
class LLMJudge:
    """Pointwise + pairwise LLM judge over a project ``TextProvider``."""

    provider: TextProvider
    name: str = "judge"

    def score_answer(
        self,
        *,
        question: str,
        answer: str,
        reference: str | None,
        contexts: list[str],
    ) -> JudgeScores:
        """Return normalized rubric scores for one answer."""
        prompt = _build_rubric_prompt(question, answer, reference, contexts)
        try:
            response = self.provider.generate(
                ProviderRequest(
                    prompt=prompt,
                    system_prompt=_SYSTEM,
                    max_tokens=512,
                    response_schema=_RUBRIC_SCHEMA,
                    response_schema_name="rag_eval_rubric",
                    reasoning_effort="low",
                )
            )
            parsed = parse_model_payload(
                response.text, _RubricResult, label="judge rubric"
            )
        except (StructuredOutputError, Exception):
            return JudgeScores(0.0, 0.0, 0.0)
        return JudgeScores(
            correctness=_clamp(parsed.correctness),
            groundedness=_clamp(parsed.groundedness),
            relevance=_clamp(parsed.relevance),
        )

    def pairwise(
        self,
        *,
        question: str,
        answer_a: str,
        answer_b: str,
        reference: str | None = None,
    ) -> str:
        """Return 'A', 'B', or 'tie' using order-swap to neutralize position bias."""
        first = self._compare(question, answer_a, answer_b, reference)
        second = self._compare(question, answer_b, answer_a, reference)
        # ``first`` ruled on (A=first, B=second); ``second`` swapped them.
        if first == "first" and second == "second":
            return "A"
        if first == "second" and second == "first":
            return "B"
        return "tie"

    def _compare(
        self, question: str, first: str, second: str, reference: str | None
    ) -> str:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "winner": {"type": "string", "enum": ["first", "second", "tie"]}
            },
            "required": ["winner"],
        }
        ref_block = f"\nReference answer:\n{reference}\n" if reference else ""
        prompt = (
            f"Question:\n{question}\n{ref_block}\n"
            f"Answer FIRST:\n{first}\n\nAnswer SECOND:\n{second}\n\n"
            "Which answer is better (more correct, grounded, concise)? "
            'Return JSON {"winner": "first"|"second"|"tie"}.'
        )
        try:
            response = self.provider.generate(
                ProviderRequest(
                    prompt=prompt,
                    system_prompt="You are an impartial judge. Do not reward length.",
                    max_tokens=64,
                    response_schema=schema,
                    response_schema_name="rag_eval_pairwise",
                    reasoning_effort="low",
                )
            )
            parsed = parse_model_payload(response.text, _Pairwise, label="pairwise")
        except (StructuredOutputError, Exception):
            return "tie"
        return parsed.winner


class _Pairwise(BaseModel):
    winner: str = "tie"


def _clamp(value: int) -> float:
    return max(1, min(5, int(value))) / 5.0


def _build_rubric_prompt(
    question: str, answer: str, reference: str | None, contexts: list[str]
) -> str:
    excerpts = "\n\n".join(f"[C{i}] {c}" for i, c in enumerate(contexts[:8], start=1))
    ref_block = f"\n## Reference answer\n{reference}\n" if reference else ""
    return (
        f"## Question\n{question}\n{ref_block}\n"
        f"## Source excerpts\n{excerpts or '(none)'}\n\n"
        f"## System answer\n{answer}\n\n"
        "Score correctness, groundedness, relevance (1-5 each) as JSON."
    )
