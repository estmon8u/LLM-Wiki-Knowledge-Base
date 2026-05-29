"""Core data types for the RAG evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RetrievedContext:
    """A single retrieved evidence unit, normalized across backends."""

    text: str
    source_id: str
    ref: str = ""
    score: float = 0.0


@dataclass
class RagSample:
    """The uniform output of running one question through one backend.

    Every backend produces this same shape so metrics are computed identically
    and fairly across backends.
    """

    question_id: str
    question: str
    backend: str
    method: str
    retrieved_contexts: list[RetrievedContext] = field(default_factory=list)
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    insufficient_evidence: bool = False
    latency_seconds: float = 0.0
    provider_mode: str = ""
    error: str | None = None

    @property
    def context_texts(self) -> list[str]:
        """The retrieved context texts (what RAGAS scores against)."""
        return [ctx.text for ctx in self.retrieved_contexts]

    @property
    def retrieved_source_ids(self) -> list[str]:
        """Ordered, de-duplicated retrieved source ids (for IR metrics)."""
        seen: set[str] = set()
        ordered: list[str] = []
        for ctx in self.retrieved_contexts:
            sid = ctx.source_id
            if sid and sid not in seen:
                seen.add(sid)
                ordered.append(sid)
        return ordered
