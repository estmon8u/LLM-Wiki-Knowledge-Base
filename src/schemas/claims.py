"""Claim and evidence schemas — the central primitives for deliberation."""

from __future__ import annotations

import hashlib
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class EvidenceItem(BaseModel):
    """One retrieved chunk from the compiled wiki."""

    model_config = ConfigDict(strict=True)

    page_path: str
    title: str
    snippet: str
    score: int = 0
    section: str = ""
    chunk_index: Optional[int] = None

    @property
    def citation_ref(self) -> str:
        if self.chunk_index is None or self.chunk_index < 0:
            return self.page_path
        return f"{self.page_path}#chunk-{self.chunk_index}"


class EvidenceBundle(BaseModel):
    """Frozen retrieval context shared by all candidates in a deliberation run."""

    model_config = ConfigDict(strict=True)

    question: str
    items: list[EvidenceItem] = Field(default_factory=list)

    @property
    def context_hash(self) -> str:
        """Deterministic hash over the question and retrieved items."""
        parts = [self.question] + [
            f"{item.citation_ref}:{item.section}:{item.snippet}" for item in self.items
        ]
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


class Claim(BaseModel):
    """An atomic factual claim extracted from a candidate answer or wiki page."""

    model_config = ConfigDict(strict=True)

    text: str
    source_page: str = ""
    section: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    grounded: bool = True


class CandidateAnswer(BaseModel):
    """One sampled answer from a provider during self-consistency query."""

    model_config = ConfigDict(strict=True)

    raw_text: str
    claims: list[Claim] = Field(default_factory=list)
    model_name: str = ""
    latency_ms: int = 0
    token_usage: Optional[int] = None
    error: Optional[str] = None


class MergedAnswer(BaseModel):
    """Final answer produced by merging multiple candidate answers."""

    model_config = ConfigDict(strict=True)

    text: str
    accepted_claims: list[Claim] = Field(default_factory=list)
    dropped_claims: list[Claim] = Field(default_factory=list)
    candidate_count: int = 0
