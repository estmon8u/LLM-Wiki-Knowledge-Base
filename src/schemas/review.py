"""Review finding and verdict schemas for adversarial review."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Verdict(str, Enum):
    """Typed outcome of an adversarial review evaluation."""

    CONSISTENT = "consistent"
    CONTRADICTORY = "contradictory"
    TERM_DRIFT = "term_drift"
    NEEDS_REVIEW = "needs_review"


class ReviewFinding(BaseModel):
    """One finding from an adversarial review pass."""

    model_config = ConfigDict(strict=True)

    issue_type: str
    affected_pages: list[str] = Field(default_factory=list)
    claim: str = ""
    evidence_for: str = ""
    evidence_against: str = ""
    verdict: Verdict = Verdict.NEEDS_REVIEW
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    citations: list[str] = Field(default_factory=list)
