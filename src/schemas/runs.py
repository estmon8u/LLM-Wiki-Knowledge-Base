"""Run record schema for deliberation artifact persistence."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.claims import CandidateAnswer, Claim, EvidenceBundle, MergedAnswer
from src.schemas.review import ReviewFinding


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]


class RunRecord(BaseModel):
    """Complete artifact for one deliberation invocation."""

    model_config = ConfigDict(strict=True)

    run_id: str = Field(default_factory=_new_run_id)
    command: str = ""
    timestamp: str = Field(default_factory=_utc_now)
    model_id: str = ""
    prompt_version: str = ""

    # Evidence
    evidence_bundle: Optional[EvidenceBundle] = None
    context_hash: str = ""

    # Candidates and merge (query path)
    candidates: list[CandidateAnswer] = Field(default_factory=list)
    merged_answer: Optional[MergedAnswer] = None

    # Review findings (review path)
    review_findings: list[ReviewFinding] = Field(default_factory=list)

    # Final output
    final_text: str = ""

    # Cost and timing
    token_cost: int = 0
    wall_time_ms: int = 0
    unresolved_disagreement: bool = False
