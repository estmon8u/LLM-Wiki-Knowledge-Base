"""Tests for Pydantic models used by the kb agent."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphwiki_kb.agents.models import (
    AskKbInput,
    AskKbOutput,
    PendingApproval,
    ResearchInput,
    ResearchRunRecord,
    SourceRecommendation,
    WebFinding,
)


def test_ask_kb_input_defaults_to_auto() -> None:
    payload = AskKbInput(question="What is RAG?")
    assert payload.method == "auto"
    assert payload.save is False
    assert payload.show_source_trace is False


def test_ask_kb_input_rejects_unknown_method() -> None:
    with pytest.raises(ValidationError):
        AskKbInput(question="x", method="nope")  # type: ignore[arg-type]


def test_ask_kb_input_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        AskKbInput(question="x", method="auto", surprise=True)  # type: ignore[call-arg]


def test_ask_kb_output_round_trip() -> None:
    output = AskKbOutput(
        answer="Hi",
        method="local",
        staleness_warnings=["stale"],
        claim_support="cited-graph-answer",
    )
    dump = output.model_dump()
    assert dump["method"] == "local"
    assert dump["staleness_warnings"] == ["stale"]
    assert dump["claim_support"] == "cited-graph-answer"


def test_research_input_validates_search_context_size() -> None:
    with pytest.raises(ValidationError):
        ResearchInput(question="x", search_context_size="huge")  # type: ignore[arg-type]


def test_research_input_validates_max_recommendations_range() -> None:
    with pytest.raises(ValidationError):
        ResearchInput(question="x", max_recommendations=0)
    with pytest.raises(ValidationError):
        ResearchInput(question="x", max_recommendations=100)


def test_source_recommendation_normalizes_optional_fields() -> None:
    rec = SourceRecommendation(
        id=1,
        title="Paper",
        url="https://example.com/paper",
        source_type="paper",
        retrieved_at="2026-05-19T12:34:56+00:00",
        why_add="fills a KB gap",
    )
    assert rec.suggested_tags == []
    assert rec.citation_urls == []
    assert rec.novelty == "medium"


def test_research_run_record_serializes_recommendations() -> None:
    record = ResearchRunRecord(
        run_id="research_20260519T120000Z_x",
        question="What",
        created_at="2026-05-19T12:00:00+00:00",
        local_answer={"answer": "..."},
        kb_gaps=["a gap"],
        web_findings=[
            WebFinding(
                title="t",
                url="https://example.com",
                summary="s",
                relevance="high",
            )
        ],
        recommendations=[
            SourceRecommendation(
                id=1,
                title="p",
                url="https://example.com/p",
                source_type="paper",
                retrieved_at="2026-05-19T12:00:00+00:00",
                why_add="r",
            )
        ],
    )
    payload = record.model_dump()
    assert payload["kb_gaps"] == ["a gap"]
    assert payload["web_findings"][0]["url"] == "https://example.com"
    assert payload["recommendations"][0]["id"] == 1


def test_pending_approval_contains_payload() -> None:
    approval = PendingApproval(
        tool_name="ingest_recommendation",
        summary="Ingest 1 recommendation",
        payload={"ids": [1]},
    )
    assert approval.payload["ids"] == [1]
