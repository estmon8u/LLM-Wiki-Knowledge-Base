"""Tests for the ``list_recommendations`` agent tool."""

from __future__ import annotations

from typing import Any

from graphwiki_kb.agents.models import (
    ListRecommendationsInput,
    ResearchRunRecord,
    SourceRecommendation,
)
from graphwiki_kb.agents.tools.list_recommendations import (
    run_list_recommendations,
)


def _make_rec(idx: int, title: str = "T") -> SourceRecommendation:
    return SourceRecommendation(
        id=idx,
        title=f"{title}-{idx}",
        url=f"https://example.test/{idx}",
        retrieved_at="2026-05-19T00:00:00Z",
        why_add="fills gap",
    )


def _save_run(
    runtime: Any,
    run_id: str,
    *,
    recommendations: list[SourceRecommendation],
    created_at: str,
) -> None:
    record = ResearchRunRecord(
        run_id=run_id,
        question=f"q-{run_id}",
        created_at=created_at,
        local_answer={"answer": "ok"},
        kb_gaps=[],
        web_findings=[],
        recommendations=recommendations,
    )
    runtime.recommendation_store.save(record)


def test_list_recommendations_returns_empty_when_no_runs(runtime: Any) -> None:
    output = run_list_recommendations(runtime, ListRecommendationsInput())
    assert output.recommendations == []
    assert output.run_id is None
    assert "No research run with recommendations" in output.message
    assert runtime.tool_results[-1].tool_name == "list_recommendations"


def test_list_recommendations_returns_latest_with_recs(runtime: Any) -> None:
    _save_run(
        runtime,
        "research_20260519T010000Z_topic_a",
        recommendations=[_make_rec(1), _make_rec(2)],
        created_at="2026-05-19T01:00:00Z",
    )

    output = run_list_recommendations(runtime, ListRecommendationsInput())

    assert output.run_id == "research_20260519T010000Z_topic_a"
    assert [rec.id for rec in output.recommendations] == [1, 2]
    assert "Found 2 recommendation(s)" in output.message
    assert "No sources were added" in output.message


def test_list_recommendations_skips_empty_runs_when_choosing_latest(
    runtime: Any,
) -> None:
    _save_run(
        runtime,
        "research_20260519T010000Z_topic_a",
        recommendations=[_make_rec(1)],
        created_at="2026-05-19T01:00:00Z",
    )
    # Even a later run with zero recommendations should not mask the prior
    # run when the user just asks for "previous recommendations".
    _save_run(
        runtime,
        "research_20260519T020000Z_topic_b",
        recommendations=[],
        created_at="2026-05-19T02:00:00Z",
    )

    output = run_list_recommendations(runtime, ListRecommendationsInput())

    assert output.run_id == "research_20260519T010000Z_topic_a"
    assert [rec.id for rec in output.recommendations] == [1]


def test_list_recommendations_explicit_run_id(runtime: Any) -> None:
    _save_run(
        runtime,
        "research_20260519T010000Z_topic_a",
        recommendations=[_make_rec(1)],
        created_at="2026-05-19T01:00:00Z",
    )
    _save_run(
        runtime,
        "research_20260519T020000Z_topic_b",
        recommendations=[_make_rec(7)],
        created_at="2026-05-19T02:00:00Z",
    )

    output = run_list_recommendations(
        runtime,
        ListRecommendationsInput(run_id="research_20260519T010000Z_topic_a"),
    )

    assert output.run_id == "research_20260519T010000Z_topic_a"
    assert [rec.id for rec in output.recommendations] == [1]


def test_list_recommendations_unknown_run_id_reports_error(runtime: Any) -> None:
    output = run_list_recommendations(
        runtime,
        ListRecommendationsInput(run_id="does-not-exist"),
    )
    assert output.recommendations == []
    assert "No research run" in output.message
    # No exception should reach the agent; the tool result should still be
    # recorded so the conversation has a trail.
    assert runtime.tool_results[-1].tool_name == "list_recommendations"
