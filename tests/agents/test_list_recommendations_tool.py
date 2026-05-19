"""Tests for list_recommendations tool and store latest-run behavior."""

from __future__ import annotations

import json

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import ResearchRunRecord, SourceRecommendation
from graphwiki_kb.agents.tools.list_recommendations import run_list_recommendations
from graphwiki_kb.services.source_recommendation_store import SourceRecommendationStore


def _sample_recommendation(rec_id: int = 1) -> SourceRecommendation:
    return SourceRecommendation(
        id=rec_id,
        title=f"Source {rec_id}",
        url=f"https://example.com/{rec_id}",
        source_type="article",
        retrieved_at="2026-05-19T00:00:00+00:00",
        why_add="reason",
        knowledge_gap="gap",
        novelty="medium",
        confidence="medium",
        ingestable=True,
    )


def test_save_run_skips_latest_pointer_when_no_recommendations(test_project) -> None:
    store = SourceRecommendationStore(test_project.paths)
    with_recs = ResearchRunRecord(
        run_id="research_with_recs",
        question="q1",
        created_at="2026-05-19T10:00:00+00:00",
        local_answer={},
        recommendations=[_sample_recommendation(1)],
    )
    store.save_run(with_recs)
    empty = ResearchRunRecord(
        run_id="research_empty",
        question="meta query",
        created_at="2026-05-19T11:00:00+00:00",
        local_answer={},
        recommendations=[],
    )
    store.save_run(empty)
    pointer = json.loads(
        (test_project.paths.graph_dir / "runs" / "agent" / "latest.json").read_text()
    )
    assert pointer["run_id"] == "research_with_recs"
    assert store.latest_run_with_recommendations() is not None
    assert store.latest_run_with_recommendations().run_id == "research_with_recs"


def test_list_recommendations_returns_numbered_items(test_project) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.save_run(
        ResearchRunRecord(
            run_id="research_list_test",
            question="benchmarks",
            created_at="2026-05-19T12:00:00+00:00",
            local_answer={},
            recommendations=[_sample_recommendation(1), _sample_recommendation(2)],
        )
    )
    runtime = AgentRuntimeContext(
        command_context=test_project.command_context,
        services=test_project.services,
    )
    payload = json.loads(run_list_recommendations(runtime))
    assert payload["run_id"] == "research_list_test"
    assert len(payload["recommendations"]) == 2
    assert "No sources were added" in payload["message"]
