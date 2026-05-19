"""Tests for source recommendation store."""

from __future__ import annotations

from graphwiki_kb.agents.models import ResearchRunRecord, SourceRecommendation
from graphwiki_kb.services.source_recommendation_store import SourceRecommendationStore


def test_store_resolves_latest_run(test_project) -> None:
    store = SourceRecommendationStore(test_project.paths)
    record = ResearchRunRecord(
        run_id="research_test_run",
        question="memory benchmarks",
        created_at="2026-05-19T00:00:00+00:00",
        local_answer={"method": "local"},
        recommendations=[
            SourceRecommendation(
                id=1,
                title="Bench",
                url="https://example.com/bench",
                source_type="article",
                retrieved_at="2026-05-19T00:00:00+00:00",
                why_add="reason",
                knowledge_gap="gap",
                novelty="medium",
                confidence="medium",
                ingestable=True,
            ),
            SourceRecommendation(
                id=2,
                title="Other",
                url="https://example.com/other",
                source_type="article",
                retrieved_at="2026-05-19T00:00:00+00:00",
                why_add="reason",
                knowledge_gap="gap",
                novelty="low",
                confidence="low",
                ingestable=False,
            ),
        ],
    )
    store.save_run(record)
    loaded, recs = store.resolve_recommendations([2])
    assert loaded.run_id == "research_test_run"
    assert recs[0].id == 2
    assert store.latest_run_id() == "research_test_run"
