"""Tests for recommendation store latest-run resolution."""

from __future__ import annotations

from graphwiki_kb.agents.models import ResearchRunRecord, SourceRecommendation
from graphwiki_kb.services.source_recommendation_store import SourceRecommendationStore


def test_resolve_falls_back_to_latest_with_recommendations(test_project) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.save_run(
        ResearchRunRecord(
            run_id="research_good",
            question="good",
            created_at="2026-05-19T10:00:00+00:00",
            local_answer={},
            recommendations=[
                SourceRecommendation(
                    id=2,
                    title="B",
                    url="https://example.com/b",
                    source_type="article",
                    retrieved_at="2026-05-19T00:00:00+00:00",
                    why_add="r",
                    knowledge_gap="g",
                    novelty="low",
                    confidence="medium",
                    ingestable=True,
                )
            ],
        )
    )
    store.save_run(
        ResearchRunRecord(
            run_id="research_empty",
            question="empty",
            created_at="2026-05-19T11:00:00+00:00",
            local_answer={},
            recommendations=[],
        )
    )
    record, recs = store.resolve_recommendations([2])
    assert record.run_id == "research_good"
    assert recs[0].id == 2
