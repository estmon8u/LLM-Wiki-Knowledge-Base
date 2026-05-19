"""Tests for research tool output separation."""

from __future__ import annotations

import json

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import ResearchInput
from graphwiki_kb.agents.tools.research import run_research


def test_research_output_separates_local_web_recommendations(
    test_project, monkeypatch, tmp_path
) -> None:
    from graphwiki_kb.agents.models import (
        AskKbOutput,
        ResearchOutput,
        SourceRecommendation,
    )

    research_output = ResearchOutput(
        run_id="research_test",
        question="RAG evaluation",
        local_answer=AskKbOutput(
            answer="Local coverage only.",
            method="global",
            claim_support="graph-index-answer",
        ),
        kb_gaps=["No recent benchmarks in KB"],
        web_findings=[],
        recommendations=[
            SourceRecommendation(
                id=1,
                title="Paper",
                url="https://example.com/paper.pdf",
                source_type="paper",
                retrieved_at="2026-05-19T00:00:00+00:00",
                why_add="Fills gap",
                knowledge_gap="benchmarks",
                novelty="high",
                confidence="high",
                ingestable=True,
            )
        ],
    )

    class _ResearchService:
        def run(self, **kwargs: object) -> ResearchOutput:
            return research_output

    monkeypatch.setattr(
        "graphwiki_kb.agents.tools.research.ResearchService",
        lambda *args, **kwargs: _ResearchService(),
    )
    runtime = AgentRuntimeContext(
        command_context=test_project.command_context,
        services=test_project.services,
    )
    text = run_research(
        runtime,
        ResearchInput(question="RAG evaluation", use_web=True),
    )
    data = json.loads(text)
    assert data["local_answer"]["answer"] == "Local coverage only."
    assert data["kb_gaps"]
    assert data["recommendations"][0]["id"] == 1
    assert "No sources were added" in runtime.tool_results[-1]["summary"]
