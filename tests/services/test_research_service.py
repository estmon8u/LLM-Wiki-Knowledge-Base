"""Additional research service coverage."""

from __future__ import annotations

from graphwiki_kb.agents.models import AskKbOutput
from graphwiki_kb.services.graphrag_query_service import GraphRAGQueryAnswer
from graphwiki_kb.services.research_service import extract_kb_gaps, project_ask_output


def test_extract_kb_gaps_flags_empty_answer() -> None:
    local = AskKbOutput(answer="", method="local", claim_support="no-answer")
    gaps = extract_kb_gaps(local, "topic")
    assert any("no substantive" in gap.lower() for gap in gaps)


def test_project_ask_output_maps_fields() -> None:
    answer = GraphRAGQueryAnswer(
        question="q",
        answer="a",
        raw_output="",
        method="global",
        created_at="2026-05-19T00:00:00+00:00",
        index_run_id="x",
        command=(),
        stdout="",
        stderr="",
        graph_input_hash="h",
        planner="p",
        route_reason="r",
        claim_support="cited-graph-answer",
        staleness_warnings=["stale"],
    )
    projected = project_ask_output(answer)
    assert projected.method == "global"
    assert projected.staleness_warnings == ["stale"]
