"""Tests for ask_kb tool projection."""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import AskKbInput
from graphwiki_kb.agents.tools.ask_kb import run_ask_kb
from graphwiki_kb.services.graphrag_query_service import GraphRAGQueryAnswer


def test_ask_kb_projects_graph_answer(test_project, monkeypatch) -> None:
    answer = GraphRAGQueryAnswer(
        question="What is GraphRAG?",
        answer="GraphRAG is a graph-based RAG system.",
        raw_output="",
        method="local",
        created_at="2026-05-19T00:00:00+00:00",
        index_run_id="run-1",
        command=("query",),
        stdout="",
        stderr="",
        graph_input_hash="abc",
        planner="local",
        route_reason="matched",
        claim_support="cited-graph-answer",
    )

    class _Controller:
        def ask(self, question: str, **kwargs: object) -> GraphRAGQueryAnswer:
            assert question == "What is GraphRAG?"
            return answer

    test_project.services.graph_ask_controller = _Controller()  # type: ignore[assignment]
    runtime = AgentRuntimeContext(
        command_context=test_project.command_context,
        services=test_project.services,
    )
    payload = run_ask_kb(runtime, AskKbInput(question="What is GraphRAG?"))
    assert "GraphRAG is a graph-based RAG system" in payload
    assert "local" in payload
    assert runtime.tool_results[0]["ok"] is True
