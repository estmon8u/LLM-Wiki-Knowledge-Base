"""Tests for the ask_kb agent tool."""

from __future__ import annotations

from graphwiki_kb.agents.models import AskKbInput, AskKbOutput
from graphwiki_kb.agents.tools import ask_kb as ask_kb_tool
from graphwiki_kb.agents.tools.ask_kb import run_ask_kb
from graphwiki_kb.services.graph_ask_controller_service import GraphAskControllerError
from graphwiki_kb.services.graphrag_query_service import GraphRAGQueryAnswer
from graphwiki_kb.wikigraph.models import WikiGraphAnswer


def _build_answer(**overrides) -> GraphRAGQueryAnswer:
    defaults: dict = {
        "question": "q",
        "answer": "hello world [Data: Sources (1)]",
        "raw_output": "hello world",
        "method": "local",
        "created_at": "2026-05-19T00:00:00+00:00",
        "index_run_id": "run-123",
        "command": ("graphrag", "query"),
        "stdout": "",
        "stderr": "",
        "graph_input_hash": "abc",
        "input_manifest_hash": "def",
        "planner": "auto",
        "route_reason": "matched",
        "route_confidence": "high",
        "claim_support": "cited-graph-answer",
        "source_trace": {"input_path": "graph/input", "graph_input_hash": "abc"},
        "staleness_warnings": [],
    }
    defaults.update(overrides)
    return GraphRAGQueryAnswer(**defaults)


class _FakeController:
    def __init__(self, answer=None, error=None) -> None:
        self.answer = answer
        self.error = error
        self.calls: list[dict] = []

    def ask(self, question, **kwargs):
        self.calls.append({"question": question, **kwargs})
        if self.error is not None:
            raise self.error
        return self.answer


def test_run_ask_kb_projects_graph_answer(runtime) -> None:
    controller = _FakeController(answer=_build_answer())
    runtime.services.graph_ask_controller = controller  # type: ignore[assignment]

    result = run_ask_kb(runtime, AskKbInput(question="What is RAG?"))

    assert isinstance(result, AskKbOutput)
    assert result.answer.startswith("hello world")
    assert result.method == "local"
    assert result.planner == "auto"
    assert result.claim_support == "cited-graph-answer"
    assert result.source_trace == {
        "input_path": "graph/input",
        "graph_input_hash": "abc",
    }
    assert controller.calls == [
        {"question": "What is RAG?", "method": "auto", "save": False}
    ]
    assert runtime.tool_results[-1].data.get("engine") == "graphrag"
    assert runtime.tool_results
    trace = runtime.tool_results[-1]
    assert trace.tool_name == "ask_kb"
    assert trace.ok is True


def test_run_ask_kb_returns_failure_output_when_controller_raises(runtime) -> None:
    runtime.services.graph_ask_controller = _FakeController(  # type: ignore[assignment]
        error=GraphAskControllerError("no credentials"),
    )

    result = run_ask_kb(runtime, AskKbInput(question="x"))

    assert result.answer == ""
    assert result.claim_support == "no-answer"
    assert any("no credentials" in s for s in result.staleness_warnings)
    assert runtime.tool_results[-1].ok is False
    assert runtime.tool_results[-1].error == "no credentials"


def test_run_ask_kb_catches_unexpected_exceptions(runtime) -> None:
    """Regression: unexpected errors (e.g. FileNotFoundError from a broken
    subprocess fallback) must not propagate to the SDK as raw stack traces.

    They must produce a structured AskKbOutput and a recorded tool trace.
    """
    runtime.services.graph_ask_controller = _FakeController(
        error=FileNotFoundError(2, "No such file or directory", "graphrag"),
    )

    result = run_ask_kb(runtime, AskKbInput(question="x"))

    assert isinstance(result, AskKbOutput)
    assert result.answer == ""
    assert result.claim_support == "no-answer"
    assert any("FileNotFoundError" in warning for warning in result.staleness_warnings)
    last_trace = runtime.tool_results[-1]
    assert last_trace.tool_name == "ask_kb"
    assert last_trace.ok is False
    assert "FileNotFoundError" in (last_trace.error or "")


def test_run_ask_kb_propagates_save_flag(runtime) -> None:
    controller = _FakeController(answer=_build_answer(saved_path="wiki/analysis/x.md"))
    runtime.services.graph_ask_controller = controller  # type: ignore[assignment]

    result = run_ask_kb(
        runtime,
        AskKbInput(question="q", save=True, method="global"),
    )

    assert controller.calls[0]["save"] is True
    assert controller.calls[0]["method"] == "global"
    assert result.saved_path == "wiki/analysis/x.md"


def test_run_ask_kb_wikigraph_engine(monkeypatch, runtime) -> None:
    answer = WikiGraphAnswer(
        method="local",
        question="q",
        answer="wikigraph answer",
        contexts=[],
        citations=[{"title": "A"}],
        trace=[{"step": "method", "value": "local"}],
        warnings=[],
    )

    def _fake_wikigraph_ask(ctx, question, *, method, save_answer):
        assert question == "What is WikiGraph?"
        assert method == "local"
        assert save_answer is False
        return answer

    monkeypatch.setattr(ask_kb_tool, "run_wikigraph_ask", _fake_wikigraph_ask)

    result = run_ask_kb(
        runtime,
        AskKbInput(question="What is WikiGraph?", engine="wikigraph", method="local"),
    )

    assert result.answer == "wikigraph answer"
    assert result.method == "local"
    assert result.claim_support == "cited-graph-answer"
    assert runtime.tool_results[-1].data.get("engine") == "wikigraph"


def test_run_ask_kb_rejects_drift_lite_on_graphrag(runtime) -> None:
    result = run_ask_kb(
        runtime,
        AskKbInput(question="q", method="drift-lite"),
    )
    assert result.answer == ""
    assert result.claim_support == "no-answer"
    assert any("drift-lite" in warning for warning in result.staleness_warnings)
