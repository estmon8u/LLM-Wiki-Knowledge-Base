"""Agent tool coverage for WikiGraphRAG engine paths."""

from __future__ import annotations

from graphwiki_kb.agents.models import AskKbInput, FindKbInput
from graphwiki_kb.agents.tools.ask_kb import run_ask_kb
from graphwiki_kb.agents.tools.find_kb import run_find_kb


def test_agent_find_kb_invalid_engine(runtime) -> None:
    payload = FindKbInput.model_construct(query="x", engine="graphrag")
    out = run_find_kb(runtime, payload)
    assert out.results == []
    assert any("Unsupported find engine" in d for d in out.graph_diagnostics)


def test_agent_find_kb_wikigraph_missing_index(runtime) -> None:
    out = run_find_kb(
        runtime,
        FindKbInput(query="missing corpus topic", engine="wikigraph", limit=3),
    )
    assert out.results == []
    assert runtime.tool_results[-1].ok is False


def test_agent_ask_kb_wikigraph_unexpected_error(runtime, monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("index corrupt")

    monkeypatch.setattr(
        "graphwiki_kb.agents.tools.ask_kb.run_wikigraph_ask",
        _boom,
    )
    result = run_ask_kb(
        runtime,
        AskKbInput(question="q", engine="wikigraph", method="local"),
    )
    assert result.answer == ""
    assert any("index corrupt" in w for w in result.staleness_warnings)
