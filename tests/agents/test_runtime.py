"""Tests for the OpenAI Agents SDK runtime wiring."""

from __future__ import annotations

import asyncio

import pytest

from graphwiki_kb.agents.runtime import (
    AgentRuntimeError,
    build_kb_agent,
    build_session,
    is_agents_sdk_available,
    run_agent,
)
from graphwiki_kb.agents.tool_registry import (
    ALL_TOOL_NAMES,
    build_agent_tools,
)


def test_is_agents_sdk_available() -> None:
    assert is_agents_sdk_available() is True


def test_build_kb_agent_returns_real_agent() -> None:
    agent = build_kb_agent(model="test-model", allow_writes=False)
    tool_names = [t.name for t in agent.tools]
    assert agent.name == "GraphWiki KB Agent"
    assert agent.model == "test-model"
    assert "ingest_recommendation" not in tool_names
    assert "ask_kb" in tool_names


def test_build_kb_agent_includes_writes_when_requested() -> None:
    agent = build_kb_agent(allow_writes=True)
    names = [t.name for t in agent.tools]
    assert set(ALL_TOOL_NAMES).issubset(set(names))


def test_build_session_returns_none_without_id(tmp_path) -> None:
    assert build_session(session_id=None, storage_path=str(tmp_path / "x")) is None
    assert build_session(session_id="abc", storage_path=None) is None


def test_build_session_creates_sqlite_session(tmp_path) -> None:
    session = build_session(
        session_id="s1",
        storage_path=str(tmp_path / "sessions.sqlite"),
    )
    assert session is not None
    assert hasattr(session, "session_id")


def test_run_agent_executes_runner(monkeypatch, runtime) -> None:
    """Verify the run_agent helper delegates to the SDK Runner.run."""
    import agents as agents_sdk

    captured: dict = {}

    async def _fake_run(agent, prompt, *, context, session=None, max_turns=8):
        captured["prompt"] = prompt
        captured["context"] = context
        captured["max_turns"] = max_turns
        return type("Result", (), {"final_output": "ok"})()

    monkeypatch.setattr(agents_sdk.Runner, "run", _fake_run)
    agent = build_kb_agent(allow_writes=False)

    result = asyncio.run(
        run_agent(agent=agent, prompt="hi", runtime=runtime, max_turns=3)
    )
    assert result.final_output == "ok"
    assert captured["prompt"] == "hi"
    assert captured["context"] is runtime
    assert captured["max_turns"] == 3


def test_runtime_error_when_sdk_missing(monkeypatch, runtime) -> None:
    monkeypatch.setattr(
        "graphwiki_kb.agents.runtime.is_agents_sdk_available",
        lambda: False,
    )
    agent = object()
    with pytest.raises(AgentRuntimeError):
        asyncio.run(run_agent(agent=agent, prompt="hi", runtime=runtime))


def test_build_kb_agent_raises_when_sdk_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "graphwiki_kb.agents.runtime.is_agents_sdk_available",
        lambda: False,
    )
    with pytest.raises(AgentRuntimeError):
        build_kb_agent()


def test_function_tools_call_through_runtime_context(runtime) -> None:
    """Smoke test the FunctionTool wrappers by invoking them with a wrapper."""
    from agents import RunContextWrapper

    tools = {t.name: t for t in build_agent_tools(allow_writes=True)}
    wrapper = RunContextWrapper(context=runtime)

    status_invocation = asyncio.run(tools["status"].on_invoke_tool(wrapper, "{}"))
    assert "graph_freshness" in str(status_invocation)


def test_function_tool_reports_invalid_context_through_sdk() -> None:
    """The SDK surfaces our runtime check as an error string."""
    from agents import RunContextWrapper

    tools = {t.name: t for t in build_agent_tools(allow_writes=False)}
    wrapper = RunContextWrapper(context=object())
    result = asyncio.run(tools["status"].on_invoke_tool(wrapper, "{}"))
    assert "AgentRuntimeContext" in str(result)
