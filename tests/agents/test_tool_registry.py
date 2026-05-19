"""Tests for the agent tool registry (callable + FunctionTool builders)."""

from __future__ import annotations

import pytest

from graphwiki_kb.agents.tool_registry import (
    ALL_TOOL_NAMES,
    CALLABLE_REGISTRY,
    READ_ONLY_TOOL_NAMES,
    WRITE_TOOL_NAMES,
    build_agent_tools,
)


def test_registry_lists_split_into_read_and_write() -> None:
    assert set(READ_ONLY_TOOL_NAMES).issubset(set(ALL_TOOL_NAMES))
    assert set(WRITE_TOOL_NAMES).issubset(set(ALL_TOOL_NAMES))
    assert not set(READ_ONLY_TOOL_NAMES).intersection(WRITE_TOOL_NAMES)


def test_callable_registry_covers_all_known_tools() -> None:
    assert set(CALLABLE_REGISTRY.keys()) == set(ALL_TOOL_NAMES)


def test_build_agent_tools_returns_eight_tools() -> None:
    tools = build_agent_tools(allow_writes=True)
    names = [t.name for t in tools]
    assert names == [
        "ask_kb",
        "find_kb",
        "status",
        "lint",
        "review",
        "research",
        "ingest_recommendation",
        "update_kb",
    ]


def test_build_agent_tools_can_drop_write_tools() -> None:
    tools = build_agent_tools(allow_writes=False)
    names = [t.name for t in tools]
    assert "ingest_recommendation" not in names
    assert "update_kb" not in names
    assert "ask_kb" in names
    assert "research" in names


def test_runtime_helper_requires_agent_runtime_context() -> None:
    from graphwiki_kb.agents.tool_registry import _runtime

    class FakeCtx:
        context = object()

    with pytest.raises(RuntimeError):
        _runtime(FakeCtx())
