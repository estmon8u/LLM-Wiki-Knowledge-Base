"""Tests for agent tool registry."""

from __future__ import annotations

import pytest

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.tool_registry import build_tools, tool_names


def test_tool_names_includes_core_tools(test_project) -> None:
    runtime = AgentRuntimeContext(
        command_context=test_project.command_context,
        services=test_project.services,
    )
    names = tool_names(runtime)
    assert "ask_kb" in names
    assert "research" in names
    assert "update_kb" in names
    assert "list_recommendations" in names


def test_build_tools_requires_agents_sdk(test_project) -> None:
    runtime = AgentRuntimeContext(
        command_context=test_project.command_context,
        services=test_project.services,
    )
    try:
        tools = build_tools(runtime)
    except RuntimeError:
        pytest.skip("openai-agents not installed")
    assert len(tools) >= 8
