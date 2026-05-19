"""Tool registry approval flag tests."""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.tool_registry import agents_sdk_available, build_tools


def test_build_tools_auto_approve_skips_write_approval(test_project) -> None:
    if not agents_sdk_available():
        return
    runtime = AgentRuntimeContext(
        command_context=test_project.command_context,
        services=test_project.services,
        auto_approve=True,
    )
    tools = build_tools(runtime)
    by_name = {tool.name: tool for tool in tools}
    assert by_name["update_kb"].needs_approval is False
    assert by_name["ingest_recommendation"].needs_approval is False
