"""Shared helpers for agent tool implementations."""

from __future__ import annotations

import json
from typing import Any

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import AgentToolResult


def record_tool(
    runtime: AgentRuntimeContext,
    *,
    tool_name: str,
    ok: bool,
    summary: str,
    data: dict[str, object] | None = None,
    error: str | None = None,
) -> None:
    """Append a tool result to the runtime trace list."""
    runtime.tool_results.append(
        AgentToolResult(
            tool_name=tool_name,
            ok=ok,
            summary=summary,
            data=data or {},
            error=error,
        ).model_dump()
    )


def tool_json(payload: Any) -> str:
    """Serialize tool output for the agent."""
    if hasattr(payload, "model_dump"):
        return json.dumps(payload.model_dump(), indent=2)
    return json.dumps(payload, indent=2, default=str)
