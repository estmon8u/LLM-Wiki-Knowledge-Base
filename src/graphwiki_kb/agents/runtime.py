"""Build the OpenAI Agents SDK runtime that powers the kb agent command.

This module belongs to `graphwiki_kb.agents.runtime` and keeps related
behavior close to the command, service, model, provider, storage, script, or
test surface that uses it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.prompts import KB_AGENT_INSTRUCTIONS
from graphwiki_kb.agents.tool_registry import build_agent_tools

if TYPE_CHECKING:
    from agents import Agent


class AgentRuntimeError(RuntimeError):
    """Raised when the OpenAI Agents SDK runtime cannot be constructed."""


def is_agents_sdk_available() -> bool:
    """Return True if the openai-agents package is importable."""
    try:
        import agents  # noqa: F401
    except ImportError:
        return False
    return True


def build_kb_agent(
    *,
    model: str = "gpt-5.5",
    instructions: str = KB_AGENT_INSTRUCTIONS,
    allow_writes: bool = True,
    extra_tools: list[Any] | None = None,
) -> Agent:
    """Build the manager-style KB agent."""
    if not is_agents_sdk_available():
        raise AgentRuntimeError(
            "openai-agents is not installed. Install with: pip install openai-agents"
        )
    from agents import Agent

    tools = build_agent_tools(allow_writes=allow_writes)
    if extra_tools:
        tools.extend(extra_tools)
    return Agent(
        name="GraphWiki KB Agent",
        instructions=instructions,
        model=model,
        tools=tools,
    )


def build_session(
    *,
    session_id: str | None,
    storage_path: str | None,
) -> Any | None:
    """Build a SQLiteSession when an id and storage path are provided."""
    if not session_id or not storage_path:
        return None
    if not is_agents_sdk_available():
        return None
    from agents import SQLiteSession

    return SQLiteSession(session_id, storage_path)


async def run_agent(
    *,
    agent: Agent,
    prompt: str,
    runtime: AgentRuntimeContext,
    session: Any | None = None,
    max_turns: int = 8,
) -> Any:
    """Run the agent once and return the SDK ``RunResult``.

    Tools record their progress on ``runtime`` directly, so callers can build
    a durable run record even before reading the SDK result.
    """
    if not is_agents_sdk_available():
        raise AgentRuntimeError(
            "openai-agents is not installed. Install with: pip install openai-agents"
        )
    from agents import Runner

    return await Runner.run(
        agent,
        prompt,
        context=runtime,
        session=session,
        max_turns=max_turns,
    )
