"""OpenAI Agents SDK orchestration for kb agent."""

from __future__ import annotations

import os
from typing import Any

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import AgentRunRecord, AgentToolResult
from graphwiki_kb.agents.prompts import KB_AGENT_INSTRUCTIONS
from graphwiki_kb.agents.tool_registry import build_tools
from graphwiki_kb.services.project_service import utc_now_iso
from graphwiki_kb.services.source_recommendation_store import agent_runs_dir


def build_kb_agent(tools: list[Any], *, model: str) -> Any:
    """Construct the GraphWiki KB manager agent."""
    from agents import Agent

    return Agent(
        name="GraphWiki KB Agent",
        instructions=KB_AGENT_INSTRUCTIONS,
        model=model,
        tools=tools,
    )


def _configure_tracing(enabled: bool) -> None:
    if not enabled:
        os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")
    else:
        os.environ.pop("OPENAI_AGENTS_DISABLE_TRACING", None)


def _session_for_runtime(runtime: AgentRuntimeContext) -> Any | None:
    agent_cfg = dict(runtime.config.get("agent", {}) or {})
    backend = str(agent_cfg.get("session_backend", "sqlite"))
    if backend != "sqlite":
        return None
    from agents import SQLiteSession

    runs_dir = agent_runs_dir(runtime.services.project.paths)
    runs_dir.mkdir(parents=True, exist_ok=True)
    session_id = runtime.session_id or "default"
    return SQLiteSession(session_id, db_path=runs_dir / "sessions.sqlite")


def _save_local_trace(
    runtime: AgentRuntimeContext,
    *,
    run_id: str,
    prompt: str,
    final_output: str,
    pending_approvals: list[dict[str, object]],
) -> None:
    agent_cfg = dict(runtime.config.get("agent", {}) or {})
    if not agent_cfg.get("save_runs", True):
        return
    runs_dir = agent_runs_dir(runtime.services.project.paths)
    runs_dir.mkdir(parents=True, exist_ok=True)
    tool_results = [
        AgentToolResult.model_validate(item) for item in runtime.tool_results
    ]
    record = AgentRunRecord(
        run_id=run_id,
        prompt=prompt,
        created_at=utc_now_iso(),
        tool_results=tool_results,
        final_output=final_output,
        pending_approvals=pending_approvals,
    )
    stamp = utc_now_iso().replace(":", "").replace("-", "")
    path = runs_dir / f"agent-run-{stamp}.json"
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")


def _format_approval_prompt(interruption: Any) -> str:
    name = getattr(interruption, "name", None) or "unknown_tool"
    arguments = getattr(interruption, "arguments", None)
    return f"The agent wants to run `{name}` with arguments: {arguments}"


def run_agent_turn(
    runtime: AgentRuntimeContext,
    prompt: str,
    *,
    approval_callback: Any | None = None,
) -> tuple[str, list[dict[str, object]]]:
    """Execute one agent turn, handling approval interruptions."""
    from agents import Runner

    agent_cfg = dict(runtime.config.get("agent", {}) or {})
    model = str(agent_cfg.get("model", "gpt-5.4-nano"))
    max_turns = int(agent_cfg.get("max_turns", 8))
    _configure_tracing(bool(agent_cfg.get("trace", True)))

    tools = build_tools(runtime)
    agent = build_kb_agent(tools, model=model)
    session = _session_for_runtime(runtime)
    run_id = f"agent_{utc_now_iso().replace(':', '').replace('-', '')}"

    result = Runner.run_sync(
        agent,
        prompt,
        max_turns=max_turns,
        session=session,
    )
    pending: list[dict[str, object]] = []

    while result.interruptions:
        state = result.to_state()
        for interruption in result.interruptions:
            approved = False
            if approval_callback is not None:
                approved = bool(approval_callback(interruption))
            elif runtime.auto_approve:
                approved = True
            else:
                message = _format_approval_prompt(interruption)
                answer = input(f"{message}\nProceed? [y/N]: ").strip().lower()
                approved = answer in {"y", "yes"}
            if approved:
                state.approve(interruption)
                pending.append(
                    {
                        "tool": getattr(interruption, "name", None),
                        "status": "approved",
                    }
                )
            else:
                state.reject(interruption)
                pending.append(
                    {
                        "tool": getattr(interruption, "name", None),
                        "status": "rejected",
                    }
                )
        result = Runner.run_sync(agent, state, max_turns=max_turns, session=session)

    final_output = str(result.final_output or "")
    _save_local_trace(
        runtime,
        run_id=run_id,
        prompt=prompt,
        final_output=final_output,
        pending_approvals=pending,
    )
    return final_output, pending
