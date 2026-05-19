"""Public service interface for the kb agent command."""

from __future__ import annotations

from typing import Any

from graphwiki_kb.agents.config_helpers import config_section
from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import AgentRunResult, AgentToolResult
from graphwiki_kb.agents.runtime import run_agent_turn
from graphwiki_kb.agents.tool_registry import agents_sdk_available, tool_names
from graphwiki_kb.models.command_models import CommandContext
from graphwiki_kb.services.container import ServiceContainer


class AgentService:
    """Runs natural-language KB agent sessions."""

    def __init__(self, config: dict[str, Any], services: ServiceContainer) -> None:
        self.config = config
        self.services = services

    def ensure_available(self) -> None:
        """Raise when agent support is disabled or not installed."""
        agent_cfg = config_section(self.config, "agent")
        if not agent_cfg.get("enabled", True):
            raise RuntimeError(
                "kb agent is disabled in kb.config.yaml (agent.enabled)."
            )
        if not agents_sdk_available():
            raise RuntimeError(
                "The agent extra is not installed. "
                "Run: poetry install --extras agent"
            )

    def run_once(
        self,
        prompt: str,
        *,
        command_context: CommandContext,
        auto_approve: bool = False,
        show_plan: bool = False,
        session_id: str | None = None,
        approval_callback: Any | None = None,
    ) -> AgentRunResult:
        """Run a single agent prompt to completion."""
        self.ensure_available()
        runtime = AgentRuntimeContext(
            command_context=command_context,
            services=self.services,
            auto_approve=auto_approve,
            show_plan=show_plan,
            session_id=session_id,
        )
        planned = tool_names(runtime) if show_plan else []
        final_output, pending = run_agent_turn(
            runtime,
            prompt,
            approval_callback=approval_callback,
        )
        tool_results = [
            AgentToolResult.model_validate(item) for item in runtime.tool_results
        ]
        return AgentRunResult(
            run_id=session_id or "agent-run",
            final_output=final_output,
            tool_results=tool_results,
            pending_approvals=pending,
            planned_tools=planned,
        )
