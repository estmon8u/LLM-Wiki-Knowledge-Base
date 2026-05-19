"""Runtime context passed to KB agent tools and orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

from graphwiki_kb.models.command_models import CommandContext
from graphwiki_kb.services.container import ServiceContainer


@dataclass
class AgentRuntimeContext:
    """Execution context for one agent command or REPL session."""

    command_context: CommandContext
    services: ServiceContainer
    auto_approve: bool = False
    show_plan: bool = False
    session_id: str | None = None
    tool_results: list[dict[str, object]] = field(default_factory=list)

    @property
    def config(self) -> dict[str, object]:
        """Project configuration from the command context."""
        return self.command_context.config

    @property
    def project_root(self):
        """Project root path."""
        return self.command_context.project_root
