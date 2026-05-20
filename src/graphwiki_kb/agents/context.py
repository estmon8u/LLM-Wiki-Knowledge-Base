"""Runtime context for the kb agent control plane.

This module belongs to `graphwiki_kb.agents.context` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graphwiki_kb.agents.models import AgentToolResult, PendingApproval
    from graphwiki_kb.models.command_models import CommandContext
    from graphwiki_kb.services.container import ServiceContainer
    from graphwiki_kb.services.source_recommendation_store import (
        SourceRecommendationStore,
    )


@dataclass
class AgentRuntimeContext:
    """Per-run context shared with every kb agent tool.

    Tools resolve KB paths and services through this object. The instance is
    passed to the Agents SDK as the ``context`` argument and is therefore
    available on every ``RunContextWrapper.context``.
    """

    command_context: CommandContext
    services: ServiceContainer
    recommendation_store: SourceRecommendationStore
    auto_approve: bool = False
    show_plan: bool = False
    session_id: str | None = None
    tool_results: list[AgentToolResult] = field(default_factory=list)
    pending_approvals: list[PendingApproval] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def record_tool_result(self, result: AgentToolResult) -> None:
        """Append a tool result to the run-scoped trace."""
        self.tool_results.append(result)

    def add_pending_approval(self, approval: PendingApproval) -> None:
        """Append a pending approval to the run-scoped queue."""
        self.pending_approvals.append(approval)
