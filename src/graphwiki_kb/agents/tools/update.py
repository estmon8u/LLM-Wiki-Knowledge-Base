"""Agent tool: refresh the GraphRAG index for the local KB.

This module belongs to `graphwiki_kb.agents.tools.update` and keeps related
behavior close to the command, service, model, provider, storage, script, or
test surface that uses it.
"""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import (
    AgentToolResult,
    PendingApproval,
    UpdateInput,
    UpdateOutput,
)
from graphwiki_kb.services.graphrag_sync_service import GraphRAGSyncError

TOOL_NAME = "update_kb"
TOOL_DESCRIPTION = (
    "Refresh the GraphRAG index after new sources have been added. Mutates "
    "the graph; requires user approval unless the agent runtime was launched "
    "with auto-approve."
)


def run_update_kb(
    runtime: AgentRuntimeContext,
    payload: UpdateInput,
) -> UpdateOutput:
    """Plan and (if approved) run a graph sync."""
    sync_service = runtime.services.graphrag_sync
    if not runtime.auto_approve and not payload.dry_run:
        approval = PendingApproval(
            tool_name=TOOL_NAME,
            summary="Run `kb update` to refresh the GraphRAG index.",
            payload={"force": payload.force},
        )
        runtime.add_pending_approval(approval)
        runtime.record_tool_result(
            AgentToolResult(
                tool_name=TOOL_NAME,
                ok=True,
                summary="awaiting approval before update",
                data={"force": payload.force},
            )
        )
        return UpdateOutput(
            ok=False,
            summary="Approval required before refreshing the graph index.",
            next_action="approve",
        )
    try:
        result = sync_service.sync(
            force=payload.force,
            dry_run=payload.dry_run,
            run_index=not payload.dry_run,
        )
    except GraphRAGSyncError as exc:
        runtime.record_tool_result(
            AgentToolResult(
                tool_name=TOOL_NAME,
                ok=False,
                summary="kb update failed",
                error=str(exc),
            )
        )
        return UpdateOutput(
            ok=False,
            summary=f"update failed: {exc}",
            diagnostics=[str(exc)],
        )

    method = result.decision.method or result.decision.action
    output = UpdateOutput(
        ok=True,
        summary=(
            f"update action={result.decision.action}, "
            f"reason={result.decision.reason}"
        ),
        method=method,
        next_action="kb status",
    )
    runtime.record_tool_result(
        AgentToolResult(
            tool_name=TOOL_NAME,
            ok=True,
            summary=output.summary,
            data={
                "action": result.decision.action,
                "method": method,
                "input_changed": result.decision.input_changed,
                "config_changed": result.decision.config_changed,
            },
        )
    )
    return output
