"""Agent tool: run the full ``kb update`` pipeline on the local KB.

This module belongs to ``graphwiki_kb.agents.tools.update`` and keeps
related behavior close to the command, service, model, provider, storage,
script, or test surface that uses it.

The tool drives the canonical :class:`UpdateService` so that
``update_kb`` matches ``kb update`` semantics — ingest, compile,
concept/search refresh, GraphRAG sync, and wiki export — instead of only
re-indexing GraphRAG. When the tool runs off the interpreter's main thread
it falls back to a ``kb update`` subprocess to keep GraphRAG's signal
handlers safe.

Mutations still require explicit user approval unless the agent runtime
was launched with ``auto_approve=True`` (i.e. ``kb agent --yes``).
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import (
    AgentToolResult,
    PendingApproval,
    UpdateInput,
    UpdateOutput,
)
from graphwiki_kb.agents.tools.main_thread import run_on_main_thread
from graphwiki_kb.services.update_service import (
    UpdateOptions,
    UpdateResult,
    UpdateService,
)

TOOL_NAME = "update_kb"
TOOL_DESCRIPTION = (
    "Run the full kb update pipeline: ingest pending sources, compile wiki "
    "pages, refresh search, and synchronize the GraphRAG index. Mutates the "
    "KB; requires user approval unless the agent runtime was launched with "
    "auto-approve."
)


def _build_update_service(runtime: AgentRuntimeContext) -> UpdateService:
    services = runtime.services
    return UpdateService(
        ingest_service=services.ingest,
        compile_service=services.compile,
        concept_service=services.concepts,
        search_service=services.search,
        config=runtime.command_context.config,
        graphrag_workspace_service=services.graphrag_workspace,
        graphrag_sync_service=services.graphrag_sync,
        graphrag_wiki_export_service=services.graphrag_wiki_export,
        wikigraph_index_service=services.wikigraph_index,
        export_service=services.export,
    )


def _graph_freshness(runtime: AgentRuntimeContext) -> tuple[str | None, list[str]]:
    try:
        status = runtime.services.graphrag_status.status()
    except Exception:  # pragma: no cover - defensive: status should not raise
        return None, []
    freshness = getattr(status, "graph_freshness_state", None)
    warnings: list[str] = []
    reasons = list(getattr(status, "graph_stale_reasons", []) or [])
    if freshness and freshness != "fresh":
        warnings.extend(reasons)
        warnings.append(f"Graph index is {freshness}. Review `kb status` for details.")
    return freshness, warnings


def _summarize_result(result: UpdateResult) -> tuple[str, str, dict[str, Any]]:
    """Return (summary, method, details) for an in-process update result."""
    method: str | None = None
    if result.graph_result is not None:
        sync = result.graph_result.sync_result
        if sync is not None:
            method = sync.decision.method or sync.decision.action
        elif result.graph_result.preflight_result is not None:
            method = result.graph_result.preflight_result.decision.method
    parts: list[str] = []
    if result.compile_result is not None:
        parts.append("compile")
    if result.search_refreshed:
        parts.append("search")
    if result.graph_result is not None:
        if result.graph_result.skipped:
            parts.append(f"graph(skipped:{result.graph_result.skip_reason})")
        else:
            parts.append(f"graph({method or 'sync'})")
    wikigraph_block: dict[str, Any] = {"ran": False, "skipped": False}
    if result.wikigraph_result is not None:
        wikigraph_block = {
            "ran": True,
            "skipped": False,
            "node_count": result.wikigraph_result.node_count,
            "edge_count": result.wikigraph_result.edge_count,
            "community_count": result.wikigraph_result.community_count,
            "source_count": result.wikigraph_result.source_count,
            "include_graphrag_export_pages": (
                result.wikigraph_result.include_graphrag_export_pages
            ),
            "warnings": list(result.wikigraph_result.warnings),
            "exported_artifact_count": len(
                getattr(result, "wikigraph_artifact_paths", []) or []
            ),
        }
        parts.append(
            f"wikigraph({result.wikigraph_result.node_count}n/"
            f"{result.wikigraph_result.edge_count}e)"
        )
    elif result.wikigraph_skipped:
        wikigraph_block = {
            "ran": False,
            "skipped": True,
            "skip_reason": result.wikigraph_skip_reason,
        }
        parts.append(f"wikigraph(skipped:{result.wikigraph_skip_reason})")
    if not parts:
        summary = "Update produced no changes."
    else:
        summary = "Update completed: " + ", ".join(parts) + "."
    details: dict[str, Any] = {
        "compile": result.compile_result is not None,
        "search_refreshed": result.search_refreshed,
        "graph": result.graph_result is not None,
        "graph_skipped": (
            result.graph_result.skipped if result.graph_result is not None else None
        ),
        "wikigraph": wikigraph_block,
    }
    return summary, method or "", details


def _run_inprocess(runtime: AgentRuntimeContext, payload: UpdateInput) -> UpdateOutput:
    service = _build_update_service(runtime)
    options = UpdateOptions(
        force=payload.force,
        graph_method=payload.graph_method,
        no_graph=payload.no_graph,
        graph_only=payload.graph_only,
        wikigraph=payload.wikigraph,
        wikigraph_include_graphrag_export_pages=(
            payload.wikigraph_include_graphrag_export_pages
        ),
        wikigraph_include_normalized_text_units=(
            payload.wikigraph_include_normalized_text_units
        ),
        export_wikigraph_artifacts=payload.export_wikigraph_artifacts,
    )
    result = service.run(options)
    summary, method, details = _summarize_result(result)
    freshness, staleness = _graph_freshness(runtime)
    return UpdateOutput(
        ok=result.ok,
        summary=summary,
        method=method or None,
        next_action="kb status",
        graph_freshness=freshness,
        staleness_warnings=staleness,
        details=details,
    )


def _kb_cli_base(runtime: AgentRuntimeContext) -> list[str]:
    return [
        sys.executable,
        "-m",
        "graphwiki_kb.cli",
        "--project-root",
        str(runtime.command_context.project_root),
    ]


def _run_subprocess(runtime: AgentRuntimeContext, payload: UpdateInput) -> UpdateOutput:
    command = [*_kb_cli_base(runtime), "update"]
    if payload.graph_only:
        command.append("--graph-only")
    if payload.no_graph:
        command.append("--no-graph")
    if payload.force:
        command.append("--force")
    if payload.graph_method != "auto":
        command.extend(["--graph-method", payload.graph_method])
    if payload.wikigraph is True:
        command.append("--wikigraph")
    elif payload.wikigraph is False:
        command.append("--no-wikigraph")
    if payload.wikigraph_include_graphrag_export_pages:
        command.append("--wikigraph-include-graphrag-export-pages")
    if payload.wikigraph_include_normalized_text_units is True:
        command.append("--wikigraph-normalized-text")
    elif payload.wikigraph_include_normalized_text_units is False:
        command.append("--no-wikigraph-normalized-text")
    if payload.export_wikigraph_artifacts is True:
        command.append("--export-wikigraph-artifacts")
    elif payload.export_wikigraph_artifacts is False:
        command.append("--no-export-wikigraph-artifacts")
    completed = subprocess.run(
        command,
        cwd=str(runtime.command_context.project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    freshness, staleness = _graph_freshness(runtime)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        return UpdateOutput(
            ok=False,
            summary="kb update subprocess failed.",
            diagnostics=[detail or f"exit code {completed.returncode}"],
            graph_freshness=freshness,
            staleness_warnings=staleness,
            details={"mode": "subprocess", "exit_code": completed.returncode},
        )
    return UpdateOutput(
        ok=True,
        summary="kb update completed via subprocess.",
        next_action="kb status",
        graph_freshness=freshness,
        staleness_warnings=staleness,
        details={"mode": "subprocess"},
    )


def run_update_kb(
    runtime: AgentRuntimeContext,
    payload: UpdateInput,
) -> UpdateOutput:
    """Plan, approve, and (when authorized) run a full kb update."""
    if not runtime.auto_approve and not payload.dry_run:
        approval = PendingApproval(
            tool_name=TOOL_NAME,
            summary=(
                "Run `kb update` to ingest pending sources, compile wiki pages, "
                "refresh search, and synchronize the GraphRAG index."
            ),
            payload={
                "force": payload.force,
                "graph_method": payload.graph_method,
                "no_graph": payload.no_graph,
                "graph_only": payload.graph_only,
                "wikigraph": payload.wikigraph,
                "wikigraph_include_graphrag_export_pages": (
                    payload.wikigraph_include_graphrag_export_pages
                ),
                "export_wikigraph_artifacts": payload.export_wikigraph_artifacts,
            },
        )
        runtime.add_pending_approval(approval)
        runtime.record_tool_result(
            AgentToolResult(
                tool_name=TOOL_NAME,
                ok=True,
                summary="awaiting approval before update",
                data=approval.payload,
            )
        )
        return UpdateOutput(
            ok=False,
            summary="Approval required before running `kb update`.",
            next_action="approve",
        )

    if payload.dry_run:
        freshness, staleness = _graph_freshness(runtime)
        output = UpdateOutput(
            ok=True,
            summary="Dry run: no changes were made.",
            next_action="run without --dry-run to apply",
            graph_freshness=freshness,
            staleness_warnings=staleness,
            details={"dry_run": True},
        )
        runtime.record_tool_result(
            AgentToolResult(
                tool_name=TOOL_NAME,
                ok=True,
                summary=output.summary,
                data=output.model_dump(),
            )
        )
        return output

    try:
        output = run_on_main_thread(
            lambda: _run_inprocess(runtime, payload),
            fallback=lambda: _run_subprocess(runtime, payload),
        )
    except Exception as exc:  # pragma: no cover - covered by tests via patching
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

    runtime.record_tool_result(
        AgentToolResult(
            tool_name=TOOL_NAME,
            ok=output.ok,
            summary=output.summary,
            data=output.model_dump(),
        )
    )
    return output
