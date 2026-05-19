"""update_kb agent tool."""

from __future__ import annotations

import subprocess
import sys

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import UpdateKbInput, UpdateKbOutput
from graphwiki_kb.agents.tools._helpers import record_tool, tool_json
from graphwiki_kb.agents.tools.main_thread import is_main_thread, run_on_main_thread
from graphwiki_kb.services.update_service import UpdateOptions, UpdateService


def _build_update_service(runtime: AgentRuntimeContext) -> UpdateService:
    return UpdateService(
        ingest_service=runtime.services.ingest,
        compile_service=runtime.services.compile,
        concept_service=runtime.services.concepts,
        search_service=runtime.services.search,
        config=runtime.command_context.config,
        graphrag_workspace_service=runtime.services.graphrag_workspace,
        graphrag_sync_service=runtime.services.graphrag_sync,
        graphrag_wiki_export_service=runtime.services.graphrag_wiki_export,
    )


def _format_update_output(
    runtime: AgentRuntimeContext,
    *,
    ok: bool,
    summary: str,
    compile_ran: bool,
    graph_ran: bool,
) -> str:
    graph_status = runtime.services.graphrag_status.status()
    staleness: list[str] = []
    if graph_status.graph_freshness_state != "fresh":
        staleness.extend(graph_status.graph_stale_reasons or [])
        staleness.append(
            f"Graph index is {graph_status.graph_freshness_state}. "
            "Review `kb status` for details."
        )
    output = UpdateKbOutput(
        ok=ok,
        summary=summary,
        graph_freshness=graph_status.graph_freshness_state,
        staleness_warnings=staleness,
        details={"compile": compile_ran, "graph": graph_ran},
    )
    record_tool(
        runtime,
        tool_name="update_kb",
        ok=ok,
        summary=summary,
        data=output.model_dump(),
    )
    return tool_json(output)


def _run_update_inprocess(
    runtime: AgentRuntimeContext,
    params: UpdateKbInput,
) -> str:
    update_service = _build_update_service(runtime)
    options = UpdateOptions(
        graph_method=params.graph_method,
        no_graph=params.no_graph,
        graph_only=params.graph_only,
    )
    result = update_service.run(options)
    summary = "Update completed." if result.ok else "Update did not produce changes."
    return _format_update_output(
        runtime,
        ok=result.ok,
        summary=summary,
        compile_ran=result.compile_result is not None,
        graph_ran=result.graph_result is not None,
    )


def _kb_cli_base(runtime: AgentRuntimeContext) -> list[str]:
    return [
        sys.executable,
        "-m",
        "graphwiki_kb.cli",
        "--project-root",
        str(runtime.project_root),
    ]


def _run_update_subprocess(
    runtime: AgentRuntimeContext,
    params: UpdateKbInput,
) -> str:
    """Run `kb update` in a subprocess (GraphRAG uses signals; agent tools run in threads)."""
    command = _kb_cli_base(runtime) + ["update"]
    if params.graph_only:
        command.append("--graph-only")
    if params.no_graph:
        command.append("--no-graph")
    if params.graph_method != "auto":
        command.extend(["--graph-method", params.graph_method])
    completed = subprocess.run(
        command,
        cwd=runtime.project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        record_tool(
            runtime,
            tool_name="update_kb",
            ok=False,
            summary="Update failed in subprocess.",
            error=detail or f"exit code {completed.returncode}",
        )
        raise RuntimeError(
            detail or f"kb update failed with exit code {completed.returncode}"
        )
    summary = "Update completed via subprocess."
    return _format_update_output(
        runtime,
        ok=True,
        summary=summary,
        compile_ran=not params.graph_only,
        graph_ran=not params.no_graph,
    )


def run_update_kb(runtime: AgentRuntimeContext, params: UpdateKbInput) -> str:
    """Run the full kb update pipeline."""
    return run_on_main_thread(
        lambda: _run_update_inprocess(runtime, params),
        fallback=lambda: _run_update_subprocess(runtime, params),
    )
