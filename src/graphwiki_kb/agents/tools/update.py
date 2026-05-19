"""update_kb agent tool."""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import UpdateKbInput, UpdateKbOutput
from graphwiki_kb.agents.tools._helpers import record_tool, tool_json
from graphwiki_kb.services.update_service import UpdateOptions, UpdateService


def run_update_kb(runtime: AgentRuntimeContext, params: UpdateKbInput) -> str:
    """Run the full kb update pipeline."""
    update_service = UpdateService(
        ingest_service=runtime.services.ingest,
        compile_service=runtime.services.compile,
        concept_service=runtime.services.concepts,
        search_service=runtime.services.search,
        config=runtime.command_context.config,
        graphrag_workspace_service=runtime.services.graphrag_workspace,
        graphrag_sync_service=runtime.services.graphrag_sync,
        graphrag_wiki_export_service=runtime.services.graphrag_wiki_export,
    )
    options = UpdateOptions(
        graph_method=params.graph_method,
        no_graph=params.no_graph,
        graph_only=params.graph_only,
    )
    result = update_service.run(options)
    graph_status = runtime.services.graphrag_status.status()
    staleness: list[str] = []
    if graph_status.graph_freshness_state != "fresh":
        staleness.extend(graph_status.graph_stale_reasons or [])
        staleness.append(
            f"Graph index is {graph_status.graph_freshness_state}. "
            "Review `kb status` for details."
        )
    summary = "Update completed." if result.ok else "Update did not produce changes."
    output = UpdateKbOutput(
        ok=result.ok,
        summary=summary,
        graph_freshness=graph_status.graph_freshness_state,
        staleness_warnings=staleness,
        details={
            "compile": result.compile_result is not None,
            "graph": result.graph_result is not None,
        },
    )
    record_tool(
        runtime,
        tool_name="update_kb",
        ok=result.ok,
        summary=summary,
        data=output.model_dump(),
    )
    return tool_json(output)
