"""Agent tool: KB and graph health snapshot.

This module belongs to `graphwiki_kb.agents.tools.status` and keeps related
behavior close to the command, service, model, provider, storage, script, or
test surface that uses it.
"""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import AgentToolResult, StatusOutput
from graphwiki_kb.wikigraph.status_service import wikigraph_status

TOOL_NAME = "status"
TOOL_DESCRIPTION = (
    "Return a small health snapshot for the KB: source/compile counts, GraphRAG "
    "freshness, WikiGraphRAG build state, staleness reasons, and the recommended "
    "next action."
)


def run_status(runtime: AgentRuntimeContext) -> StatusOutput:
    """Build a small StatusOutput projection from existing services."""
    services = runtime.services
    initialized = services.project.is_initialized()
    snapshot = services.status.snapshot(initialized=initialized)
    graph_status = services.graphrag_status.status()
    staleness_reasons = [
        reason.rstrip(".") for reason in graph_status.graph_stale_reasons or []
    ]
    wikigraph = wikigraph_status(services.project.paths) if initialized else None
    output = StatusOutput(
        project_initialized=initialized,
        source_count=snapshot.source_count,
        compiled_source_count=snapshot.compiled_source_count,
        concept_count=snapshot.concept_page_count,
        analysis_count=snapshot.analysis_page_count,
        graph_state=graph_status.last_index_method or "unknown",
        graph_freshness=graph_status.graph_freshness_state,
        next_action=graph_status.next_action,
        staleness_reasons=staleness_reasons,
        wikigraph_built=wikigraph.built if wikigraph is not None else False,
        wikigraph_node_count=wikigraph.node_count if wikigraph is not None else 0,
        wikigraph_chunk_count=wikigraph.chunk_count if wikigraph is not None else 0,
    )
    runtime.record_tool_result(
        AgentToolResult(
            tool_name=TOOL_NAME,
            ok=True,
            summary=(
                f"graph freshness={output.graph_freshness}, "
                f"wikigraph built={output.wikigraph_built}, "
                f"next_action={output.next_action}"
            ),
            data={
                "project_initialized": output.project_initialized,
                "source_count": output.source_count,
                "graph_freshness": output.graph_freshness,
                "wikigraph_built": output.wikigraph_built,
                "wikigraph_node_count": output.wikigraph_node_count,
            },
        )
    )
    return output
