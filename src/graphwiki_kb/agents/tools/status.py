"""status_kb agent tool."""

from __future__ import annotations

from dataclasses import asdict

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import StatusKbOutput
from graphwiki_kb.agents.tools._helpers import record_tool, tool_json


def run_status_kb(runtime: AgentRuntimeContext) -> str:
    """Return project and GraphRAG health."""
    services = runtime.services
    initialized = services.project.is_initialized()
    snapshot = services.status.snapshot(initialized=initialized)
    graph = services.graphrag_status.status()
    next_steps: list[str] = []
    if not initialized:
        next_steps.append("Run `kb init` to initialize the project.")
    if graph.graph_freshness_state != "fresh":
        next_steps.append("Run `kb update` to refresh a stale graph index.")
    summary = (
        f"Sources: {snapshot.source_count}, compiled: {snapshot.compiled_source_count}. "
        f"Graph freshness: {graph.graph_freshness_state}."
    )
    output = StatusKbOutput(
        initialized=initialized,
        summary=summary,
        graph_freshness=graph.graph_freshness_state,
        graph_stale_reasons=list(graph.graph_stale_reasons),
        next_steps=next_steps,
        details={
            "snapshot": asdict(snapshot),
            "graph": graph.to_dict(runtime.project_root),
        },
    )
    record_tool(
        runtime,
        tool_name="status_kb",
        ok=True,
        summary=summary,
        data=output.model_dump(),
    )
    return tool_json(output)
