"""Agent tool: KB and graph health snapshot.

This module belongs to `graphwiki_kb.agents.tools.status` and keeps related
behavior close to the command, service, model, provider, storage, script, or
test surface that uses it.
"""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import (
    AgentToolResult,
    StatusOutput,
    WikiGraphStatusBlock,
)

TOOL_NAME = "status"
TOOL_DESCRIPTION = (
    "Return a small health snapshot for the KB: source/compile counts, graph "
    "freshness, staleness reasons, the WikiGraphRAG index state, and the "
    "recommended next action."
)


def _wikigraph_block(runtime: AgentRuntimeContext) -> WikiGraphStatusBlock | None:
    """Build a compact :class:`WikiGraphStatusBlock` from the index service."""
    service = getattr(runtime.services, "wikigraph_index", None)
    if service is None:
        return None
    try:
        snapshot = service.status()
    except Exception as exc:  # pragma: no cover - defensive
        return WikiGraphStatusBlock(
            initialized=False,
            readable=False,
            message=f"WikiGraphRAG status unavailable: {exc.__class__.__name__}",
        )
    initialized = bool(snapshot.get("initialized", False))
    if not initialized:
        return WikiGraphStatusBlock(
            initialized=False,
            message=str(snapshot.get("message", "")),
        )
    if not snapshot.get("readable", True):
        return WikiGraphStatusBlock(
            initialized=True,
            readable=False,
            message=str(snapshot.get("message", "")),
        )
    return WikiGraphStatusBlock(
        initialized=True,
        readable=True,
        built_at=str(snapshot.get("built_at") or "") or None,
        node_count=int(snapshot.get("node_count", 0) or 0),
        edge_count=int(snapshot.get("edge_count", 0) or 0),
        chunk_count=int(snapshot.get("chunk_count", 0) or 0),
        text_unit_count=int(snapshot.get("text_unit_count", 0) or 0),
        document_count=int(snapshot.get("document_count", 0) or 0),
        entity_count=int(snapshot.get("entity_count", 0) or 0),
        community_count=int(snapshot.get("community_count", 0) or 0),
        source_count=int(snapshot.get("source_count", 0) or 0),
        include_graphrag_export_pages=bool(
            snapshot.get("include_graphrag_export_pages", False)
        ),
        include_normalized_text_units=bool(
            snapshot.get("include_normalized_text_units", False)
        ),
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
        wikigraph=_wikigraph_block(runtime),
    )
    runtime.record_tool_result(
        AgentToolResult(
            tool_name=TOOL_NAME,
            ok=True,
            summary=(
                f"graph freshness={output.graph_freshness}, "
                f"next_action={output.next_action}"
            ),
            data={
                "project_initialized": output.project_initialized,
                "source_count": output.source_count,
                "graph_freshness": output.graph_freshness,
                "wikigraph_initialized": (
                    output.wikigraph.initialized
                    if output.wikigraph is not None
                    else False
                ),
            },
        )
    )
    return output
