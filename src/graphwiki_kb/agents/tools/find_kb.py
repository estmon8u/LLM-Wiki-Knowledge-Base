"""find_kb agent tool."""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import FindKbInput, FindKbOutput, FindKbResultItem
from graphwiki_kb.agents.tools._helpers import record_tool, tool_json
from graphwiki_kb.commands.find import _graph_find_diagnostics, _merge_results


def run_find_kb(runtime: AgentRuntimeContext, params: FindKbInput) -> str:
    """Search graph artifacts and the wiki index."""
    services = runtime.services
    graph_status = services.graphrag_status.status().to_dict(runtime.project_root)
    diagnostics = _graph_find_diagnostics(graph_status)
    candidate_limit = max(params.limit * 4, 20)
    graph_results = services.graphrag_find.search(params.query, limit=candidate_limit)
    wiki_results = services.search.search(
        params.query,
        limit=candidate_limit,
        include_concepts=True,
    )
    merged = _merge_results(graph_results, wiki_results, limit=params.limit)
    items = [
        FindKbResultItem(
            title=result.title,
            path=result.path,
            score=float(result.score),
            kind=("graph" if str(result.path).startswith("graph://") else "wiki"),
            snippet=result.snippet or "",
        )
        for result in merged
    ]
    output = FindKbOutput(
        query=params.query,
        diagnostics=diagnostics,
        results=items,
    )
    record_tool(
        runtime,
        tool_name="find_kb",
        ok=True,
        summary=f"Found {len(items)} result(s).",
        data=output.model_dump(),
    )
    return tool_json(output)
