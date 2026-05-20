"""Agent tool: search graph artifacts and the wiki index.

This module belongs to `graphwiki_kb.agents.tools.find_kb` and keeps related
behavior close to the command, service, model, provider, storage, script, or
test surface that uses it.
"""

from __future__ import annotations

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import (
    AgentToolResult,
    FindKbInput,
    FindKbOutput,
    FindKbResult,
)
from graphwiki_kb.models.wiki_models import SearchResult

TOOL_NAME = "find_kb"
TOOL_DESCRIPTION = (
    "Search GraphRAG entity/relationship artifacts and the maintained wiki index "
    "for pages, entities, and topics. Returns ranked snippets, not full answers."
)


def run_find_kb(
    runtime: AgentRuntimeContext,
    payload: FindKbInput,
) -> FindKbOutput:
    """Search across graph artifacts and the wiki search index."""
    services = runtime.services
    query = payload.query.strip()
    if not query:
        runtime.record_tool_result(
            AgentToolResult(
                tool_name=TOOL_NAME,
                ok=False,
                summary="find_kb requires a non-empty query",
                error="empty query",
            )
        )
        return FindKbOutput(query="", results=[], graph_diagnostics=[])

    candidate_limit = max(payload.limit * 2, 10)
    diagnostics: list[str] = []
    try:
        graph_results = services.graphrag_find.search(query, limit=candidate_limit)
    except Exception as exc:
        diagnostics.append(f"graph search unavailable: {exc.__class__.__name__}")
        graph_results = []
    try:
        wiki_results = services.search.search(
            query,
            limit=candidate_limit,
            include_concepts=True,
        )
    except Exception as exc:
        diagnostics.append(f"wiki search unavailable: {exc.__class__.__name__}")
        wiki_results = []
    merged = _merge_results(graph_results, wiki_results, limit=payload.limit)

    try:
        graph_status = services.graphrag_status.status()
        if not graph_status.entities_present:
            diagnostics.append(
                "GraphRAG entity artifacts are missing; "
                "run `kb update` for full search."
            )
    except Exception:
        pass

    runtime.record_tool_result(
        AgentToolResult(
            tool_name=TOOL_NAME,
            ok=True,
            summary=f"find_kb returned {len(merged)} result(s)",
            data={"query": query, "count": len(merged)},
        )
    )
    return FindKbOutput(
        query=query,
        results=merged,
        graph_diagnostics=diagnostics,
    )


def _merge_results(
    graph_results: list[SearchResult],
    wiki_results: list[SearchResult],
    *,
    limit: int,
) -> list[FindKbResult]:
    seen: set[tuple[str, str]] = set()
    merged: list[FindKbResult] = []

    def _add(result: SearchResult, retriever: str) -> None:
        key = (retriever, str(result.path).casefold())
        if key in seen:
            return
        seen.add(key)
        merged.append(
            FindKbResult(
                title=result.title,
                path=str(result.path),
                score=float(result.score),
                snippet=result.snippet,
                retriever="graph" if retriever == "graph" else "wiki",
            )
        )

    for result in graph_results:
        _add(result, "graph")
        if len(merged) >= limit:
            return merged
    for result in wiki_results:
        _add(result, "wiki")
        if len(merged) >= limit:
            return merged
    return merged
