"""Agent tool: search graph artifacts, wiki index, or WikiGraphRAG.

This module belongs to `graphwiki_kb.agents.tools.find_kb` and keeps related
behavior close to the command, service, model, provider, storage, script, or
test surface that uses it.
"""

from __future__ import annotations

import click

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import (
    AgentToolResult,
    FindKbInput,
    FindKbOutput,
    FindKbResult,
)
from graphwiki_kb.commands.retrieval_engines import (
    normalize_find_engine,
    normalize_wikigraph_method,
)
from graphwiki_kb.models.wiki_models import SearchResult

TOOL_NAME = "find_kb"
TOOL_DESCRIPTION = (
    "Search the KB for pages, entities, and topics. Default engine `graph` "
    "searches GraphRAG artifacts plus the wiki index; `wikigraph` searches the "
    "custom WikiGraphRAG index from kb update. Returns ranked snippets."
)


def run_find_kb(
    runtime: AgentRuntimeContext,
    payload: FindKbInput,
) -> FindKbOutput:
    """Search using the requested retrieval engine."""
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

    try:
        engine = normalize_find_engine(payload.engine)
    except click.ClickException as exc:
        runtime.record_tool_result(
            AgentToolResult(
                tool_name=TOOL_NAME,
                ok=False,
                summary="find_kb rejected invalid engine",
                error=str(exc),
            )
        )
        return FindKbOutput(
            query=query,
            results=[],
            graph_diagnostics=[str(exc)],
        )

    if engine == "wikigraph":
        return _run_wikigraph_find(runtime, payload, query=query)

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
            data={"query": query, "count": len(merged), "engine": "graph"},
        )
    )
    return FindKbOutput(
        query=query,
        results=merged,
        graph_diagnostics=diagnostics,
    )


def _run_wikigraph_find(
    runtime: AgentRuntimeContext,
    payload: FindKbInput,
    *,
    query: str,
) -> FindKbOutput:
    diagnostics: list[str] = []
    try:
        method = normalize_wikigraph_method(payload.method)
    except click.ClickException as exc:
        runtime.record_tool_result(
            AgentToolResult(
                tool_name=TOOL_NAME,
                ok=False,
                summary="find_kb rejected invalid method",
                error=str(exc),
            )
        )
        return FindKbOutput(query=query, results=[], graph_diagnostics=[str(exc)])

    from graphwiki_kb.services.wikigraph_query_service import WikiGraphQueryFacade

    facade = WikiGraphQueryFacade(
        runtime.command_context.services.project.paths,
        runtime.command_context.config,
    )
    try:
        result_payload = facade.find(
            query,
            method=method,  # type: ignore[arg-type]
            limit=payload.limit,
        )
    except (ImportError, FileNotFoundError, click.ClickException) as exc:
        runtime.record_tool_result(
            AgentToolResult(
                tool_name=TOOL_NAME,
                ok=False,
                summary="find_kb failed (wikigraph)",
                error=str(exc),
            )
        )
        return FindKbOutput(query=query, results=[], graph_diagnostics=[str(exc)])

    results = _wikigraph_contexts_to_results(
        result_payload.get("contexts") or [],
        limit=payload.limit,
    )
    status = result_payload.get("status") or {}
    if not status.get("built"):
        diagnostics.append(
            "WikiGraphRAG index is not built; run `kb update` to build it."
        )
    for warning in result_payload.get("warnings") or []:
        diagnostics.append(str(warning))

    runtime.record_tool_result(
        AgentToolResult(
            tool_name=TOOL_NAME,
            ok=True,
            summary=f"find_kb returned {len(results)} WikiGraphRAG result(s)",
            data={
                "query": query,
                "count": len(results),
                "engine": "wikigraph",
                "method": result_payload.get("method"),
            },
        )
    )
    return FindKbOutput(query=query, results=results, graph_diagnostics=diagnostics)


def _wikigraph_contexts_to_results(
    contexts: list[dict[str, object]],
    *,
    limit: int,
) -> list[FindKbResult]:
    results: list[FindKbResult] = []
    for context in contexts[:limit]:
        title = str(context.get("title") or "Untitled")
        path = str(context.get("path") or context.get("node_id") or "")
        snippet = str(context.get("text") or "")[:240]
        results.append(
            FindKbResult(
                title=title,
                path=path,
                score=float(context.get("score") or 0.0),
                snippet=snippet,
                retriever="wikigraph",
            )
        )
    return results


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
