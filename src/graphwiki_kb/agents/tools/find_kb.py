"""Agent tool: search graph artifacts, the wiki index, and WikiGraphRAG.

This module belongs to `graphwiki_kb.agents.tools.find_kb` and keeps related
behavior close to the command, service, model, provider, storage, script, or
test surface that uses it.

The tool mirrors the folded ``kb find --engine ...`` behavior. ``engine``
selects the backend(s); ``auto`` and ``all`` fuse GraphRAG entity/relationship
artifacts, the wiki search index, and the WikiGraphRAG contexts.
"""

from __future__ import annotations

from collections import defaultdict

from graphwiki_kb.agents.context import AgentRuntimeContext
from graphwiki_kb.agents.models import (
    AgentToolResult,
    FindKbInput,
    FindKbOutput,
    FindKbResult,
)
from graphwiki_kb.models.wiki_models import SearchResult
from graphwiki_kb.services.wikigraph_query_service import WikiGraphQueryError
from graphwiki_kb.wikigraph.models import WikiGraphRetrievedContext

TOOL_NAME = "find_kb"
TOOL_DESCRIPTION = (
    "Search GraphRAG entity/relationship artifacts, the maintained wiki "
    "index, and the WikiGraphRAG backend for pages, entities, and topics. "
    "Returns ranked snippets, not full answers. The optional `engine` "
    "argument restricts the search to a single backend; the default `auto` "
    "fuses all three via reciprocal rank fusion."
)


def run_find_kb(
    runtime: AgentRuntimeContext,
    payload: FindKbInput,
) -> FindKbOutput:
    """Search across graph artifacts, the wiki search index, and WikiGraphRAG."""
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

    engine = payload.engine
    candidate_limit = max(payload.limit * 2, 10)
    diagnostics: list[str] = []

    run_graph = engine in {"auto", "all", "graphrag"}
    run_wiki = engine in {"auto", "all", "wiki"}
    run_wikigraph = engine in {"auto", "all", "wikigraph"}

    graph_results: list[SearchResult] = []
    wiki_results: list[SearchResult] = []
    wikigraph_contexts: list[WikiGraphRetrievedContext] = []
    if run_graph:
        try:
            graph_results = services.graphrag_find.search(query, limit=candidate_limit)
        except Exception as exc:
            diagnostics.append(f"graph search unavailable: {exc.__class__.__name__}")
    if run_wiki:
        try:
            wiki_results = services.search.search(
                query,
                limit=candidate_limit,
                include_concepts=True,
            )
        except Exception as exc:
            diagnostics.append(f"wiki search unavailable: {exc.__class__.__name__}")
    if run_wikigraph:
        try:
            find_result = services.wikigraph_query.find(query, method="auto")
            wikigraph_contexts = list(find_result.contexts)
        except WikiGraphQueryError as exc:
            if engine == "wikigraph":
                runtime.record_tool_result(
                    AgentToolResult(
                        tool_name=TOOL_NAME,
                        ok=False,
                        summary="find_kb failed (wikigraph index missing)",
                        data={"engine": engine, "query": query},
                        error=str(exc),
                    )
                )
                return FindKbOutput(
                    query=query,
                    results=[],
                    graph_diagnostics=[str(exc)],
                )
            diagnostics.append(f"WikiGraphRAG unavailable: {exc}")
        except Exception as exc:
            diagnostics.append(
                f"WikiGraphRAG search unavailable: {exc.__class__.__name__}"
            )

    merged = _merge_results(
        graph_results=graph_results,
        wiki_results=wiki_results,
        wikigraph_contexts=wikigraph_contexts,
        limit=payload.limit,
    )

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
            data={"query": query, "count": len(merged), "engine": engine},
        )
    )
    return FindKbOutput(
        query=query,
        results=merged,
        graph_diagnostics=diagnostics,
    )


def _merge_results(
    *,
    graph_results: list[SearchResult],
    wiki_results: list[SearchResult],
    wikigraph_contexts: list[WikiGraphRetrievedContext],
    limit: int,
) -> list[FindKbResult]:
    """Merge graph + wiki + wikigraph hits via reciprocal rank fusion.

    The merge mirrors :mod:`graphwiki_kb.commands.find` so the agent and the
    CLI produce comparable ranked lists. Items with the same (retriever, path)
    are deduplicated; the highest source score wins.
    """
    candidates: dict[tuple[str, str], FindKbResult] = {}
    rrf: defaultdict[tuple[str, str], float] = defaultdict(float)

    def _key(retriever: str, path: str) -> tuple[str, str]:
        return retriever, path.casefold()

    def _add_search_result(retriever: str, results: list[SearchResult]) -> None:
        for rank, item in enumerate(results, start=1):
            key = _key(retriever, str(item.path))
            rrf[key] += 1 / (60 + rank)
            existing = candidates.get(key)
            converted = FindKbResult(
                title=item.title,
                path=str(item.path),
                score=float(item.score),
                snippet=item.snippet,
                retriever=retriever,
            )
            if existing is None or converted.score > existing.score:
                candidates[key] = converted

    _add_search_result("graph", graph_results)
    _add_search_result("wiki", wiki_results)
    for rank, ctx in enumerate(wikigraph_contexts, start=1):
        snippet = (ctx.text or "").replace("\n", " ").strip()
        if len(snippet) > 240:
            snippet = snippet[:240].rstrip() + "..."
        path = ctx.path or ctx.node_id
        key = _key("wikigraph", path)
        rrf[key] += 1 / (60 + rank)
        converted = FindKbResult(
            title=ctx.title,
            path=path,
            score=float(ctx.score),
            snippet=snippet or ctx.title,
            retriever="wikigraph",
        )
        existing = candidates.get(key)
        if existing is None or converted.score > existing.score:
            candidates[key] = converted

    ranked_keys = sorted(
        candidates,
        key=lambda key: (
            rrf[key],
            candidates[key].score,
            candidates[key].title.casefold(),
        ),
        reverse=True,
    )
    return [candidates[key] for key in ranked_keys[:limit]]
