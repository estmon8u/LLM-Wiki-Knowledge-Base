"""Click command implementation for the kb find command."""

from __future__ import annotations

from collections import defaultdict

import click

from graphwiki_kb.commands.common import (
    console,
    emit_json,
    make_table,
    require_initialized,
)
from graphwiki_kb.models.command_models import CommandContext, CommandSpec
from graphwiki_kb.models.wiki_models import SearchResult
from graphwiki_kb.services.wikigraph_query_service import WikiGraphQueryError
from graphwiki_kb.wikigraph.models import WikiGraphRetrievedContext

SUMMARY = (
    "Search direct GraphRAG artifacts, the maintained wiki index, and the "
    "WikiGraphRAG backend."
)

ENGINE_CHOICES = ("auto", "graphrag", "wiki", "wikigraph", "all")


def build_spec(_: CommandContext | None = None) -> CommandSpec:
    """Builds the command registry specification for this module.

    Args:
        _: Value value used by the operation.

    Returns:
        CommandSpec produced by the operation.
    """
    return CommandSpec(name="find", summary=SUMMARY)


def create_command() -> click.Command:
    """Creates the Click command exposed by this module.

    Returns:
        click.Command produced by the operation.
    """

    @click.command(name="find", help=SUMMARY, short_help="Search graph and wiki.")
    @click.argument("query_terms", nargs=-1)
    @click.option(
        "--limit",
        default=5,
        show_default=True,
        type=click.IntRange(min=1, max=100),
    )
    @click.option(
        "--engine",
        type=click.Choice(ENGINE_CHOICES),
        default="auto",
        show_default=True,
        help=(
            "Backend to query. `auto`/`all` fuse GraphRAG artifacts, the wiki "
            "index, and WikiGraphRAG via reciprocal rank fusion."
        ),
    )
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def command(
        command_context: CommandContext,
        query_terms: tuple[str, ...],
        limit: int,
        engine: str,
        as_json: bool,
    ) -> None:
        """Command.

        Args:
            command_context: Command context value used by the operation.
            query_terms: Query terms value used by the operation.
            limit: Maximum number of results to return or process.
            engine: Retrieval backend selector.
            as_json: As json value used by the operation.
        """
        require_initialized(command_context)
        if not query_terms:
            raise click.ClickException("Provide at least one search term.")
        search_service = command_context.services.search
        graph_find_service = command_context.services.graphrag_find
        wikigraph_query_service = command_context.services.wikigraph_query
        graph_status = command_context.services.graphrag_status.status().to_dict(
            command_context.project_root
        )
        diagnostics = _graph_find_diagnostics(graph_status)
        query = " ".join(query_terms).strip()
        candidate_limit = max(limit * 4, 20)

        run_graph = engine in {"auto", "all", "graphrag"}
        run_wiki = engine in {"auto", "all", "wiki"}
        run_wikigraph = engine in {"auto", "all", "wikigraph"}

        graph_results: list[SearchResult] = []
        wiki_results: list[SearchResult] = []
        wikigraph_results: list[SearchResult] = []
        wikigraph_contexts: list[WikiGraphRetrievedContext] = []

        if run_graph:
            graph_results = graph_find_service.search(query, limit=candidate_limit)
        if run_wiki:
            wiki_results = search_service.search(
                query,
                limit=candidate_limit,
                include_concepts=True,
            )
        if run_wikigraph:
            try:
                find_result = wikigraph_query_service.find(query, method="auto")
                wikigraph_contexts = find_result.contexts
                wikigraph_results = [
                    _wikigraph_to_search_result(ctx) for ctx in wikigraph_contexts
                ]
            except WikiGraphQueryError as exc:
                if engine == "wikigraph":
                    raise click.ClickException(str(exc)) from exc
                diagnostics.append(f"WikiGraphRAG unavailable: {exc}")

        results = _merge_results(
            graph_results, wiki_results, wikigraph_results, limit=limit
        )

        if as_json:
            emit_json(
                {
                    "retriever": _retriever_label(engine),
                    "engine": engine,
                    "query": query,
                    "diagnostics": diagnostics,
                    "results": [_search_result_payload(result) for result in results],
                    "wikigraph": {
                        "contexts": [ctx.model_dump() for ctx in wikigraph_contexts],
                    },
                }
            )
            return

        for diagnostic in diagnostics:
            console.print(f"[yellow]{diagnostic}[/yellow]")
        if not results:
            console.print("No graph artifacts or wiki pages matched that query.")
            return

        table = make_table(
            columns=[
                ("Title", {"style": "bold"}),
                ("Path", {}),
                ("Score", {"justify": "right"}),
                ("Snippet", {"style": "dim"}),
            ],
            rows=[
                (result.title, result.path, f"{result.score:.2f}", result.snippet)
                for result in results
            ],
            title="Graph and Wiki Search Results",
        )
        console.print(table)

    return command


def _wikigraph_to_search_result(
    ctx: WikiGraphRetrievedContext,
) -> SearchResult:
    snippet = ctx.text.replace("\n", " ").strip()
    if len(snippet) > 240:
        snippet = snippet[:240].rstrip() + "..."
    return SearchResult(
        title=ctx.title,
        path=ctx.path or ctx.node_id,
        score=ctx.score,
        snippet=snippet or ctx.title,
        section=ctx.section,
        chunk_index=ctx.chunk_index,
        retriever="wikigraph",
    )


def _retriever_label(engine: str) -> str:
    if engine == "wikigraph":
        return "wikigraph"
    if engine == "graphrag":
        return "graphrag-artifacts"
    if engine == "wiki":
        return "wiki-index"
    # auto/all preserves the existing JSON contract.
    return "graph-and-wiki-index"


def _search_result_payload(result: SearchResult) -> dict[str, object]:
    path = str(result.path)
    if result.retriever:
        retriever = result.retriever
    elif path.startswith("graph://"):
        retriever = "graphrag-artifacts"
    else:
        retriever = "wiki-index"
    return {
        "retriever": retriever,
        "title": result.title,
        "path": result.path,
        "score": result.score,
        "snippet": result.snippet,
        "section": result.section,
        "chunk_index": result.chunk_index,
    }


def _merge_results(
    graph_results: list[SearchResult],
    wiki_results: list[SearchResult],
    wikigraph_results: list[SearchResult] | None = None,
    *,
    limit: int,
) -> list[SearchResult]:
    wikigraph_results = wikigraph_results or []
    candidates: dict[tuple[str, str, str, str], SearchResult] = {}
    rrf_scores: defaultdict[tuple[str, str, str, str], float] = defaultdict(float)
    for source_results in (graph_results, wiki_results, wikigraph_results):
        for rank, result in enumerate(source_results, start=1):
            key = _result_identity(result)
            rrf_scores[key] += 1 / (60 + rank)
            current = candidates.get(key)
            if current is None or result.score > current.score:
                candidates[key] = result

    ranked_keys = sorted(
        candidates,
        key=lambda key: (
            rrf_scores[key],
            candidates[key].score,
            candidates[key].title.casefold(),
        ),
        reverse=True,
    )
    return [
        _with_score(candidates[key], rrf_scores[key]) for key in ranked_keys[:limit]
    ]


def _with_score(result: SearchResult, score: float) -> SearchResult:
    return SearchResult(
        title=result.title,
        path=result.path,
        score=score,
        snippet=result.snippet,
        section=result.section,
        chunk_index=result.chunk_index,
        retriever=result.retriever,
    )


def _graph_find_diagnostics(graph_status: dict[str, object]) -> list[str]:
    table_states = graph_status.get("table_states")
    if not isinstance(table_states, dict):
        return []
    messages: list[str] = []
    for table_name in ("entities", "relationships"):
        state = table_states.get(table_name)
        if state in {"dependency_missing", "present_unreadable"}:
            messages.append(
                "GraphRAG "
                f"{table_name} artifacts are {state}; run `kb status` for details."
            )
    return messages


def _result_identity(result: SearchResult) -> tuple[str, str, str, str]:
    path = str(result.path)
    if result.retriever == "wikigraph":
        namespace = "wikigraph"
    elif path.startswith("graph://"):
        namespace = "graph"
    else:
        namespace = "wiki"
    section_id = (
        str(result.chunk_index) if result.chunk_index is not None else result.section
    )
    return (
        namespace,
        path.casefold(),
        str(result.section).casefold(),
        str(section_id).casefold(),
    )
