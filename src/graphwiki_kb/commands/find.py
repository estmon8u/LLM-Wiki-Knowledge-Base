"""Click command implementation for the kb find command."""

from __future__ import annotations

import click

from graphwiki_kb.commands.common import (
    console,
    emit_json,
    make_table,
    require_initialized,
)
from graphwiki_kb.models.command_models import CommandContext, CommandSpec
from graphwiki_kb.models.wiki_models import SearchResult

SUMMARY = "Search direct GraphRAG artifacts plus the maintained wiki index."


def build_spec(_: CommandContext = None) -> CommandSpec:
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
    @click.option("--limit", default=5, show_default=True, type=int)
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def command(
        command_context: CommandContext,
        query_terms: tuple[str, ...],
        limit: int,
        as_json: bool,
    ) -> None:
        """Command.

        Args:
            command_context: Command context value used by the operation.
            query_terms: Query terms value used by the operation.
            limit: Maximum number of results to return or process.
            as_json: As json value used by the operation.
        """
        require_initialized(command_context)
        if not query_terms:
            raise click.ClickException("Provide at least one search term.")
        search_service = command_context.services.search
        graph_find_service = command_context.services.graphrag_find
        query = " ".join(query_terms).strip()
        candidate_limit = max(limit * 4, 20)
        graph_results = graph_find_service.search(query, limit=candidate_limit)
        wiki_results = search_service.search(
            query,
            limit=candidate_limit,
            include_concepts=True,
        )
        results = _merge_results(graph_results, wiki_results, limit=limit)

        if as_json:
            emit_json(
                {
                    "retriever": "graph-and-wiki-index",
                    "query": query,
                    "results": [_search_result_payload(result) for result in results],
                }
            )
            return

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


def _search_result_payload(result: SearchResult) -> dict[str, object]:
    retriever = (
        "graphrag-artifacts"
        if str(result.path).startswith("graph://")
        else "wiki-index"
    )
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
    *,
    limit: int,
) -> list[SearchResult]:
    candidates: list[SearchResult] = []
    seen: set[tuple[str, str]] = set()
    for result in [*graph_results, *wiki_results]:
        key = (str(result.title).casefold(), str(result.section).casefold())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(result)

    graph_max = _max_score(graph_results)
    wiki_max = _max_score(wiki_results)
    ranked = sorted(
        candidates,
        key=lambda result: (
            _normalized_score(result, graph_max=graph_max, wiki_max=wiki_max),
            result.score,
            result.title.casefold(),
        ),
        reverse=True,
    )
    return ranked[:limit]


def _max_score(results: list[SearchResult]) -> float:
    return max((float(result.score) for result in results), default=1.0) or 1.0


def _normalized_score(
    result: SearchResult,
    *,
    graph_max: float,
    wiki_max: float,
) -> float:
    max_score = graph_max if str(result.path).startswith("graph://") else wiki_max
    rank_score = float(result.score) / max(max_score, 1.0)
    absolute_score = min(float(result.score) / 10.0, 1.0)
    return (rank_score * 0.75) + (absolute_score * 0.25)
