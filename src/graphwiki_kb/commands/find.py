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

SUMMARY = "Search direct GraphRAG artifacts plus the maintained wiki index."


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
        graph_status = command_context.services.graphrag_status.status().to_dict(
            command_context.project_root
        )
        graph_diagnostics = _graph_find_diagnostics(graph_status)
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
                    "diagnostics": graph_diagnostics,
                    "results": [_search_result_payload(result) for result in results],
                }
            )
            return

        for diagnostic in graph_diagnostics:
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
    candidates: dict[tuple[str, str, str, str], SearchResult] = {}
    rrf_scores: defaultdict[tuple[str, str, str, str], float] = defaultdict(float)
    for source_results in (graph_results, wiki_results):
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
    return [candidates[key] for key in ranked_keys[:limit]]


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
    namespace = "graph" if str(result.path).startswith("graph://") else "wiki"
    section_id = (
        str(result.chunk_index) if result.chunk_index is not None else result.section
    )
    return (
        namespace,
        str(result.path).casefold(),
        str(result.section).casefold(),
        str(section_id).casefold(),
    )
