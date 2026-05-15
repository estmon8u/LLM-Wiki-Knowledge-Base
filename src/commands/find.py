"""Click command implementation for the kb find command."""


from __future__ import annotations

import click

from src.commands.common import console, emit_json, make_table, require_initialized
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Search maintained wiki pages, including GraphRAG export pages when present."


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

    @click.command(name="find", help=SUMMARY, short_help="Search the maintained wiki.")
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
        query = " ".join(query_terms)
        search_service = command_context.services["search"]
        results = search_service.search(query, limit=limit)

        if as_json:
            emit_json(
                {
                    "retriever": "local-index",
                    "query": query,
                    "results": [_search_result_payload(result) for result in results],
                }
            )
            return

        if not results:
            console.print("No wiki pages matched that query.")
            return

        rows = [
            (result.title, result.path, f"{result.score:.2f}", result.snippet)
            for result in results
        ]
        table = make_table(
            columns=[
                ("Title", {"style": "bold"}),
                ("Path", {}),
                ("Score", {"justify": "right"}),
                ("Snippet", {"style": "dim"}),
            ],
            rows=rows,
            title="Wiki Search Results",
        )
        console.print(table)

    return command


def _search_result_payload(result: object) -> dict[str, object]:
    return {
        "retriever": "local-index",
        "title": result.title,
        "path": result.path,
        "score": result.score,
        "snippet": result.snippet,
        "section": result.section,
        "chunk_index": result.chunk_index,
    }
