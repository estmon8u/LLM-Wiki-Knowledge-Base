from __future__ import annotations

import click

from src.commands.common import console, emit_json, make_table, require_initialized
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Search the compiled wiki for relevant pages and snippets."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="find", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(name="find", help=SUMMARY, short_help="Search the compiled wiki.")
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
        require_initialized(command_context)
        if not query_terms:
            raise click.ClickException("Provide at least one search term.")
        search_service = command_context.services["search"]
        query = " ".join(query_terms)
        results = search_service.search(query, limit=limit, include_concepts=True)
        if not results:
            if as_json:
                emit_json([])
            else:
                console.print("No wiki pages matched that query.")
            return

        if as_json:
            emit_json(
                [
                    {
                        "title": r.title,
                        "path": r.path,
                        "score": r.score,
                        "snippet": r.snippet,
                    }
                    for r in results
                ]
            )
            return

        rows = []
        for result in results:
            rows.append(
                (result.title, result.path, f"{result.score:.2f}", result.snippet)
            )

        table = make_table(
            columns=[
                ("Title", {"style": "bold"}),
                ("Path", {}),
                ("Score", {"justify": "right"}),
                ("Snippet", {"style": "dim"}),
            ],
            rows=rows,
            title="Search Results",
        )
        console.print(table)

    return command
