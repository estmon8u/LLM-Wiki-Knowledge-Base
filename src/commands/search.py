from __future__ import annotations

import click

from src.commands.common import require_initialized
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Search the compiled wiki for relevant pages and snippets."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="query search", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(name="search", help=SUMMARY, short_help="Search the compiled wiki.")
    @click.argument("query_terms", nargs=-1)
    @click.option("--limit", default=5, show_default=True, type=int)
    @click.pass_obj
    def command(
        command_context: CommandContext, query_terms: tuple[str, ...], limit: int
    ) -> None:
        require_initialized(command_context)
        if not query_terms:
            raise click.ClickException("Provide at least one search term.")
        search_service = command_context.services["search"]
        query = " ".join(query_terms)
        results = search_service.search(query, limit=limit)
        if not results:
            click.echo("No wiki pages matched that query.")
            return

        for result in results:
            click.echo(f"- {result.title} [{result.path}] score={result.score}")
            click.echo(f"  {result.snippet}")

    return command
