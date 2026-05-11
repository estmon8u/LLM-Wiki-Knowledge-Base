from __future__ import annotations

import click

from src.commands.common import require_initialized
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "GraphRAG search entry point. Legacy FTS search lives under kb legacy find."
GRAPH_SEARCH_PENDING = (
    "GraphRAG search is the default target path, but graph search is not wired yet. "
    "The old SQLite FTS5 path is deprecated and only available as "
    "'kb legacy find ...' for comparison or exact lexical lookup."
)


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="find", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(name="find", help=SUMMARY, short_help="GraphRAG search placeholder.")
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
        _ = (limit, as_json)
        raise click.ClickException(GRAPH_SEARCH_PENDING)

    return command
