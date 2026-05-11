from __future__ import annotations

from typing import Optional

import click

from src.commands.common import require_initialized
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = (
    "GraphRAG answer entry point. Legacy FTS-backed ask lives under kb legacy ask."
)
GRAPH_ASK_PENDING = (
    "GraphRAG answering is the default target path, but graph querying is not wired yet. "
    "The old SQLite FTS5-backed answer path is deprecated and only available as "
    "'kb legacy ask ...' for comparison."
)


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="ask", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(name="ask", help=SUMMARY, short_help="GraphRAG answer placeholder.")
    @click.argument("question_terms", nargs=-1)
    @click.option("--limit", default=3, type=int, help="Evidence page limit.")
    @click.option(
        "--save",
        "save_answer",
        is_flag=True,
        help="Save the answer as an analysis page in the wiki.",
    )
    @click.option(
        "--save-as",
        "save_as_name",
        type=str,
        default=None,
        help="Save the answer as an analysis page with a custom slug.",
    )
    @click.option(
        "--show-evidence",
        is_flag=True,
        help="Print the retrieved evidence snippets before the answer.",
    )
    @click.pass_obj
    def command(
        command_context: CommandContext,
        question_terms: tuple[str, ...],
        limit: int,
        save_answer: bool,
        save_as_name: Optional[str],
        show_evidence: bool,
    ) -> None:
        require_initialized(command_context)
        if not question_terms:
            raise click.ClickException("Provide a question to answer.")
        _ = (limit, save_answer, save_as_name, show_evidence)
        raise click.ClickException(GRAPH_ASK_PENDING)

    return command
