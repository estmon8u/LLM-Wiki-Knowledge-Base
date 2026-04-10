from __future__ import annotations

import click

from src.commands.common import require_initialized
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Compile source pages and refresh the wiki index and activity log."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="compile", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(
        name="compile", help=SUMMARY, short_help="Compile the maintained wiki."
    )
    @click.option(
        "--force",
        is_flag=True,
        help="Rebuild every source page even if nothing changed.",
    )
    @click.pass_obj
    def command(command_context: CommandContext, force: bool) -> None:
        require_initialized(command_context)
        compile_service = command_context.services["compile"]
        result = compile_service.compile(force=force)
        click.echo(f"Compiled {result.compiled_count} source page(s)")
        click.echo(f"Skipped {result.skipped_count} source page(s)")
        for path in result.compiled_paths:
            click.echo(f"- updated {path}")

    return command
