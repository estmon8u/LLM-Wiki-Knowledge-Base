from __future__ import annotations

import click

from src.models.command_models import CommandContext, CommandSpec
from src.services.tui_service import TuiService


SUMMARY = "Open a persistent terminal workspace over the knowledge-base services."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="tui", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(name="tui", help=SUMMARY, short_help="Launch terminal workspace.")
    @click.option(
        "--command",
        "scripted_commands",
        multiple=True,
        help="Run one or more TUI commands non-interactively and exit.",
    )
    @click.pass_obj
    def command(
        command_context: CommandContext, scripted_commands: tuple[str, ...]
    ) -> None:
        tui_service = TuiService(command_context)
        if scripted_commands:
            summary = tui_service.run_scripted(scripted_commands)
            click.echo(summary.transcript)
            if summary.had_errors:
                raise click.exceptions.Exit(1)
            return

        try:
            tui_service.run_interactive()
        except RuntimeError as error:
            raise click.ClickException(str(error)) from error

    return command
