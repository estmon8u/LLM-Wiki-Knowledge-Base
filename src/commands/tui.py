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
    @click.option(
        "--snapshot",
        is_flag=True,
        help="Render a static preview of the current terminal workspace and exit.",
    )
    @click.option("--snapshot-width", default=120, show_default=True, type=int)
    @click.option("--snapshot-height", default=36, show_default=True, type=int)
    @click.pass_obj
    def command(
        command_context: CommandContext,
        scripted_commands: tuple[str, ...],
        snapshot: bool,
        snapshot_width: int,
        snapshot_height: int,
    ) -> None:
        tui_service = TuiService(command_context)
        if scripted_commands:
            summary = tui_service.run_scripted(scripted_commands)
            if snapshot:
                click.echo(
                    tui_service.render(
                        width=snapshot_width,
                        height=snapshot_height,
                    )
                )
            else:
                click.echo(summary.transcript)
            if summary.had_errors:
                raise click.exceptions.Exit(1)
            return

        if snapshot:
            click.echo(tui_service.render(width=snapshot_width, height=snapshot_height))
            return

        try:
            tui_service.run_interactive()
        except RuntimeError as error:
            raise click.ClickException(str(error)) from error

    return command
