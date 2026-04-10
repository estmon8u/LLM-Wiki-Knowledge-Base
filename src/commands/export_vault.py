from __future__ import annotations

import click

from src.commands.common import require_initialized
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Export the compiled wiki into the Obsidian-friendly vault folder."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="export-vault", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(name="export-vault", help=SUMMARY, short_help="Export vault files.")
    @click.pass_obj
    def command(command_context: CommandContext) -> None:
        require_initialized(command_context)
        export_service = command_context.services["export"]
        result = export_service.export_vault()
        click.echo(
            f"Exported {len(result.exported_paths)} markdown file(s) to the vault"
        )
        for path in result.exported_paths:
            click.echo(f"- {path}")

    return command
