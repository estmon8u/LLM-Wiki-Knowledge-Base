from __future__ import annotations

import click

from src.commands.common import require_initialized
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Export the compiled wiki into the Obsidian-friendly vault folder."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="export vault", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(name="vault", help=SUMMARY, short_help="Export vault files.")
    @click.option(
        "--clean",
        is_flag=True,
        help="Remove stale vault files that no longer exist in the wiki.",
    )
    @click.pass_obj
    def command(command_context: CommandContext, clean: bool) -> None:
        require_initialized(command_context)
        export_service = command_context.services["export"]
        result = export_service.export_vault(clean=clean)
        click.echo(
            f"Exported {len(result.exported_paths)} markdown file(s) to the vault"
        )
        for path in result.exported_paths:
            click.echo(f"- {path}")
        if result.removed_paths:
            click.echo(
                f"\nRemoved {len(result.removed_paths)} stale file(s) from the vault"
            )
            for path in result.removed_paths:
                click.echo(f"- {path}")

    return command
