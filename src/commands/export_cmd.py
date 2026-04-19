from __future__ import annotations

import click

from src.commands.common import echo_bullet, echo_section, require_initialized
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = (
    "Export the compiled wiki to the configured target (defaults to Obsidian vault)."
)


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="export", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(name="export", help=SUMMARY, short_help="Export the wiki.")
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
        echo_section("Vault Export")
        click.echo(
            f"Exported {len(result.exported_paths)} markdown file(s) to the vault"
        )
        for path in result.exported_paths:
            echo_bullet(path)
        if result.removed_paths:
            click.echo("")
            echo_section("Removed Stale Files")
            click.echo(
                f"Removed {len(result.removed_paths)} stale file(s) from the vault"
            )
            for path in result.removed_paths:
                echo_bullet(path)

    return command
