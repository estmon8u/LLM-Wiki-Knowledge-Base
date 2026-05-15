"""Click command implementation for the kb export command.

This module belongs to `src.commands.export_cmd` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import click

from src.commands.common import console, echo_bullet, echo_section, require_initialized
from src.models.command_models import CommandContext, CommandSpec
from src.services.graphrag_wiki_export_service import GraphRAGWikiExportError


SUMMARY = (
    "Export the compiled wiki to the configured target (defaults to Obsidian vault)."
)


def build_spec(_: CommandContext = None) -> CommandSpec:
    """Builds the command registry specification for this module.

    Args:
        _: Value value used by the operation.

    Returns:
        CommandSpec produced by the operation.
    """
    return CommandSpec(name="export", summary=SUMMARY)


def create_command() -> click.Command:
    """Creates the Click command exposed by this module.

    Returns:
        click.Command produced by the operation.
    """

    @click.command(name="export", help=SUMMARY, short_help="Export the wiki.")
    @click.option(
        "--clean",
        is_flag=True,
        help="Remove stale vault files that no longer exist in the wiki.",
    )
    @click.pass_obj
    def command(command_context: CommandContext, clean: bool) -> None:
        """Command.

        Args:
            command_context: Command context value used by the operation.
            clean: Clean value used by the operation.
        """
        require_initialized(command_context)
        export_service = command_context.services.export

        graph_result = None
        graph_status = command_context.services.graphrag_status.status()
        if graph_status.workspace_initialized and graph_status.output_complete:
            try:
                graph_result = (
                    command_context.services.graphrag_wiki_export.export_wiki()
                )
            except GraphRAGWikiExportError as exc:
                console.print(f"[yellow]Graph wiki export skipped: {exc}[/yellow]")

        result = export_service.export_vault(clean=clean)
        echo_section("Vault Export")
        console.print(
            f"Exported {len(result.exported_paths)} markdown file(s) to the vault"
        )
        for path in result.exported_paths:
            echo_bullet(path)
        if result.removed_paths:
            console.print("")
            echo_section("Removed Stale Files")
            console.print(
                f"Removed {len(result.removed_paths)} stale file(s) from the vault"
            )
            for path in result.removed_paths:
                echo_bullet(path)

        if graph_result is None:
            return
        console.print("")
        echo_section("Graph Wiki Export")
        console.print(
            f"Exported {graph_result.exported_count} GraphRAG wiki page(s) to wiki/graph"
        )

    return command
