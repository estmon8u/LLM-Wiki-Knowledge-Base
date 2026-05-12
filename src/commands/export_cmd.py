from __future__ import annotations

import click

from src.commands.common import console, echo_bullet, echo_section, require_initialized
from src.models.command_models import CommandContext, CommandSpec
from src.services.graphrag_wiki_export_service import GraphRAGWikiExportError


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

        graph_status_service = command_context.services.get("graphrag_status")
        graph_export_service = command_context.services.get("graphrag_wiki_export")
        if graph_status_service is None or graph_export_service is None:
            return
        graph_status = graph_status_service.status()
        if not graph_status.workspace_initialized or not graph_status.output_present:
            return
        try:
            graph_result = graph_export_service.export_wiki()
        except GraphRAGWikiExportError as exc:
            console.print(f"[yellow]Graph wiki export skipped: {exc}[/yellow]")
            return
        console.print("")
        echo_section("Graph Wiki Export")
        console.print(
            f"Exported {graph_result.exported_count} GraphRAG wiki page(s) to wiki/graph"
        )

    return command
