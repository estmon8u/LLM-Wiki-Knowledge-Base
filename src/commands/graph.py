from __future__ import annotations

import click

from src.commands.common import console, emit_json, require_initialized
from src.models.command_models import CommandContext, CommandSpec
from src.services.graphrag_input_sync_service import GraphRAGInputSyncError


SUMMARY = "GraphRAG workspace commands."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="graph", summary=SUMMARY)


def create_command() -> click.Command:
    @click.group(name="graph", help=SUMMARY, short_help="Run GraphRAG commands.")
    def graph_group() -> None:
        """GraphRAG workspace commands."""

    @graph_group.command(
        name="sync",
        help="Sync normalized source artifacts into GraphRAG JSON input.",
        short_help="Sync normalized sources.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def sync(command_context: CommandContext, as_json: bool) -> None:
        require_initialized(command_context)
        sync_service = command_context.services["graphrag_input_sync"]

        try:
            result = sync_service.sync()
        except GraphRAGInputSyncError as exc:
            raise click.ClickException(str(exc)) from exc

        output_path = result.output_path.relative_to(command_context.project_root)
        settings_path = result.settings_path.relative_to(command_context.project_root)

        if as_json:
            emit_json(
                {
                    "source_count": result.source_count,
                    "output_path": output_path.as_posix(),
                    "settings_path": settings_path.as_posix(),
                    "metadata_fields": list(result.metadata_fields),
                    "settings_updated": result.settings_updated,
                }
            )
            return

        console.print(
            f"Synced {result.source_count} normalized source(s) to "
            f"{output_path.as_posix()}"
        )
        console.print(f"Configured GraphRAG JSON input in {settings_path.as_posix()}")

    return graph_group
