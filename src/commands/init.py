"""Click command implementation for the kb init command.

This module belongs to `src.commands.init` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import click

from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Create the project folders, config, schema, and manifest files."


def build_spec(_: CommandContext = None) -> CommandSpec:
    """Builds the command registry specification for this module.

    Args:
        _: Value value used by the operation.

    Returns:
        CommandSpec produced by the operation.
    """
    return CommandSpec(name="init", summary=SUMMARY)


def create_command() -> click.Command:
    """Creates the Click command exposed by this module.

    Returns:
        click.Command produced by the operation.
    """

    @click.command(
        name="init", help=SUMMARY, short_help="Initialize the project scaffold."
    )
    @click.pass_obj
    def command(command_context: CommandContext) -> None:
        """Command.

        Args:
            command_context: Command context value used by the operation.
        """
        project_service = command_context.services["project"]
        config_service = command_context.services["config"]
        manifest_service = command_context.services["manifest"]
        graphrag_workspace_service = command_context.services.get("graphrag_workspace")

        created_items = project_service.ensure_structure()
        created_items.extend(config_service.ensure_files(repair_invalid=True))
        if manifest_service.ensure_manifest():
            created_items.append("raw/_manifest.json")
        if graphrag_workspace_service is not None and command_context.config.get(
            "graph"
        ):
            created_items.extend(graphrag_workspace_service.ensure_workspace())

        click.echo(f"Initialized project at {command_context.project_root}")
        if created_items:
            for item in created_items:
                click.echo(f"- created {item}")
        else:
            click.echo("- project already had the required scaffold")

    return command
