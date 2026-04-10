from __future__ import annotations

import click

from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Create the project folders, config, schema, and manifest files."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="init", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(
        name="init", help=SUMMARY, short_help="Initialize the project scaffold."
    )
    @click.pass_obj
    def command(command_context: CommandContext) -> None:
        project_service = command_context.services["project"]
        config_service = command_context.services["config"]
        manifest_service = command_context.services["manifest"]

        created_items = project_service.ensure_structure()
        created_items.extend(config_service.ensure_files())
        if manifest_service.ensure_manifest():
            created_items.append("raw/_manifest.json")

        click.echo(f"Initialized project at {command_context.project_root}")
        if created_items:
            for item in created_items:
                click.echo(f"- created {item}")
        else:
            click.echo("- project already had the required scaffold")

    return command
