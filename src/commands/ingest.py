from __future__ import annotations

from pathlib import Path

import click

from src.commands.common import require_initialized
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Ingest and normalize a source file into the raw corpus."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="ingest", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(name="ingest", help=SUMMARY, short_help="Ingest a source file.")
    @click.argument(
        "source_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
    )
    @click.pass_obj
    def command(command_context: CommandContext, source_path: Path) -> None:
        require_initialized(command_context)
        ingest_service = command_context.services["ingest"]
        try:
            result = ingest_service.ingest_path(source_path)
        except (FileNotFoundError, ValueError) as error:
            raise click.ClickException(str(error)) from error

        click.echo(result.message)
        if result.source is not None:
            click.echo(f"- slug: {result.source.slug}")
            click.echo(f"- raw path: {result.source.raw_path}")
            if result.source.normalized_path is not None:
                click.echo(f"- normalized path: {result.source.normalized_path}")

    return command
