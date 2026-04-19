from __future__ import annotations

from pathlib import Path

import click

from src.commands.common import require_initialized
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Ingest and normalize a source file or directory into the raw corpus."
SHORT_HELP = "Ingest source files."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="ingest", summary=SUMMARY)


def create_command(
    *,
    name: str = "ingest",
    help_text: str = SUMMARY,
    short_help: str = SHORT_HELP,
) -> click.Command:
    @click.command(name=name, help=help_text, short_help=short_help)
    @click.argument(
        "source_path",
        type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    )
    @click.option(
        "--recursive",
        is_flag=True,
        help="Recursively ingest all supported source files under a directory.",
    )
    @click.pass_obj
    def command(
        command_context: CommandContext,
        source_path: Path,
        recursive: bool,
    ) -> None:
        require_initialized(command_context)
        ingest_service = command_context.services["ingest"]
        try:
            if source_path.is_dir():
                if not recursive:
                    raise ValueError(
                        f"Directory ingest requires --recursive: {source_path.resolve()}"
                    )
                directory_result = ingest_service.ingest_directory(source_path)
                _echo_directory_result(directory_result)
                return

            result = ingest_service.ingest_path(source_path)
        except (FileNotFoundError, ValueError) as error:
            raise click.ClickException(str(error)) from error

        _echo_single_result(result)

    return command


def _echo_single_result(result) -> None:
    click.echo(result.message)
    if result.source is not None:
        click.echo(f"- slug: {result.source.slug}")
        click.echo(f"- raw path: {result.source.raw_path}")
        if result.source.normalized_path is not None:
            click.echo(f"- normalized path: {result.source.normalized_path}")


def _echo_directory_result(result) -> None:
    click.echo(
        "Processed "
        f"{result.scanned_file_count} supported source file(s) under "
        f"{result.directory_path}"
    )
    click.echo(f"- created: {result.created_count}")
    click.echo(f"- duplicates skipped: {result.duplicate_count}")
    for item in result.created_results:
        if item.source is None:
            continue
        click.echo(f"- ingested: {item.source.slug} ({item.source.raw_path})")
    for item in result.duplicate_results:
        duplicate = item.duplicate_of or item.source
        if duplicate is None:
            continue
        click.echo(f"- duplicate: {duplicate.slug}")
