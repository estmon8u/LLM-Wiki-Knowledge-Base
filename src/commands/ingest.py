from __future__ import annotations

from pathlib import Path

import click

from src.commands.common import (
    echo_bullet,
    echo_section,
    progress_report,
    require_initialized,
)
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
        help="Compatibility flag; directory inputs recurse automatically.",
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
                candidate_paths = ingest_service.discover_source_paths(source_path)
                with progress_report(
                    label="Ingesting",
                    length=len(candidate_paths),
                    item_label="source file",
                ) as advance:
                    directory_result = ingest_service.ingest_directory(
                        source_path,
                        progress_callback=lambda _path: advance(),
                    )
                _echo_directory_result(directory_result)
                return

            result = ingest_service.ingest_path(source_path)
        except (FileNotFoundError, ValueError) as error:
            raise click.ClickException(str(error)) from error

        _echo_single_result(result)

    return command


def _echo_single_result(result) -> None:
    echo_section("Ingest Summary")
    click.echo(result.message)
    if result.source is not None:
        echo_bullet(f"slug: {result.source.slug}")
        echo_bullet(f"raw path: {result.source.raw_path}")
        if result.source.normalized_path is not None:
            echo_bullet(f"normalized path: {result.source.normalized_path}")


def _echo_directory_result(result) -> None:
    echo_section("Ingest Summary")
    click.echo(
        "Processed "
        f"{result.scanned_file_count} supported source file(s) under "
        f"{result.directory_path}"
    )
    echo_bullet(f"created: {result.created_count}")
    echo_bullet(f"duplicates skipped: {result.duplicate_count}")
    for item in result.created_results:
        if item.source is None:
            continue
        echo_bullet(f"ingested: {item.source.slug} ({item.source.raw_path})")
    for item in result.duplicate_results:
        duplicate = item.duplicate_of or item.source
        if duplicate is None:
            continue
        echo_bullet(f"duplicate: {duplicate.slug}")
