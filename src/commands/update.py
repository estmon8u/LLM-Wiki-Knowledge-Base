from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from src.commands.common import (
    echo_bullet,
    echo_section,
    echo_status_line,
    progress_report,
    require_initialized,
)
from src.models.command_models import CommandContext, CommandSpec
from src.providers import ProviderError


SUMMARY = (
    "Bring the knowledge base current. Optionally add new sources first, "
    "then compile, generate concepts, and refresh indexes."
)


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(
        name="update",
        summary=SUMMARY,
        aliases=("compile", "build"),
    )


def create_command() -> click.Command:
    @click.command(
        name="update",
        help=SUMMARY,
        short_help="Update the knowledge base.",
    )
    @click.argument(
        "source_paths",
        nargs=-1,
        type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    )
    @click.option(
        "--force",
        is_flag=True,
        help="Rebuild every source page even if nothing changed.",
    )
    @click.option(
        "--resume",
        is_flag=True,
        help="Resume the most recent interrupted or failed compile run.",
    )
    @click.pass_obj
    def command(
        command_context: CommandContext,
        source_paths: tuple[Path, ...],
        force: bool,
        resume: bool,
    ) -> None:
        require_initialized(command_context)
        if force and resume:
            raise click.ClickException("--resume cannot be combined with --force.")

        # Preflight: fail before disk mutations if the provider is missing.
        provider_name = command_context.config.get("provider", {}).get("name")
        if not provider_name:
            raise click.ClickException(
                "Provider is not configured, so the KB cannot be updated yet.\n"
                "Next: add a provider section to kb.config.yaml and set the "
                "matching API key environment variable."
            )

        # If paths are provided, add them first.
        if source_paths:
            ingest_service = command_context.services["ingest"]
            for source_path in source_paths:
                try:
                    if source_path.is_dir():
                        candidate_paths = ingest_service.discover_source_paths(
                            source_path
                        )
                        with progress_report(
                            label="Adding",
                            length=len(candidate_paths),
                            item_label="source file",
                        ) as advance:
                            dir_result = ingest_service.ingest_directory(
                                source_path,
                                progress_callback=lambda _path: advance(),
                            )
                        click.echo(
                            f"Added {dir_result.created_count} source(s) from "
                            f"{source_path}"
                        )
                    else:
                        result = ingest_service.ingest_path(source_path)
                        if result.created:
                            click.echo(f"Added {source_path.name}")
                        else:
                            click.echo(f"Already present: {source_path.name}")
                except (FileNotFoundError, ValueError) as error:
                    raise click.ClickException(str(error)) from error
            click.echo("")

        # Compile
        compile_service = command_context.services["compile"]
        try:
            plan = compile_service.plan(force=force, resume=resume)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        try:
            with progress_report(
                label="Updating",
                length=plan.pending_count,
                item_label="source page",
            ) as advance:
                result = compile_service.compile(
                    force=force,
                    resume=resume,
                    progress_callback=lambda _source: advance(),
                )
        except (ProviderError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

        if result.resumed_from_run_id:
            echo_status_line(
                "resume", f"resumed failed compile run {result.resumed_from_run_id}"
            )
            click.echo("")

        echo_section("Update Summary")
        click.echo(f"Compiled {result.compiled_count} source page(s)")
        click.echo(f"Skipped {result.skipped_count} source page(s)")
        for path in result.compiled_paths:
            echo_bullet(f"updated {path}")

        # Always generate concepts during update
        click.echo("")
        echo_section("Concept Summary")
        concept_service = command_context.services["concepts"]
        concept_result = concept_service.generate()
        click.echo(f"Generated {len(concept_result.concept_paths)} concept page(s)")
        click.echo(
            f"Updated {len(concept_result.updated_source_paths)} source page backlink section(s)"
        )
        if concept_result.removed_paths:
            click.echo(
                f"Removed {len(concept_result.removed_paths)} stale concept page(s)"
            )
        for path in concept_result.concept_paths:
            echo_bullet(path)

        command_context.services["search"].refresh(force=True)

    return command
