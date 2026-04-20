from __future__ import annotations

from pathlib import Path

import click

from src.commands.common import (
    echo_bullet,
    echo_section,
    echo_status_line,
    progress_report,
    require_initialized,
)
from src.models.command_models import CommandContext, CommandSpec
from src.services.update_service import (
    UpdateOptions,
    UpdatePreflightError,
    UpdateService,
)


SUMMARY = (
    "Bring the knowledge base current. Optionally add new sources first, "
    "then compile, generate concepts, and refresh indexes."
)


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(
        name="update",
        summary=SUMMARY,
    )


def _get_update_service(command_context: CommandContext) -> UpdateService:
    return UpdateService(
        ingest_service=command_context.services["ingest"],
        compile_service=command_context.services["compile"],
        concept_service=command_context.services["concepts"],
        search_service=command_context.services["search"],
        config=command_context.config,
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
        service = _get_update_service(command_context)

        options = UpdateOptions(source_paths=source_paths, force=force, resume=resume)

        try:

            def _progress_factory(pending_count):
                return progress_report(
                    label="Compiling",
                    length=pending_count,
                    item_label="source",
                )

            result = service.run(
                options,
                compile_progress_factory=_progress_factory,
            )
        except (UpdatePreflightError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc

        # Render ingest phase
        for summary in result.ingest_summaries:
            if summary.is_dir:
                click.echo(
                    f"Added {summary.created_count} source(s) from {summary.path}"
                )
            elif summary.message:
                click.echo(summary.message)
            else:
                click.echo(f"Added {summary.path.name}")
        if result.ingest_summaries:
            click.echo("")

        # Render compile phase
        cr = result.compile_result
        if cr.resumed_from_run_id:
            echo_status_line(
                "resume", f"resumed failed compile run {cr.resumed_from_run_id}"
            )
            click.echo("")

        echo_section("Update Summary")
        click.echo(f"Compiled {cr.compiled_count} source page(s)")
        click.echo(f"Skipped {cr.skipped_count} source page(s)")
        for path in cr.compiled_paths:
            echo_bullet(f"updated {path}")

        # Render concept phase
        click.echo("")
        echo_section("Concept Summary")
        concept_result = result.concept_result
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

    return command
