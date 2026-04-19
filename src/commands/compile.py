from __future__ import annotations

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


SUMMARY = "Compile source pages and refresh the wiki index and activity log (requires a configured provider)."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="compile", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(
        name="compile", help=SUMMARY, short_help="Compile the maintained wiki."
    )
    @click.option(
        "--force",
        is_flag=True,
        help="Rebuild every source page even if nothing changed.",
    )
    @click.option(
        "--with-concepts",
        is_flag=True,
        help="Generate concept pages and maintain source-page backlinks after compiling.",
    )
    @click.option(
        "--resume",
        is_flag=True,
        help="Resume the most recent interrupted or failed compile run.",
    )
    @click.pass_obj
    def command(
        command_context: CommandContext,
        force: bool,
        with_concepts: bool,
        resume: bool,
    ) -> None:
        require_initialized(command_context)
        if force and resume:
            raise click.ClickException("--resume cannot be combined with --force.")
        compile_service = command_context.services["compile"]
        try:
            plan = compile_service.plan(force=force, resume=resume)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
        try:
            with progress_report(
                label="Compiling",
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
        echo_section("Compile Summary")
        click.echo(f"Compiled {result.compiled_count} source page(s)")
        click.echo(f"Skipped {result.skipped_count} source page(s)")
        for path in result.compiled_paths:
            echo_bullet(f"updated {path}")
        if with_concepts:
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
