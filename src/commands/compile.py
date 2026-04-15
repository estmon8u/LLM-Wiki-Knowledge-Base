from __future__ import annotations

import click

from src.commands.common import require_initialized
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
    @click.pass_obj
    def command(
        command_context: CommandContext, force: bool, with_concepts: bool
    ) -> None:
        require_initialized(command_context)
        compile_service = command_context.services["compile"]
        try:
            result = compile_service.compile(force=force)
        except ProviderError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Compiled {result.compiled_count} source page(s)")
        click.echo(f"Skipped {result.skipped_count} source page(s)")
        for path in result.compiled_paths:
            click.echo(f"- updated {path}")
        if with_concepts:
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
                click.echo(f"- {path}")

    return command
