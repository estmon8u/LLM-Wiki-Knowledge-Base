from __future__ import annotations

import click

from src.commands.common import require_initialized
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Run deterministic structural lint checks over the maintained wiki."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="lint", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(name="lint", help=SUMMARY, short_help="Check wiki health.")
    @click.pass_obj
    def command(command_context: CommandContext) -> None:
        require_initialized(command_context)
        lint_service = command_context.services["lint"]
        report = lint_service.lint()

        if not report.issues:
            click.echo("No lint issues found.")
            return

        for severity in ("error", "warning", "suggestion"):
            scoped = [issue for issue in report.issues if issue.severity == severity]
            if not scoped:
                continue
            click.echo(f"{severity.upper()}S ({len(scoped)}):")
            for issue in scoped:
                click.echo(f"- {issue.code} [{issue.path}] {issue.message}")

        if report.error_count > 0:
            raise click.exceptions.Exit(1)

    return command
