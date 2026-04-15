from __future__ import annotations

import click

from src.commands.common import require_initialized
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = (
    "Run semantic review checks for contradictions and terminology drift "
    "(heuristic; model-backed when a provider is configured)."
)


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="review", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(
        name="review",
        help=SUMMARY,
        short_help="Semantic review for contradictions and terminology.",
    )
    @click.option(
        "--adversarial",
        is_flag=True,
        help="Run extractor, skeptic, and arbiter review over candidate page pairs.",
    )
    @click.pass_obj
    def command(command_context: CommandContext, adversarial: bool) -> None:
        require_initialized(command_context)
        review_service = command_context.services["review"]
        report = review_service.review(adversarial=adversarial)

        click.echo(f"Review mode: {report.mode}")

        if not report.issues:
            click.echo("No review issues found.")
            return

        for issue in report.issues:
            pages = ", ".join(issue.pages)
            click.echo(
                f"[{issue.severity.upper()}] {issue.code}: {issue.message} ({pages})"
            )

        click.echo("")
        click.echo(f"Total review issues: {report.issue_count}")

    return command
