from __future__ import annotations

import click

from src.commands.common import require_initialized
from src.models.command_models import CommandContext, CommandSpec
from src.providers import ProviderError


SUMMARY = (
    "Run semantic review checks for contradictions and terminology drift "
    "(requires a configured provider)."
)


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="check review", summary=SUMMARY)


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
        try:
            report = review_service.review(adversarial=adversarial)
        except ProviderError as exc:
            raise click.ClickException(str(exc)) from exc

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
