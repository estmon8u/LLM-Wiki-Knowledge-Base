from __future__ import annotations

import click

from src.commands.common import console, require_initialized
from src.models.command_models import CommandContext, CommandSpec
from src.providers import ProviderError
from rich.markup import escape as _esc


SUMMARY = (
    "Run semantic review checks for contradictions and terminology drift "
    "(requires a configured provider)."
)

_SEVERITY_STYLE = {
    "error": "red",
    "warning": "yellow",
    "info": "dim",
}


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="review", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(
        name="review",
        help=SUMMARY,
        short_help="Semantic review for contradictions and terminology.",
    )
    @click.option(
        "--deep",
        "adversarial",
        is_flag=True,
        help="Run extractor, skeptic, and arbiter review over candidate page pairs.",
    )
    @click.option(
        "--adversarial",
        "adversarial",
        is_flag=True,
        hidden=True,
        help="Legacy alias for --deep.",
    )
    @click.pass_obj
    def command(command_context: CommandContext, adversarial: bool) -> None:
        require_initialized(command_context)
        review_service = command_context.services["review"]
        try:
            report = review_service.review(adversarial=adversarial)
        except ProviderError as exc:
            raise click.ClickException(str(exc)) from exc

        console.print(f"Review mode: [bold]{report.mode}[/bold]")

        if not report.issues:
            console.print("[green]No review issues found.[/green]")
            return

        for issue in report.issues:
            pages = ", ".join(issue.pages)
            sev_label = _esc(issue.severity.upper())
            body = f"{_esc(issue.code)}: {_esc(issue.message)} ({_esc(pages)})"
            style = _SEVERITY_STYLE.get(issue.severity)
            if style:
                console.print(f"[{style}]\\[{sev_label}][/{style}] {body}")
            else:
                console.print(f"\\[{sev_label}] {body}")

        console.print("")
        console.print(f"Total review issues: {report.issue_count}")

    return command
