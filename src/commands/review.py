"""Click command implementation for the kb review command.

This module belongs to `src.commands.review` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""


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
    """Builds the command registry specification for this module.

    Args:
        _: Value value used by the operation.

    Returns:
        CommandSpec produced by the operation.
    """
    return CommandSpec(name="review", summary=SUMMARY)


def create_command() -> click.Command:
    """Creates the Click command exposed by this module.

    Returns:
        click.Command produced by the operation.
    """

    @click.command(
        name="review",
        help=SUMMARY,
        short_help="Semantic review for contradictions and terminology.",
    )
    @click.pass_obj
    def command(command_context: CommandContext) -> None:
        """Command.

        Args:
            command_context: Command context value used by the operation.
        """
        require_initialized(command_context)
        review_service = command_context.services["review"]
        try:
            report = review_service.review()
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
