"""Click command implementation for the kb review command.

This module belongs to `graphwiki_kb.commands.review` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import click
from rich.markup import escape as _esc

from graphwiki_kb.commands.common import (
    SEVERITY_STYLE,
    console,
    emit_json,
    require_initialized,
)
from graphwiki_kb.models.command_models import CommandContext, CommandSpec
from graphwiki_kb.providers import ProviderError

SUMMARY = (
    "Run semantic review checks for contradictions and terminology drift "
    "(requires a configured provider)."
)

_SEVERITY_RANK = {"suggestion": 0, "warning": 1, "error": 2}


def build_spec(_: CommandContext | None = None) -> CommandSpec:
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
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.option(
        "--fail-on",
        type=click.Choice(("error", "warning", "suggestion")),
        help="Exit non-zero when this severity or higher is present.",
    )
    @click.pass_obj
    def command(
        command_context: CommandContext,
        as_json: bool,
        fail_on: str | None,
    ) -> None:
        """Command.

        Args:
            command_context: Command context value used by the operation.
            as_json: As json value used by the operation.
            fail_on: Failure threshold severity.
        """
        require_initialized(command_context)
        review_service = command_context.services.review
        try:
            report = review_service.review()
        except ProviderError as exc:
            raise click.ClickException(str(exc)) from exc

        should_fail = _exceeds_fail_threshold(report.issues, fail_on)
        if as_json:
            emit_json(
                {
                    "ok": not should_fail,
                    "mode": report.mode,
                    "issue_count": report.issue_count,
                    "issues": [
                        {
                            "severity": issue.severity,
                            "code": issue.code,
                            "pages": issue.pages,
                            "message": issue.message,
                        }
                        for issue in report.issues
                    ],
                }
            )
            if should_fail:
                raise click.exceptions.Exit(1)
            return

        console.print(f"Review mode: [bold]{report.mode}[/bold]")

        if not report.issues:
            console.print("[green]No review issues found.[/green]")
            return

        for issue in report.issues:
            pages = ", ".join(issue.pages)
            sev_label = _esc(issue.severity.upper())
            body = f"{_esc(issue.code)}: {_esc(issue.message)} ({_esc(pages)})"
            style = SEVERITY_STYLE.get(issue.severity)
            if style:
                console.print(f"[{style}]\\[{sev_label}][/{style}] {body}")
            else:
                console.print(f"\\[{sev_label}] {body}")

        console.print("")
        console.print(f"Total review issues: {report.issue_count}")
        if should_fail:
            raise click.exceptions.Exit(1)

    return command


def _exceeds_fail_threshold(issues, fail_on: str | None) -> bool:
    if not fail_on:
        return False
    threshold = _SEVERITY_RANK[fail_on]
    return any(_SEVERITY_RANK.get(issue.severity, 0) >= threshold for issue in issues)
