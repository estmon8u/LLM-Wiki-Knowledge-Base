"""Click command implementation for the kb lint command.

This module belongs to `src.commands.lint` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import click

from src.commands.common import console, emit_json, require_initialized
from src.models.command_models import CommandContext, CommandSpec
from rich.markup import escape as _esc


SUMMARY = "Run deterministic structural lint checks over the maintained wiki."


def build_spec(_: CommandContext = None) -> CommandSpec:
    """Builds the command registry specification for this module.

    Args:
        _: Value value used by the operation.

    Returns:
        CommandSpec produced by the operation.
    """
    return CommandSpec(name="lint", summary=SUMMARY)


_SEVERITY_STYLE = {
    "error": "red",
    "warning": "yellow",
    "suggestion": "dim",
}


def create_command() -> click.Command:
    """Creates the Click command exposed by this module.

    Returns:
        click.Command produced by the operation.
    """

    @click.command(name="lint", help=SUMMARY, short_help="Check wiki health.")
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def command(command_context: CommandContext, as_json: bool) -> None:
        """Command.

        Args:
            command_context: Command context value used by the operation.
            as_json: As json value used by the operation.
        """
        require_initialized(command_context)
        lint_service = command_context.services["lint"]
        report = lint_service.lint()

        if as_json:
            emit_json(
                {
                    "ok": report.error_count == 0,
                    "error_count": report.error_count,
                    "warning_count": report.warning_count,
                    "suggestion_count": report.suggestion_count,
                    "issues": [
                        {
                            "severity": issue.severity,
                            "code": issue.code,
                            "path": issue.path,
                            "message": issue.message,
                        }
                        for issue in report.issues
                    ],
                }
            )
            if report.error_count > 0:
                raise click.exceptions.Exit(1)
            return

        if not report.issues:
            console.print("[green]No lint issues found.[/green]")
            return

        for severity in ("error", "warning", "suggestion"):
            scoped = [issue for issue in report.issues if issue.severity == severity]
            if not scoped:
                continue
            style = _SEVERITY_STYLE.get(severity)
            if style:
                console.print(
                    f"[{style}]{severity.upper()}S ({len(scoped)}):[/{style}]"
                )
            else:
                console.print(f"{severity.upper()}S ({len(scoped)}):")
            for issue in scoped:
                code = _esc(issue.code)
                path = _esc(issue.path)
                msg = _esc(issue.message)
                if style:
                    console.print(
                        f"  [{style}]\u2022[/{style}] {code} \\[{path}] {msg}"
                    )
                else:
                    console.print(f"  \u2022 {code} \\[{path}] {msg}")

        if report.error_count > 0:
            raise click.exceptions.Exit(1)

    return command
