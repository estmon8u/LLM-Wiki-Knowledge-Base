from __future__ import annotations

import click

from src.commands.common import console, require_initialized
from src.models.command_models import CommandContext, CommandSpec
from rich.markup import escape as _esc


SUMMARY = "Run deterministic structural lint checks over the maintained wiki."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="lint", summary=SUMMARY)


_SEVERITY_STYLE = {
    "error": "red",
    "warning": "yellow",
    "suggestion": "dim",
}


def create_command() -> click.Command:
    @click.command(name="lint", help=SUMMARY, short_help="Check wiki health.")
    @click.pass_obj
    def command(command_context: CommandContext) -> None:
        require_initialized(command_context)
        lint_service = command_context.services["lint"]
        report = lint_service.lint()

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
