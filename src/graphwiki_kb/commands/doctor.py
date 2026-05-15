"""Click command implementation for the kb doctor command.

This module belongs to `graphwiki_kb.commands.doctor` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import click

from graphwiki_kb.commands.common import (
    CHECK_SEVERITY_LABEL,
    console,
    emit_json,
    echo_section,
    make_table,
)
from graphwiki_kb.models.command_models import CommandContext, CommandSpec


SUMMARY = "Validate project structure, provider config, API keys, and converters."


def build_spec(_: CommandContext = None) -> CommandSpec:
    """Builds the command registry specification for this module.

    Args:
        _: Value value used by the operation.

    Returns:
        CommandSpec produced by the operation.
    """
    return CommandSpec(name="doctor", summary=SUMMARY)


def create_command() -> click.Command:
    """Creates the Click command exposed by this module.

    Returns:
        click.Command produced by the operation.
    """

    @click.command(
        name="doctor",
        help=SUMMARY,
        short_help="Run project health checks.",
    )
    @click.option(
        "--strict",
        is_flag=True,
        help="Treat warnings (e.g. missing provider) as errors.",
    )
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def command(command_context: CommandContext, strict: bool, as_json: bool) -> None:
        """Command.

        Args:
            command_context: Command context value used by the operation.
            strict: Strict value used by the operation.
            as_json: As json value used by the operation.
        """
        doctor_service = command_context.services.doctor
        report = doctor_service.diagnose(strict=strict)

        if as_json:
            emit_json(
                {
                    "ok": report.ok,
                    "passed": report.passed_count,
                    "warnings": report.warning_count,
                    "failed": report.failed_count,
                    "checks": [
                        {
                            "name": c.name,
                            "severity": c.severity,
                            "detail": c.detail,
                        }
                        for c in report.checks
                    ],
                }
            )
            if not report.ok:
                raise SystemExit(1)
            return

        rows = []
        for check in report.checks:
            status = CHECK_SEVERITY_LABEL.get(check.severity, "[red]FAIL[/red]")
            rows.append((check.name, status, check.detail))

        table = make_table(
            columns=[
                ("Check", {"style": "bold"}),
                ("Status", {}),
                ("Detail", {}),
            ],
            rows=rows,
            title="Health Checks",
        )
        console.print(table)

        console.print("")
        parts = [f"[green]{report.passed_count} passed[/green]"]
        if report.warning_count:
            parts.append(f"[yellow]{report.warning_count} warnings[/yellow]")
        parts.append(
            f"{'[red]' if report.failed_count else ''}{report.failed_count} failed{'[/red]' if report.failed_count else ''}"
        )
        console.print(", ".join(parts))
        if not report.ok:
            raise SystemExit(1)

    return command
