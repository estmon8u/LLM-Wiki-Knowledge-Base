from __future__ import annotations

import click

from src.commands.common import console, emit_json, echo_section, make_table
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Validate project structure, provider config, API keys, and converters."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="doctor", summary=SUMMARY)


def create_command() -> click.Command:
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
        doctor_service = command_context.services["doctor"]
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

        _SEVERITY_STYLE = {
            "ok": "[green]OK[/green]",
            "warning": "[yellow]WARNING[/yellow]",
            "error": "[red]FAIL[/red]",
        }
        rows = []
        for check in report.checks:
            status = _SEVERITY_STYLE.get(check.severity, "[red]FAIL[/red]")
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
