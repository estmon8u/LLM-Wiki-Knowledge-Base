from __future__ import annotations

import click

from src.commands.common import echo_section, echo_status_line
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
    @click.pass_obj
    def command(command_context: CommandContext, strict: bool) -> None:
        doctor_service = command_context.services["doctor"]
        report = doctor_service.diagnose(strict=strict)
        echo_section("Health Checks")
        _SEVERITY_LABELS = {"ok": "OK", "warning": "WARNING", "error": "FAIL"}
        for check in report.checks:
            label = _SEVERITY_LABELS.get(check.severity, "FAIL")
            echo_status_line(label, f"{check.name}: {check.detail}")
        click.echo("")
        echo_section("Summary")
        parts = [f"{report.passed_count} passed"]
        if report.warning_count:
            parts.append(f"{report.warning_count} warnings")
        parts.append(f"{report.failed_count} failed")
        click.echo(", ".join(parts))
        if not report.ok:
            raise SystemExit(1)

    return command
