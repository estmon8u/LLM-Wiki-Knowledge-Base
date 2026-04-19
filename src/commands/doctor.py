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
    @click.pass_obj
    def command(command_context: CommandContext) -> None:
        doctor_service = command_context.services["doctor"]
        report = doctor_service.diagnose()
        echo_section("Health Checks")
        for check in report.checks:
            marker = "OK" if check.ok else "FAIL"
            echo_status_line(marker, f"{check.name}: {check.detail}")
        click.echo("")
        echo_section("Summary")
        click.echo(f"{report.passed_count} passed, {report.failed_count} failed")
        if not report.ok:
            raise SystemExit(1)

    return command
