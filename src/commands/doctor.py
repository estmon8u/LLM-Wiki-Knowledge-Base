from __future__ import annotations

import click

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
        for check in report.checks:
            marker = "OK" if check.ok else "FAIL"
            click.echo(f"[{marker}] {check.name}: {check.detail}")
        click.echo("")
        click.echo(f"{report.passed_count} passed, {report.failed_count} failed")
        if not report.ok:
            raise SystemExit(1)

    return command
