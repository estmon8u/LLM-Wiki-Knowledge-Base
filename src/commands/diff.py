from __future__ import annotations

import click

from src.commands.common import (
    echo_kv,
    echo_section,
    echo_status_line,
    require_initialized,
)
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Show a pre-compile preview of new, changed, and up-to-date sources."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="show diff", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(name="diff", help=SUMMARY, short_help="Pre-compile source diff.")
    @click.pass_obj
    def command(command_context: CommandContext) -> None:
        require_initialized(command_context)
        diff_service = command_context.services["diff"]
        report = diff_service.diff()

        echo_section("Source Diff")
        for entry in report.entries:
            label = {"new": "NEW", "changed": "CHANGED", "up_to_date": "OK"}[
                entry.status
            ]
            suffix = f"  ({entry.details})" if entry.details else ""
            echo_status_line(label, f"{entry.title} — {entry.raw_path}{suffix}")

        click.echo("")
        echo_section("Summary")
        echo_kv("new", report.new_count)
        echo_kv("changed", report.changed_count)
        echo_kv("up_to_date", report.up_to_date_count)

    return command
