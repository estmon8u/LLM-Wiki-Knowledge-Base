from __future__ import annotations

import click

from src.commands.common import echo_kv
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Show high-level project, corpus, and compile state."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="show status", summary=SUMMARY)


def create_command() -> click.Command:
    @click.command(name="status", help=SUMMARY, short_help="Show project status.")
    @click.pass_obj
    def command(command_context: CommandContext) -> None:
        project_service = command_context.services["project"]
        status_service = command_context.services["status"]
        snapshot = status_service.snapshot(initialized=project_service.is_initialized())

        echo_kv("project_root", command_context.project_root)
        echo_kv("initialized", str(snapshot.initialized).lower())
        echo_kv("source_count", snapshot.source_count)
        echo_kv("compiled_source_count", snapshot.compiled_source_count)
        echo_kv("concept_page_count", snapshot.concept_page_count)
        echo_kv("last_compile_at", snapshot.last_compile_at)

    return command
