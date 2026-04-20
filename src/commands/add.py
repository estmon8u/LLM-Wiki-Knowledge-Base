from __future__ import annotations

import click

from src.commands.ingest import create_command as create_ingest_command
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Add and normalize source files or folders."


def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="add", summary=SUMMARY, aliases=("ingest",))


def create_command() -> click.Command:
    return create_ingest_command(
        name="add",
        help_text=SUMMARY,
        short_help="Add source files.",
    )
