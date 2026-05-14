"""Click command implementation for the kb add command.

This module belongs to `src.commands.add` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""


from __future__ import annotations

import click

from src.commands.ingest import create_command as create_ingest_command
from src.models.command_models import CommandContext, CommandSpec


SUMMARY = "Add and normalize source files or folders."


def build_spec(_: CommandContext = None) -> CommandSpec:
    """Builds the command registry specification for this module.

    Args:
        _: Value value used by the operation.

    Returns:
        CommandSpec produced by the operation.
    """
    return CommandSpec(name="add", summary=SUMMARY)


def create_command() -> click.Command:
    """Creates the Click command exposed by this module.

    Returns:
        click.Command produced by the operation.
    """
    return create_ingest_command(
        name="add",
        help_text=SUMMARY,
        short_help="Add source files.",
    )
