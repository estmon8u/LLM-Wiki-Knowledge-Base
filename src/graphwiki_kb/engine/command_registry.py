"""Command engine helpers for command registry.

This module belongs to `graphwiki_kb.engine.command_registry` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from importlib import import_module

import click

from graphwiki_kb.models.command_models import CommandContext, CommandSpec

# ---------------------------------------------------------------------------
# Primary top-level commands.
# GraphRAG-first behavior is folded into the main command surface.
# ---------------------------------------------------------------------------
FLAT_COMMAND_MODULES = {
    "add": "graphwiki_kb.commands.add",
    "ask": "graphwiki_kb.commands.ask",
    "config": "graphwiki_kb.commands.config_cmd",
    "doctor": "graphwiki_kb.commands.doctor",
    "export": "graphwiki_kb.commands.export_cmd",
    "find": "graphwiki_kb.commands.find",
    "init": "graphwiki_kb.commands.init",
    "legacy": "graphwiki_kb.commands.legacy",
    "lint": "graphwiki_kb.commands.lint",
    "review": "graphwiki_kb.commands.review",
    "sources": "graphwiki_kb.commands.sources",
    "status": "graphwiki_kb.commands.status",
    "update": "graphwiki_kb.commands.update",
}


def list_command_names() -> list[str]:
    """Return only canonical command names for help output."""
    return sorted(FLAT_COMMAND_MODULES)


def get_click_command(name: str) -> click.BaseCommand | None:
    """Resolve a canonical command name to a Click command."""
    module_path = FLAT_COMMAND_MODULES.get(name)
    if module_path is not None:
        module = import_module(module_path)
        return module.create_command()
    return None


def build_command_specs(context: CommandContext) -> tuple[CommandSpec, ...]:
    """Builds command specs.

    Args:
        context: Execution context shared across the operation.

    Returns:
        tuple[CommandSpec, ...] produced by the operation.
    """
    specs: list[CommandSpec] = []
    for name in sorted(FLAT_COMMAND_MODULES):
        module = import_module(FLAT_COMMAND_MODULES[name])
        spec = module.build_spec(context)
        if spec.availability is None or spec.availability(context):
            specs.append(spec)
    return tuple(specs)
