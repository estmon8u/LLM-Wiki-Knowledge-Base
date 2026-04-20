from __future__ import annotations

from importlib import import_module
from typing import Optional

import click

from src.models.command_models import CommandContext, CommandSpec


# ---------------------------------------------------------------------------
# Primary top-level commands (the simplified public CLI)
# ---------------------------------------------------------------------------
FLAT_COMMAND_MODULES = {
    "add": "src.commands.add",
    "ask": "src.commands.ask",
    "config": "src.commands.config_cmd",
    "doctor": "src.commands.doctor",
    "export": "src.commands.export_cmd",
    "find": "src.commands.find",
    "history": "src.commands.history",
    "init": "src.commands.init",
    "lint": "src.commands.lint",
    "review": "src.commands.review",
    "sources": "src.commands.sources",
    "status": "src.commands.status",
    "update": "src.commands.update",
}


def list_command_names() -> list[str]:
    """Return only canonical command names for help output."""
    return sorted(FLAT_COMMAND_MODULES)


def get_click_command(name: str) -> Optional[click.BaseCommand]:
    """Resolve a canonical command name to a Click command."""
    module_path = FLAT_COMMAND_MODULES.get(name)
    if module_path is not None:
        module = import_module(module_path)
        return module.create_command()
    return None


def build_command_specs(context: CommandContext) -> tuple[CommandSpec, ...]:
    specs: list[CommandSpec] = []
    for name in sorted(FLAT_COMMAND_MODULES):
        module = import_module(FLAT_COMMAND_MODULES[name])
        spec = module.build_spec(context)
        if spec.availability is None or spec.availability(context):
            specs.append(spec)
    return tuple(specs)
