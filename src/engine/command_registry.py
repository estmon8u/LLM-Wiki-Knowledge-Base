from __future__ import annotations

from importlib import import_module
from typing import Optional

import click

from src.models.command_models import CommandContext, CommandSpec


COMMAND_MODULES = {
    "compile": "src.commands.compile",
    "export-vault": "src.commands.export_vault",
    "ingest": "src.commands.ingest",
    "init": "src.commands.init",
    "lint": "src.commands.lint",
    "query": "src.commands.query",
    "search": "src.commands.search",
    "status": "src.commands.status",
}


ALIASES = {
    "export_vault": "export-vault",
}


def resolve_command_name(name: str) -> str:
    return ALIASES.get(name, name)


def list_command_names() -> list[str]:
    return sorted(COMMAND_MODULES)


def get_click_command(name: str) -> Optional[click.Command]:
    resolved = resolve_command_name(name)
    module_path = COMMAND_MODULES.get(resolved)
    if module_path is None:
        return None
    module = import_module(module_path)
    return module.create_command()


def build_command_specs(context: CommandContext) -> tuple[CommandSpec, ...]:
    specs: list[CommandSpec] = []
    for name in list_command_names():
        module = import_module(COMMAND_MODULES[name])
        spec = module.build_spec(context)
        if spec.availability is None or spec.availability(context):
            specs.append(spec)
    return tuple(specs)
