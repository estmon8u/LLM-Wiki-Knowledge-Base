from __future__ import annotations

from importlib import import_module
from typing import Optional

import click

from src.models.command_models import CommandContext, CommandSpec


# Flat top-level commands (primary workflow verbs)
FLAT_COMMAND_MODULES = {
    "add": "src.commands.add",
    "compile": "src.commands.compile",
    "doctor": "src.commands.doctor",
    "ingest": "src.commands.ingest",
    "init": "src.commands.init",
}

# Namespaced command groups: group_name -> {subcommand -> module_path}
GROUP_COMMAND_MODULES = {
    "check": {
        "lint": "src.commands.lint",
        "review": "src.commands.review",
    },
    "export": {
        "vault": "src.commands.export_vault",
    },
    "query": {
        "ask": "src.commands.query",
        "search": "src.commands.search",
    },
    "show": {
        "status": "src.commands.status",
        "diff": "src.commands.diff",
    },
}


def list_command_names() -> list[str]:
    names = list(FLAT_COMMAND_MODULES)
    names.extend(GROUP_COMMAND_MODULES)
    return sorted(names)


def get_click_command(name: str) -> Optional[click.BaseCommand]:
    module_path = FLAT_COMMAND_MODULES.get(name)
    if module_path is not None:
        module = import_module(module_path)
        return module.create_command()

    group_spec = GROUP_COMMAND_MODULES.get(name)
    if group_spec is not None:
        return _build_click_group(name, group_spec)

    return None


def _build_click_group(
    group_name: str, subcommand_modules: dict[str, str]
) -> click.Group:
    group = click.Group(name=group_name)
    for sub_name, module_path in sorted(subcommand_modules.items()):
        module = import_module(module_path)
        group.add_command(module.create_command(), sub_name)
    return group


def build_command_specs(context: CommandContext) -> tuple[CommandSpec, ...]:
    specs: list[CommandSpec] = []
    for name in sorted(FLAT_COMMAND_MODULES):
        module = import_module(FLAT_COMMAND_MODULES[name])
        spec = module.build_spec(context)
        if spec.availability is None or spec.availability(context):
            specs.append(spec)
    for group_name in sorted(GROUP_COMMAND_MODULES):
        for sub_name in sorted(GROUP_COMMAND_MODULES[group_name]):
            module = import_module(GROUP_COMMAND_MODULES[group_name][sub_name])
            spec = module.build_spec(context)
            spec = CommandSpec(
                name=f"{group_name} {sub_name}",
                summary=spec.summary,
                availability=spec.availability,
            )
            if spec.availability is None or spec.availability(context):
                specs.append(spec)
    return tuple(specs)
