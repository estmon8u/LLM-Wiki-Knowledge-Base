from __future__ import annotations

from typing import Optional, Union

import click

from src.models.command_models import CommandContext


def require_initialized(command_context: CommandContext) -> None:
    project_service = command_context.services["project"]
    if not project_service.is_initialized():
        raise click.ClickException("Project not initialized. Run 'kb init' first.")


def echo_kv(label: str, value: Optional[Union[str, int]]) -> None:
    click.echo(f"{label}: {value if value is not None else 'n/a'}")
