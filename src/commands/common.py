from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Union

import click

from src.models.command_models import CommandContext


ProgressAdvance = Callable[..., None]


def require_initialized(command_context: CommandContext) -> None:
    project_service = command_context.services["project"]
    if not project_service.is_initialized():
        raise click.ClickException("Project not initialized. Run 'kb init' first.")


def echo_section(title: str) -> None:
    click.echo(title)
    click.echo("=" * len(title))


def echo_bullet(text: str) -> None:
    click.echo(f"- {text}")


def echo_status_line(label: str, text: str) -> None:
    click.echo(f"[{label}] {text}")


def echo_kv(label: str, value: Optional[Union[str, int]]) -> None:
    click.echo(f"{label}: {value if value is not None else 'n/a'}")


@contextmanager
def progress_report(
    *,
    label: str,
    length: int,
    item_label: str,
) -> Iterator[ProgressAdvance]:
    if length <= 0:
        yield lambda *_args, **_kwargs: None
        return

    stream = click.get_text_stream("stderr")
    if not stream.isatty():
        click.echo(f"{label} {length} {item_label}(s)...")
        yield lambda *_args, **_kwargs: None
        return

    with click.progressbar(length=length, label=label, file=stream) as bar:

        def advance(*_args, **_kwargs) -> None:
            bar.update(1)

        yield advance
