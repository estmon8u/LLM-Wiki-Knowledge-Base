from __future__ import annotations

import json as _json
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, Sequence, Union

import click
from rich.console import Console
from rich.markup import escape as _esc
from rich.progress import Progress
from rich.table import Table

from src.models.command_models import CommandContext


# Module-level consoles.  Rich auto-detects TTY and respects NO_COLOR.
console = Console()
err_console = Console(stderr=True)

ProgressAdvance = Callable[..., None]


def emit_json(data: Any) -> None:
    """Print *data* as indented JSON to stdout and return."""
    click.echo(_json.dumps(data, indent=2, default=str))


def require_initialized(command_context: CommandContext) -> None:
    project_service = command_context.services["project"]
    if not project_service.is_initialized():
        raise click.ClickException("Project not initialized. Run 'kb init' first.")


def echo_section(title: str) -> None:
    console.print(f"\n[bold]{title}[/bold]")
    console.print("=" * len(title))


def echo_bullet(text: str) -> None:
    console.print(f"  \u2022 {_esc(text)}")


def echo_status_line(label: str, text: str) -> None:
    console.print(f"[bold]\\[{_esc(label)}][/bold] {_esc(text)}")


def echo_kv(label: str, value: Optional[Union[str, int]]) -> None:
    display = value if value is not None else "n/a"
    console.print(f"[dim]{_esc(str(label))}:[/dim] {_esc(str(display))}")


def make_table(
    columns: Sequence[tuple[str, dict]],
    rows: Sequence[Sequence[str]],
    *,
    title: str | None = None,
) -> Table:
    """Build a Rich Table from column specs and row data."""
    table = Table(title=title, show_lines=False)
    for col_name, col_kwargs in columns:
        table.add_column(col_name, **col_kwargs)
    for row in rows:
        table.add_row(*row)
    return table


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

    if not err_console.is_terminal:
        err_console.print(f"{label} {length} {item_label}(s)...")
        yield lambda *_args, **_kwargs: None
        return

    with Progress(console=err_console, transient=True) as progress:
        task = progress.add_task(label, total=length)

        def advance(*_args, **_kwargs) -> None:
            progress.advance(task)

        yield advance
