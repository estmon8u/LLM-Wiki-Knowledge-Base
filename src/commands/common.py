"""Shared presentation and validation helpers for CLI commands.

This module belongs to `src.commands.common` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import json as _json
import sys
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, Sequence, Union

import click
from rich.console import Console
from rich.markup import escape as _esc
from rich.progress import Progress
from rich.status import Status
from rich.table import Table

from src.models.command_models import CommandContext


def _configure_output_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (OSError, ValueError):
            continue


_configure_output_streams()

# Module-level consoles.  Rich auto-detects TTY and respects NO_COLOR.
console = Console()
err_console = Console(stderr=True)

ProgressAdvance = Callable[..., None]

SEVERITY_STYLE = {
    "error": "red",
    "warning": "yellow",
    "suggestion": "dim",
}

CHECK_SEVERITY_LABEL = {
    "ok": "[green]OK[/green]",
    "warning": "[yellow]WARNING[/yellow]",
    "error": "[red]FAIL[/red]",
}


def emit_json(data: Any) -> None:
    """Print *data* as indented JSON to stdout and return."""
    click.echo(_json.dumps(data, indent=2, default=str))


def require_initialized(command_context: CommandContext) -> None:
    """Require initialized.

    Args:
        command_context: Command context value used by the operation.
    """
    project_service = command_context.services["project"]
    if not project_service.is_initialized():
        raise click.ClickException("Project not initialized. Run 'kb init' first.")


def echo_section(title: str) -> None:
    """Echo section.

    Args:
        title: Title value used by the operation.
    """
    console.print(f"\n[bold]{title}[/bold]")
    console.print("=" * len(title))


def echo_bullet(text: str) -> None:
    """Echo bullet.

    Args:
        text: Text content being processed.
    """
    console.print(f"  \u2022 {_esc(text)}")


def echo_status_line(label: str, text: str) -> None:
    """Echo status line.

    Args:
        label: Label value used by the operation.
        text: Text content being processed.
    """
    console.print(f"[bold]\\[{_esc(label)}][/bold] {_esc(text)}")


def echo_kv(label: str, value: Optional[Union[str, int]]) -> None:
    """Echo kv.

    Args:
        label: Label value used by the operation.
        value: Input value being normalized, validated, or serialized.
    """
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
    """Progress report.

    Args:
        label: Label value used by the operation.
        length: Length value used by the operation.
        item_label: Item label value used by the operation.

    Returns:
        Iterator[ProgressAdvance] produced by the operation.
    """
    if length <= 0:
        yield lambda *_args, **_kwargs: None
        return

    if not err_console.is_terminal:
        console.print(f"{label} {length} {item_label}(s)...")
        yield lambda *_args, **_kwargs: None
        return

    with Progress(console=err_console, transient=True) as progress:
        task = progress.add_task(label, total=length)

        def advance(*_args, **_kwargs) -> None:
            """Advance.

            Args:
                _args: Args value used by the operation.
                _kwargs: Kwargs value used by the operation.
            """
            progress.advance(task)

        yield advance


@contextmanager
def live_status(
    label: str,
    *,
    spinner: str = "dots",
) -> Iterator[Callable[[str], None]]:
    """Show a Rich spinner with a live status message for indeterminate operations."""
    with lazy_live_status(label, spinner=spinner) as update:
        update("")
        yield update


@contextmanager
def lazy_live_status(
    label: str,
    *,
    spinner: str = "dots",
) -> Iterator[Callable[[str], None]]:
    """Start a Rich status spinner on the first update callback."""
    if not err_console.is_terminal:
        yield lambda _msg: None
        return

    status: Status | None = None

    def update(message: str) -> None:
        """Update.

        Args:
            message: Message value used by the operation.
        """
        nonlocal status
        cleaned = message.strip()
        display = f"{label} - {cleaned}" if cleaned else label
        if status is None:
            status = Status(display, console=err_console, spinner=spinner)
            status.start()
            return
        status.update(display)

    try:
        yield update
    finally:
        if status is not None:
            status.stop()
