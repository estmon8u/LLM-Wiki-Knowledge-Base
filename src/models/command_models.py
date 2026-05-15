"""Data models for command models.

This module belongs to `src.models.command_models` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence


@dataclass
class CommandContext:
    """Represents command context behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    project_root: Path
    cwd: Path
    config: dict[str, Any]
    schema_text: str
    services: dict[str, Any]
    verbose: bool = False


@dataclass
class CommandResult:
    """Stores command result data.

    Attributes:
        See annotated class attributes for stored values.
    """

    ok: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 0


CommandRunner = Callable[[Sequence[str], CommandContext], CommandResult]
AvailabilityChecker = Callable[[CommandContext], bool]


@dataclass
class CommandSpec:
    """Represents command spec behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    name: str
    summary: str
    availability: Optional[AvailabilityChecker] = None
    run: Optional[CommandRunner] = None
