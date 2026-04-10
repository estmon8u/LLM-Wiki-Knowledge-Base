from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence


@dataclass
class CommandContext:
    project_root: Path
    cwd: Path
    config: dict[str, Any]
    schema_text: str
    services: dict[str, Any]
    verbose: bool = False


@dataclass
class CommandResult:
    ok: bool
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 0


CommandRunner = Callable[[Sequence[str], CommandContext], CommandResult]
AvailabilityChecker = Callable[[CommandContext], bool]


@dataclass
class CommandSpec:
    name: str
    summary: str
    aliases: tuple[str, ...] = ()
    availability: Optional[AvailabilityChecker] = None
    run: Optional[CommandRunner] = None
