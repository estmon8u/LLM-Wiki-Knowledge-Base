from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolContext:
    project_root: str
    config: dict[str, Any]
    session_state: dict[str, Any]
    services: dict[str, Any]
    messages: list[dict[str, Any]]
    cancel_requested: Callable[[], bool]


@dataclass
class ToolResult:
    ok: bool
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    touched_paths: list[str] = field(default_factory=list)


@dataclass
class ToolSpec:
    name: str
    summary: str
    access_level: str
    is_concurrency_safe: bool
    run: Callable[[dict[str, Any], ToolContext], ToolResult]
