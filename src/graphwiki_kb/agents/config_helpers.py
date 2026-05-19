"""Config helpers for agent modules."""

from __future__ import annotations

from typing import Any


def config_section(config: dict[str, object], name: str) -> dict[str, Any]:
    """Return a config subsection when present and mapping-shaped."""
    value = config.get(name)
    if isinstance(value, dict):
        return value
    return {}
