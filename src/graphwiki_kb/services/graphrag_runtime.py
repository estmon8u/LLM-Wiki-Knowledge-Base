"""GraphRAG runtime identity and compatibility helpers."""

from __future__ import annotations

import importlib.metadata
import inspect
import re
from typing import Any, Callable


MANAGED_GRAPHRAG_SETTINGS_VERSION = 1
GRAPH_INPUT_SCHEMA_VERSION = 1
SUPPORTED_GRAPHRAG_MIN = (3, 0, 9)
SUPPORTED_GRAPHRAG_MAX_EXCLUSIVE = (3, 1, 0)


class GraphRAGCompatibilityError(RuntimeError):
    """Raised when the installed GraphRAG package is not compatible."""


def graphrag_runtime_identity() -> dict[str, Any]:
    """Return stable fields that affect GraphRAG output reproducibility."""
    return {
        "graphrag_version": installed_graphrag_version(),
        "managed_settings_version": MANAGED_GRAPHRAG_SETTINGS_VERSION,
        "input_schema_version": GRAPH_INPUT_SCHEMA_VERSION,
    }


def installed_graphrag_version() -> str:
    try:
        return importlib.metadata.version("graphrag")
    except importlib.metadata.PackageNotFoundError:
        return "uninstalled"


def validate_graphrag_runtime() -> None:
    """Validate the installed GraphRAG package version and callable shapes."""
    version = installed_graphrag_version()
    if version == "uninstalled":
        raise GraphRAGCompatibilityError(
            "GraphRAG is not installed. Install the project dependencies with Poetry."
        )
    parsed_version = _parse_version(version)
    if parsed_version < SUPPORTED_GRAPHRAG_MIN:
        raise GraphRAGCompatibilityError(
            f"GraphRAG {version} is too old; expected >= 3.0.9 and < 3.1."
        )
    if parsed_version >= SUPPORTED_GRAPHRAG_MAX_EXCLUSIVE:
        raise GraphRAGCompatibilityError(
            f"GraphRAG {version} is outside the supported range >= 3.0.9 and < 3.1."
        )

    try:
        from graphrag.cli.index import index_cli, update_cli
        from graphrag.cli.initialize import initialize_project_at
        from graphrag.cli.query import (
            run_basic_search,
            run_drift_search,
            run_global_search,
            run_local_search,
        )
    except Exception as exc:  # noqa: BLE001
        raise GraphRAGCompatibilityError(
            f"GraphRAG CLI entrypoints are unavailable: {exc}"
        ) from exc

    _require_parameters(
        initialize_project_at,
        "graphrag.cli.initialize.initialize_project_at",
        {"root", "force", "model", "embedding_model"},
        aliases={"root": {"root", "root_dir", "path", "workspace_dir"}},
    )
    _require_parameters(
        index_cli,
        "graphrag.cli.index.index_cli",
        {"root_dir", "method", "dry_run", "cache", "skip_validation"},
        aliases={"root_dir": {"root_dir", "root", "workspace_dir"}},
    )
    _require_parameters(
        update_cli,
        "graphrag.cli.index.update_cli",
        {"root_dir", "method", "cache", "skip_validation"},
        aliases={"root_dir": {"root_dir", "root", "workspace_dir"}},
    )
    for func, name in (
        (run_local_search, "run_local_search"),
        (run_global_search, "run_global_search"),
        (run_drift_search, "run_drift_search"),
        (run_basic_search, "run_basic_search"),
    ):
        _require_parameters(
            func,
            f"graphrag.cli.query.{name}",
            {"root_dir", "data_dir", "query", "streaming", "verbose"},
            aliases={"root_dir": {"root_dir", "root", "workspace_dir"}},
        )


def _require_parameters(
    func: Callable[..., Any],
    name: str,
    required: set[str],
    *,
    aliases: dict[str, set[str]] | None = None,
) -> None:
    parameters = set(inspect.signature(func).parameters)
    missing: list[str] = []
    for parameter in required:
        alias_set = aliases.get(parameter, {parameter}) if aliases else {parameter}
        if parameters.isdisjoint(alias_set):
            missing.append(parameter)
    if missing:
        raise GraphRAGCompatibilityError(
            f"{name} is missing expected parameter(s): {', '.join(sorted(missing))}."
        )


def _parse_version(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        raise GraphRAGCompatibilityError(
            f"Unable to parse installed GraphRAG version {version!r}."
        )
    return tuple(int(part) for part in match.groups())
