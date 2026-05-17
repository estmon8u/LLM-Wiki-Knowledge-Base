"""Digest and freshness helpers for GraphRAG index reproducibility."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphwiki_kb.services.graphrag_runtime import graphrag_runtime_identity


@dataclass(frozen=True)
class GraphRAGFreshness:
    """Current graph inputs compared with the latest successful index run."""

    current_input_digest: str | None
    current_config_digest: str | None
    current_source_hashes: dict[str, str] | None
    input_changed: bool
    config_changed: bool
    changed_source_count: int | None
    missing_metadata: bool
    reasons: tuple[str, ...]

    @property
    def is_fresh(self) -> bool:
        return not self.reasons

    @property
    def state(self) -> str:
        if self.is_fresh:
            return "fresh"
        if self.missing_metadata:
            return "missing-metadata"
        return "stale"


def evaluate_graph_freshness(
    *,
    input_path: Path,
    workspace_dir: Path,
    last_successful_run: dict[str, Any] | None,
) -> GraphRAGFreshness:
    """Compare the current graph input/runtime with recorded index metadata."""
    current_input_digest = file_digest(input_path) if input_path.exists() else None
    current_config_digest = (
        graph_runtime_digest(workspace_dir) if workspace_dir.exists() else None
    )
    current_source_hashes = (
        graph_input_source_hashes(input_path) if input_path.exists() else None
    )

    if last_successful_run is None:
        return GraphRAGFreshness(
            current_input_digest=current_input_digest,
            current_config_digest=current_config_digest,
            current_source_hashes=current_source_hashes,
            input_changed=False,
            config_changed=False,
            changed_source_count=None,
            missing_metadata=True,
            reasons=("Graph index run metadata is missing.",),
        )

    reasons: list[str] = []
    missing_metadata = False
    last_input_digest = _optional_str(last_successful_run.get("input_digest"))
    last_config_digest = _optional_str(last_successful_run.get("config_digest"))
    last_source_hashes = source_hashes_from_run(last_successful_run)

    if last_input_digest is None:
        missing_metadata = True
        reasons.append("Graph index input digest metadata is missing.")
    if last_config_digest is None:
        missing_metadata = True
        reasons.append("Graph index runtime digest metadata is missing.")
    if last_source_hashes is None:
        missing_metadata = True
        reasons.append("Graph index source-hash metadata is missing.")

    input_changed = (
        last_input_digest is not None
        and current_input_digest is not None
        and last_input_digest != current_input_digest
    )
    config_changed = (
        last_config_digest is not None
        and current_config_digest is not None
        and last_config_digest != current_config_digest
    )
    changed_source_count = count_source_hash_changes(
        last_source_hashes,
        current_source_hashes or {},
    )
    if input_changed:
        reasons.append("Graph input digest changed since the last successful index.")
    if config_changed:
        reasons.append(
            "Graph runtime settings, prompts, GraphRAG version, or schema changed."
        )
    if changed_source_count:
        reasons.append(
            f"{changed_source_count} normalized source hash(es) changed since the "
            "last successful index."
        )

    return GraphRAGFreshness(
        current_input_digest=current_input_digest,
        current_config_digest=current_config_digest,
        current_source_hashes=current_source_hashes,
        input_changed=input_changed,
        config_changed=config_changed,
        changed_source_count=changed_source_count,
        missing_metadata=missing_metadata,
        reasons=tuple(reasons),
    )


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_runtime_digest(
    workspace_dir: Path,
    *,
    settings_text: str | None = None,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            graphrag_runtime_identity(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\0")
    if settings_text is None:
        _digest_file(digest, workspace_dir / "settings.yaml", "settings.yaml")
    else:
        digest.update(b"settings.yaml\0")
        digest.update(settings_text.encode("utf-8"))
        digest.update(b"\0")
    prompt_dir = workspace_dir / "prompts"
    if prompt_dir.exists():
        for path in sorted(prompt_dir.rglob("*.txt")):
            _digest_file(digest, path, path.relative_to(workspace_dir).as_posix())
    return digest.hexdigest()


def graph_input_source_hashes(input_path: Path) -> dict[str, str]:
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        sources = payload.get("sources") or payload.get("documents")
        records = sources if isinstance(sources, list) else []
    else:
        records = []

    source_hashes: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        source_id = _optional_str(record.get("source_id") or record.get("id"))
        source_hash = _optional_str(record.get("source_hash"))
        if source_id and source_hash:
            source_hashes[source_id] = source_hash
    return source_hashes


def count_source_hash_changes(
    previous: dict[str, str] | None,
    current: dict[str, str],
) -> int | None:
    if previous is None:
        return None
    keys = set(previous) | set(current)
    return sum(1 for key in keys if previous.get(key) != current.get(key))


def source_hashes_from_run(run: dict[str, Any]) -> dict[str, str] | None:
    payload = run.get("source_hashes")
    if not isinstance(payload, dict):
        return None
    source_hashes: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, str):
            source_hashes[key] = value
    return source_hashes


def _digest_file(digest: Any, path: Path, label: str) -> None:
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    if path.exists():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    digest.update(b"\0")


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
