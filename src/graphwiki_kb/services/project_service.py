"""Project path helpers and filesystem setup for a knowledge-base workspace."""

from __future__ import annotations

import os
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from slugify import slugify as library_slugify


def utc_now_iso() -> str:
    """Return the current UTC timestamp as second-precision ISO-8601 text."""
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    """Return a filesystem-safe slug, falling back to `untitled`."""
    normalized = library_slugify(value)
    return normalized or "untitled"


def unique_markdown_heading(existing_text: str, heading: str) -> str:
    """Return a heading line that does not duplicate an existing markdown heading."""
    headings = {
        line.strip()
        for line in existing_text.splitlines()
        if line.lstrip().startswith("#")
    }
    if heading not in headings:
        return heading
    suffix = 2
    while f"{heading} ({suffix})" in headings:
        suffix += 1
    return f"{heading} ({suffix})"


def _atomic_temp_path(path: Path) -> Path:
    return path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"


def _replace_with_retry(source: Path, destination: Path) -> None:
    last_error: OSError | None = None
    for _ in range(10):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.01)
    if last_error is not None:
        raise last_error
    os.replace(source, destination)


def atomic_write_text(path: Path, contents: str, *, encoding: str = "utf-8") -> None:
    """Write text through a temporary file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(path)
    try:
        temp_path.write_text(contents, encoding=encoding, newline="")
        _replace_with_retry(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_copy_file(source: Path, destination: Path) -> None:
    """Copy a file through a temporary file and atomic replace."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(destination)
    try:
        shutil.copyfile(source, temp_path)
        _replace_with_retry(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical paths for every managed project directory and state file."""

    root: Path
    config_file: Path
    schema_file: Path
    raw_dir: Path
    raw_sources_dir: Path
    raw_normalized_dir: Path
    raw_manifest_file: Path
    wiki_dir: Path
    wiki_sources_dir: Path
    wiki_concepts_dir: Path
    wiki_analysis_dir: Path
    wiki_index_file: Path
    wiki_index_markdown: Path
    wiki_log_file: Path
    vault_dir: Path
    vault_obsidian_dir: Path
    graph_dir: Path
    graph_exports_dir: Path


def discover_project_root(start: Path) -> Path:
    """Find the nearest parent that looks like a KB project root."""
    resolved = start.resolve()
    candidates = [resolved, *resolved.parents]
    for candidate in candidates:
        if (candidate / "kb.config.yaml").exists() or (
            candidate / "kb.schema.md"
        ).exists():
            return candidate
    return resolved


def build_project_paths(root: Path) -> ProjectPaths:
    """Build canonical project paths from a root directory."""
    resolved_root = root.resolve()
    raw_dir = resolved_root / "raw"
    wiki_dir = resolved_root / "wiki"
    vault_dir = resolved_root / "vault"
    graph_dir = resolved_root / "graph"
    return ProjectPaths(
        root=resolved_root,
        config_file=resolved_root / "kb.config.yaml",
        schema_file=resolved_root / "kb.schema.md",
        raw_dir=raw_dir,
        raw_sources_dir=raw_dir / "sources",
        raw_normalized_dir=raw_dir / "normalized",
        raw_manifest_file=raw_dir / "_manifest.json",
        wiki_dir=wiki_dir,
        wiki_sources_dir=wiki_dir / "sources",
        wiki_concepts_dir=wiki_dir / "concepts",
        wiki_analysis_dir=wiki_dir / "analysis",
        wiki_index_file=wiki_dir / "_index.json",
        wiki_index_markdown=wiki_dir / "index.md",
        wiki_log_file=wiki_dir / "log.md",
        vault_dir=vault_dir,
        vault_obsidian_dir=vault_dir / "obsidian",
        graph_dir=graph_dir,
        graph_exports_dir=graph_dir / "exports",
    )


class ProjectService:
    """Creates and inspects the managed project structure."""

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def is_initialized(self) -> bool:
        """Return whether required project config files exist."""
        return self.paths.config_file.exists() and self.paths.schema_file.exists()

    def ensure_structure(self) -> list[str]:
        """Create required directories/files and return created relative paths."""
        created: list[str] = []
        for directory in (
            self.paths.root,
            self.paths.raw_dir,
            self.paths.raw_sources_dir,
            self.paths.raw_normalized_dir,
            self.paths.wiki_dir,
            self.paths.wiki_sources_dir,
            self.paths.wiki_concepts_dir,
            self.paths.wiki_analysis_dir,
            self.paths.vault_dir,
            self.paths.vault_obsidian_dir,
            self.paths.graph_dir,
            self.paths.graph_exports_dir,
        ):
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                created.append(directory.relative_to(self.paths.root).as_posix() or ".")
        if not self.paths.wiki_log_file.exists():
            atomic_write_text(self.paths.wiki_log_file, "# Activity Log\n")
            created.append(
                self.paths.wiki_log_file.relative_to(self.paths.root).as_posix()
            )
        return created

    def to_relative_path(self, path: Path) -> str:
        """Return a project-relative path, or an absolute POSIX path if outside."""
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.paths.root).as_posix()
        except ValueError:
            return resolved.as_posix()
