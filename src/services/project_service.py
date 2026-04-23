from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import time
import uuid

from slugify import slugify as library_slugify


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    normalized = library_slugify(value)
    return normalized or "untitled"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(path)
    try:
        temp_path.write_text(contents, encoding=encoding)
        _replace_with_retry(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _atomic_temp_path(destination)
    try:
        shutil.copyfile(source, temp_path)
        _replace_with_retry(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class ProjectPaths:
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
    resolved = start.resolve()
    candidates = [resolved, *resolved.parents]
    for candidate in candidates:
        if (candidate / "kb.config.yaml").exists() or (
            candidate / "kb.schema.md"
        ).exists():
            return candidate
    return resolved


def build_project_paths(root: Path) -> ProjectPaths:
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
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def is_initialized(self) -> bool:
        return self.paths.config_file.exists() and self.paths.schema_file.exists()

    def ensure_structure(self) -> list[str]:
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
        return path.resolve().relative_to(self.paths.root).as_posix()
