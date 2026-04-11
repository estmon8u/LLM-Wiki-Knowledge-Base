from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    normalized = normalized.strip("-")
    return normalized or "untitled"


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
            self.paths.vault_dir,
            self.paths.vault_obsidian_dir,
            self.paths.graph_dir,
            self.paths.graph_exports_dir,
        ):
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                created.append(directory.relative_to(self.paths.root).as_posix() or ".")
        return created

    def to_relative_path(self, path: Path) -> str:
        return path.resolve().relative_to(self.paths.root).as_posix()
