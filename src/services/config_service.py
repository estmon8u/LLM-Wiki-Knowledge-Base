from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.services.project_service import ProjectPaths, atomic_write_text


CURRENT_CONFIG_VERSION = 3


DEFAULT_CONFIG: dict[str, Any] = {
    "version": CURRENT_CONFIG_VERSION,
    "project": {
        "name": "Capstone Knowledge Base",
        "description": "Markdown-first research knowledge base maintained through a CLI workflow.",
    },
    "storage": {
        "raw_dir": "raw/sources",
        "raw_normalized_dir": "raw/normalized",
        "wiki_sources_dir": "wiki/sources",
        "wiki_concepts_dir": "wiki/concepts",
        "vault_dir": "vault/obsidian",
    },
    "compile": {
        "excerpt_character_limit": 900,
    },
    "lint": {
        "required_frontmatter_fields": [
            "title",
            "summary",
            "source_id",
            "raw_path",
            "source_hash",
            "compiled_at",
        ],
    },
    "provider": {},
    "ecosystem": {
        "observability": {
            "backend": "none",
            "enabled": False,
        },
        "workflows": {
            "query_backend": "python",
            "review_backend": "python",
        },
        "providers": {
            "backend": "direct",
        },
        "retrieval": {
            "mode": "lexical",
        },
    },
}


DEFAULT_SCHEMA = """# kb.schema.md

This file defines the default compilation rules for the capstone knowledge base.

## Source Pages

- Create one source page for every ingested document.
- Preserve source traceability with raw-path and source-id metadata.
- Keep the summary concise and grounded in the ingested file.
- Prefer extracting the document's core thesis, methods, findings, and open questions.

## Concept Pages

- Concept pages synthesize across multiple source pages.
- Only create or update concept pages when more than one source supports the topic.
- Prefer explicit backlinks instead of inferred links with no supporting text.

## Query Behavior

- Search the compiled wiki first.
- Prefer source-backed answers with page citations.
- Surface gaps when no compiled page answers the question well.

## Lint Goals

- Treat broken links and missing citations as errors.
- Treat empty summaries and orphan pages as warnings.
- Treat possible improvements, such as weak cross-linking, as suggestions.
"""


class ConfigService:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def load(self) -> dict[str, Any]:
        if not self.paths.config_file.exists():
            return deepcopy(DEFAULT_CONFIG)
        with self.paths.config_file.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("kb.config.yaml must contain a YAML mapping.")
        migrated, changed = _apply_config_migrations(loaded)
        if changed:
            atomic_write_text(
                self.paths.config_file,
                yaml.safe_dump(migrated, sort_keys=False),
            )
        merged = deepcopy(DEFAULT_CONFIG)
        return _deep_merge(merged, migrated)

    def load_schema(self) -> str:
        if not self.paths.schema_file.exists():
            return DEFAULT_SCHEMA
        return self.paths.schema_file.read_text(encoding="utf-8")

    def save(self, config: dict[str, Any]) -> None:
        """Write *config* back to kb.config.yaml (atomic)."""
        atomic_write_text(
            self.paths.config_file,
            yaml.safe_dump(config, sort_keys=False),
        )

    def ensure_files(self) -> list[str]:
        created: list[str] = []
        if not self.paths.config_file.exists():
            atomic_write_text(
                self.paths.config_file,
                yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False),
            )
            created.append(self.paths.config_file.name)
        if not self.paths.schema_file.exists():
            atomic_write_text(self.paths.schema_file, DEFAULT_SCHEMA)
            created.append(self.paths.schema_file.name)
        return created


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _config_version(config: dict[str, Any]) -> int:
    version = config.get("version", 1)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("kb.config.yaml version must be an integer.")
    if version < 1:
        raise ValueError("kb.config.yaml version must be >= 1.")
    return version


def _apply_config_migrations(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    migrated = deepcopy(config)
    changed = False
    version = _config_version(migrated)
    if version > CURRENT_CONFIG_VERSION:
        raise ValueError(
            "Unsupported kb.config.yaml version: "
            f"{version}. This CLI supports up to version {CURRENT_CONFIG_VERSION}."
        )

    while version < CURRENT_CONFIG_VERSION:
        if version == 1:
            migrated = _migrate_v1_to_v2(migrated)
            changed = True
            version = _config_version(migrated)
            continue
        if version == 2:
            migrated = _migrate_v2_to_v3(migrated)
            changed = True
            version = _config_version(migrated)
            continue
        raise ValueError(f"Unsupported kb.config.yaml version: {version}")

    return migrated, changed


def _migrate_v1_to_v2(config: dict[str, Any]) -> dict[str, Any]:
    migrated = deepcopy(config)
    storage = migrated.setdefault("storage", {})
    if isinstance(storage, dict):
        storage.setdefault(
            "raw_normalized_dir",
            DEFAULT_CONFIG["storage"]["raw_normalized_dir"],
        )

    compile_config = migrated.setdefault("compile", {})
    if isinstance(compile_config, dict):
        compile_config.pop("summary_paragraph_limit", None)

    migrated.setdefault("provider", {})
    migrated["version"] = 2
    return migrated


def _migrate_v2_to_v3(config: dict[str, Any]) -> dict[str, Any]:
    migrated = deepcopy(config)
    migrated.setdefault("ecosystem", deepcopy(DEFAULT_CONFIG["ecosystem"]))
    migrated["version"] = 3
    return migrated
