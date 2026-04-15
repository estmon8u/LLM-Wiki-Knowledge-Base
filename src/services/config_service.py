from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.services.project_service import ProjectPaths


DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
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
        "summary_paragraph_limit": 2,
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
        merged = deepcopy(DEFAULT_CONFIG)
        return _deep_merge(merged, loaded)

    def load_schema(self) -> str:
        if not self.paths.schema_file.exists():
            return DEFAULT_SCHEMA
        return self.paths.schema_file.read_text(encoding="utf-8")

    def ensure_files(self) -> list[str]:
        created: list[str] = []
        if not self.paths.config_file.exists():
            self.paths.config_file.write_text(
                yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False),
                encoding="utf-8",
            )
            created.append(self.paths.config_file.name)
        if not self.paths.schema_file.exists():
            self.paths.schema_file.write_text(DEFAULT_SCHEMA, encoding="utf-8")
            created.append(self.paths.schema_file.name)
        return created


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base
