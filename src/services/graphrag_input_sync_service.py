"""Graphrag input sync service service behavior for the knowledge-base workflow.

This module belongs to `src.services.graphrag_input_sync_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""


from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import hashlib

import yaml

from src.models.source_models import RawSourceRecord
from src.services.graphrag_defaults import (
    DEFAULT_GRAPHRAG_EMBEDDING_MODEL,
    DEFAULT_GRAPHRAG_MODEL,
)
from src.services.manifest_service import ManifestService
from src.services.project_service import ProjectPaths, atomic_write_text


GRAPH_INPUT_METADATA_FIELDS = (
    "source_id",
    "slug",
    "source_hash",
    "raw_path",
    "normalized_path",
    "converter",
    "normalization_route",
    "ingested_at",
)


class GraphRAGInputSyncError(ValueError):
    """Raised when normalized sources cannot be synced into GraphRAG input."""


@dataclass(frozen=True)
class GraphRAGInputSyncResult:
    """Stores graph raginput sync result data.

    Attributes:
        See annotated class attributes for stored values.
    """

    source_count: int
    output_path: Path
    settings_path: Path
    metadata_fields: tuple[str, ...]
    settings_updated: bool


class GraphRAGInputSyncService:
    """Coordinates graph raginput sync operations.

    Attributes:
        See annotated class attributes for stored values.
    """

    def __init__(
        self,
        paths: ProjectPaths,
        manifest_service: ManifestService,
    ) -> None:
        self.paths = paths
        self.manifest_service = manifest_service

    @property
    def workspace_dir(self) -> Path:
        """Workspace dir.

        Returns:
            Path produced by the operation.
        """
        return self.paths.graph_dir / "graphrag"

    @property
    def input_dir(self) -> Path:
        """Input dir.

        Returns:
            Path produced by the operation.
        """
        return self.workspace_dir / "input"

    @property
    def input_file(self) -> Path:
        """Input file.

        Returns:
            Path produced by the operation.
        """
        return self.input_dir / "sources.json"

    @property
    def settings_file(self) -> Path:
        """Settings file.

        Returns:
            Path produced by the operation.
        """
        return self.workspace_dir / "settings.yaml"

    def sync(self) -> GraphRAGInputSyncResult:
        """Sync.

        Returns:
            GraphRAGInputSyncResult produced by the operation.
        """
        sources = self.manifest_service.list_sources()
        self._reject_duplicate_source_ids(sources)

        manifest_hash = self._manifest_hash()
        records = [
            {**self._record_for_source(source), "manifest_hash": manifest_hash}
            for source in sources
        ]
        settings_updated = self.configure_settings()
        payload = json.dumps(records, indent=2, sort_keys=True, default=str) + "\n"
        atomic_write_text(self.input_file, payload)

        return GraphRAGInputSyncResult(
            source_count=len(records),
            output_path=self.input_file,
            settings_path=self.settings_file,
            metadata_fields=GRAPH_INPUT_METADATA_FIELDS,
            settings_updated=settings_updated,
        )

    def configure_settings(self) -> bool:
        """Configure settings.

        Returns:
            bool produced by the operation.
        """
        if not self.settings_file.exists():
            relative = self._relative(self.settings_file)
            raise GraphRAGInputSyncError(
                f"GraphRAG settings not found at {relative}. "
                "Run `poetry run graphrag init --root graph/graphrag "
                f"--model {DEFAULT_GRAPHRAG_MODEL} "
                f"--embedding {DEFAULT_GRAPHRAG_EMBEDDING_MODEL} --force` "
                "or run `kb init` before `kb update`."
            )

        original = self.settings_file.read_text(encoding="utf-8")
        settings = yaml.safe_load(original) or {}
        if not isinstance(settings, dict):
            raise GraphRAGInputSyncError(
                f"GraphRAG settings must contain a YAML mapping: "
                f"{self._relative(self.settings_file)}"
            )

        input_config = dict(settings.get("input") or {})
        input_config.update(
            {
                "type": "json",
                "encoding": "utf-8",
                "file_pattern": ".*\\.json\\Z",
                "id_column": "id",
                "title_column": "title",
                "text_column": "text",
            }
        )
        settings["input"] = input_config

        input_storage = dict(settings.get("input_storage") or {})
        input_storage.setdefault("type", "file")
        input_storage["base_dir"] = "input"
        settings["input_storage"] = input_storage

        chunking = dict(settings.get("chunking") or {})
        chunking.setdefault("type", "tokens")
        chunking.setdefault("size", 100)
        chunking.setdefault("overlap", 25)
        chunking.setdefault("encoding_model", "o200k_base")
        chunking["prepend_metadata"] = list(GRAPH_INPUT_METADATA_FIELDS)
        settings["chunking"] = chunking

        updated = yaml.safe_dump(settings, sort_keys=False)
        if not updated.endswith("\n"):
            updated += "\n"
        if updated == original:
            return False

        atomic_write_text(self.settings_file, updated)
        return True

    def _record_for_source(self, source: RawSourceRecord) -> dict[str, Any]:
        if not source.normalized_path:
            raise GraphRAGInputSyncError(
                f"Source {source.source_id} has no normalized artifact path."
            )

        normalized_path = self._resolve_project_path(source.normalized_path)
        if not normalized_path.exists():
            raise GraphRAGInputSyncError(
                f"Normalized artifact missing for source {source.source_id}: "
                f"{source.normalized_path}"
            )

        metadata = source.metadata or {}
        converter = (
            metadata.get("converter")
            or metadata.get("fallback_converter")
            or metadata.get("normalization_route")
            or "unknown"
        )

        return {
            "id": source.source_id,
            "title": source.title,
            "text": normalized_path.read_text(encoding="utf-8"),
            "source_id": source.source_id,
            "slug": source.slug,
            "source_hash": source.content_hash,
            "raw_path": self._as_posix(source.raw_path),
            "normalized_path": self._as_posix(source.normalized_path),
            "converter": converter,
            "normalization_route": metadata.get("normalization_route"),
            "source_type": source.source_type,
            "origin": source.origin,
            "origin_hash": source.origin_hash,
            "ingested_at": source.ingested_at,
            "compiled_at": source.compiled_at,
            "compiled_from_hash": source.compiled_from_hash,
            "metadata": metadata,
        }

    def _reject_duplicate_source_ids(self, sources: list[RawSourceRecord]) -> None:
        seen: set[str] = set()
        for source in sources:
            if source.source_id in seen:
                raise GraphRAGInputSyncError(
                    f"Duplicate source_id in manifest: {source.source_id}"
                )
            seen.add(source.source_id)

    def _resolve_project_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.paths.root / path

    def _relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.paths.root).as_posix()
        except ValueError:
            return str(path)

    @staticmethod
    def _as_posix(value: str | None) -> str | None:
        if value is None:
            return None
        return Path(value).as_posix()

    def _manifest_hash(self) -> str | None:
        if not self.paths.raw_manifest_file.exists():
            return None
        digest = hashlib.sha256()
        with self.paths.raw_manifest_file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
