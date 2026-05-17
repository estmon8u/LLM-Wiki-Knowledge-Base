"""Graphrag input sync service service behavior for the knowledge-base workflow.

This module belongs to `graphwiki_kb.services.graphrag_input_sync_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.services.config_service import resolve_graph_config
from graphwiki_kb.services.graphrag_defaults import (
    DEFAULT_GRAPHRAG_CHUNK_OVERLAP,
    DEFAULT_GRAPHRAG_CHUNK_SIZE,
    DEFAULT_GRAPHRAG_EMBEDDING_MODEL,
    DEFAULT_GRAPHRAG_ENCODING_MODEL,
    DEFAULT_GRAPHRAG_MAX_SOURCE_BYTES,
    DEFAULT_GRAPHRAG_MODEL,
)
from graphwiki_kb.services.manifest_service import ManifestError, ManifestService
from graphwiki_kb.services.project_service import ProjectPaths, atomic_write_text

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
GRAPH_INPUT_SIZE_WARNING_BYTES = 100 * 1024 * 1024


class GraphRAGInputSyncError(ValueError):
    """Raised when normalized sources cannot be synced into GraphRAG input."""

    def __init__(self, message: str, *, skippable: bool = False) -> None:
        super().__init__(message)
        self.skippable = skippable


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
    input_digest: str | None = None
    source_hashes: dict[str, str] = field(default_factory=dict)
    skipped_sources: tuple[str, ...] = ()
    input_size_bytes: int = 0
    warnings: tuple[str, ...] = ()


class GraphRAGInputSyncService:
    """Coordinates graph raginput sync operations.

    Attributes:
        See annotated class attributes for stored values.
    """

    def __init__(
        self,
        paths: ProjectPaths,
        manifest_service: ManifestService,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.paths = paths
        self.manifest_service = manifest_service
        self.config = config or {}

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

    def sync(
        self,
        *,
        preview_only: bool = False,
        allow_missing_sources: bool = False,
    ) -> GraphRAGInputSyncResult:
        """Sync.

        Args:
            preview_only: Build the planned input/settings payload without writing.
            allow_missing_sources: Skip isolated sources whose normalized artifact
                is missing instead of blocking every remaining source.

        Returns:
            GraphRAGInputSyncResult produced by the operation.
        """
        records, skipped_sources = self._planned_records(
            allow_missing_sources=allow_missing_sources
        )
        settings_updated = self._settings_would_update()
        payload = _records_payload(records)
        input_size_bytes = len(payload.encode("utf-8"))
        warnings = _input_size_warnings(input_size_bytes)
        if not preview_only:
            settings_updated = self.configure_settings()
            atomic_write_text(self.input_file, payload)

        return GraphRAGInputSyncResult(
            source_count=len(records),
            output_path=self.input_file,
            settings_path=self.settings_file,
            metadata_fields=GRAPH_INPUT_METADATA_FIELDS,
            settings_updated=settings_updated,
            input_digest=_text_digest(payload),
            source_hashes=_source_hashes(records),
            skipped_sources=tuple(skipped_sources),
            input_size_bytes=input_size_bytes,
            warnings=tuple(warnings),
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

        original, updated = self._configured_settings_payload()
        if updated == original:
            return False

        atomic_write_text(self.settings_file, updated)
        return True

    def _configured_settings_payload(self) -> tuple[str, str]:
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
        chunking.setdefault("size", DEFAULT_GRAPHRAG_CHUNK_SIZE)
        chunking.setdefault("overlap", DEFAULT_GRAPHRAG_CHUNK_OVERLAP)
        chunking.setdefault("encoding_model", DEFAULT_GRAPHRAG_ENCODING_MODEL)
        chunking["prepend_metadata"] = list(GRAPH_INPUT_METADATA_FIELDS)
        settings["chunking"] = chunking

        updated = yaml.safe_dump(settings, sort_keys=False)
        if not updated.endswith("\n"):
            updated += "\n"
        return original, updated

    def _settings_would_update(self) -> bool:
        original, updated = self._configured_settings_payload()
        return updated != original

    def _planned_records(
        self,
        *,
        allow_missing_sources: bool = False,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        try:
            sources = self.manifest_service.list_sources()
        except ManifestError as exc:
            message = str(exc).replace(
                "Duplicate manifest source_id",
                "Duplicate source_id",
            )
            raise GraphRAGInputSyncError(message) from exc
        self._reject_duplicate_source_ids(sources)

        manifest_hash = self._manifest_hash()
        records: list[dict[str, Any]] = []
        skipped_sources: list[str] = []
        for source in sources:
            try:
                record = self._record_for_source(source)
            except GraphRAGInputSyncError as exc:
                if not allow_missing_sources or not exc.skippable:
                    raise
                skipped_sources.append(f"{source.source_id}: {exc}")
                continue
            records.append({**record, "manifest_hash": manifest_hash})
        return records, skipped_sources

    def _record_for_source(self, source: RawSourceRecord) -> dict[str, Any]:
        if not source.normalized_path:
            raise GraphRAGInputSyncError(
                f"Source {source.source_id} has no normalized artifact path.",
                skippable=True,
            )

        normalized_path = self._resolve_normalized_path(source.normalized_path)
        if not normalized_path.exists():
            raise GraphRAGInputSyncError(
                f"Normalized artifact missing for source {source.source_id}: "
                f"{source.normalized_path}",
                skippable=True,
            )
        self._require_source_size(normalized_path, source_id=source.source_id)

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

    def _resolve_normalized_path(self, value: str) -> Path:
        path = Path(value)
        candidate = path if path.is_absolute() else self.paths.root / path
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            raise GraphRAGInputSyncError(
                f"Unable to resolve normalized source path: {value}"
            ) from exc

        allowed_roots = (
            self.paths.raw_dir.resolve(),
            self.paths.raw_normalized_dir.resolve(),
        )
        if not any(
            resolved == root or root in resolved.parents for root in allowed_roots
        ):
            allowed = ", ".join(
                root.relative_to(self.paths.root).as_posix()
                for root in allowed_roots
                if root == self.paths.root or self.paths.root in root.parents
            )
            raise GraphRAGInputSyncError(
                "Refusing to read normalized source outside project raw "
                f"directories ({allowed}): {resolved}"
            )
        return resolved

    def _require_source_size(self, path: Path, *, source_id: str) -> None:
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise GraphRAGInputSyncError(
                f"Unable to inspect normalized source size for {source_id}: {path}"
            ) from exc
        max_source_bytes = self._max_source_bytes()
        if size <= max_source_bytes:
            return
        size_mib = size / (1024 * 1024)
        limit_mib = max_source_bytes / (1024 * 1024)
        raise GraphRAGInputSyncError(
            "Source is too large for GraphRAG sync: "
            f"{source_id} ({size_mib:.1f} MiB > {limit_mib:.1f} MiB). "
            "Increase graph.input.max_source_bytes or split the source."
        )

    def _max_source_bytes(self) -> int:
        if not self.config:
            return DEFAULT_GRAPHRAG_MAX_SOURCE_BYTES
        try:
            return resolve_graph_config(self.config).max_source_bytes
        except ValueError as exc:
            raise GraphRAGInputSyncError(str(exc)) from exc

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


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _records_payload(records: list[dict[str, Any]]) -> str:
    return (
        json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    )


def _input_size_warnings(input_size_bytes: int) -> list[str]:
    if input_size_bytes <= GRAPH_INPUT_SIZE_WARNING_BYTES:
        return []
    mib = input_size_bytes / (1024 * 1024)
    limit_mib = GRAPH_INPUT_SIZE_WARNING_BYTES // (1024 * 1024)
    return [
        "GraphRAG input is "
        f"{mib:.1f} MiB; consider splitting the corpus or using graph-only "
        f"updates intentionally above {limit_mib} MiB."
    ]


def _source_hashes(records: list[dict[str, Any]]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for record in records:
        source_id = record.get("source_id") or record.get("id")
        source_hash = record.get("source_hash")
        if isinstance(source_id, str) and isinstance(source_hash, str):
            hashes[source_id] = source_hash
    return hashes
