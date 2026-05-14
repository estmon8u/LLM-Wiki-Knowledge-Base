"""Manifest service service behavior for the knowledge-base workflow.

This module belongs to `src.services.manifest_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""


from __future__ import annotations

import json
from typing import Any, Optional

from src.models.source_models import RawSourceRecord
from src.services.project_service import ProjectPaths, atomic_write_text, utc_now_iso


class ManifestService:
    """Coordinates manifest operations.

    Attributes:
        See annotated class attributes for stored values.
    """

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def ensure_manifest(self) -> bool:
        """Ensure manifest.

        Returns:
            bool produced by the operation.
        """
        if self.paths.raw_manifest_file.exists():
            return False
        payload = {
            "version": 1,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "sources": [],
        }
        self._write(payload)
        return True

    def list_sources(self) -> list[RawSourceRecord]:
        """List sources.

        Returns:
            list[RawSourceRecord] produced by the operation.
        """
        payload = self._read()
        return [RawSourceRecord.from_dict(item) for item in payload["sources"]]

    def find_by_hash(self, content_hash: str) -> Optional[RawSourceRecord]:
        """Find by hash.

        Args:
            content_hash: Content hash value used by the operation.

        Returns:
            Optional[RawSourceRecord] produced by the operation.
        """
        for source in self.list_sources():
            if source.content_hash == content_hash:
                return source
        return None

    def find_by_origin_hash(self, origin_hash: str) -> Optional[RawSourceRecord]:
        """Find by origin hash.

        Args:
            origin_hash: Origin hash value used by the operation.

        Returns:
            Optional[RawSourceRecord] produced by the operation.
        """
        for source in self.list_sources():
            if source.origin_hash == origin_hash:
                return source
        return None

    def save_source(self, source: RawSourceRecord) -> None:
        """Saves source.

        Args:
            source: Source record or path being processed.
        """
        payload = self._read()
        sources = [RawSourceRecord.from_dict(item) for item in payload["sources"]]
        updated = False
        for index, existing in enumerate(sources):
            if existing.source_id == source.source_id:
                sources[index] = source
                updated = True
                break
        if not updated:
            sources.append(source)
        payload["sources"] = [item.to_dict() for item in sources]
        payload["updated_at"] = utc_now_iso()
        self._write(payload)

    def _read(self) -> dict[str, Any]:
        if not self.paths.raw_manifest_file.exists():
            self.ensure_manifest()
        with self.paths.raw_manifest_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write(self, payload: dict[str, Any]) -> None:
        self.paths.raw_manifest_file.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            self.paths.raw_manifest_file,
            json.dumps(payload, indent=2, sort_keys=True),
        )
