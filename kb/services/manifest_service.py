from __future__ import annotations

import json
from typing import Any, Optional

from kb.models.source_models import RawSourceRecord
from kb.services.project_service import ProjectPaths, utc_now_iso


class ManifestService:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def ensure_manifest(self) -> bool:
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
        payload = self._read()
        return [RawSourceRecord.from_dict(item) for item in payload["sources"]]

    def find_by_hash(self, content_hash: str) -> Optional[RawSourceRecord]:
        for source in self.list_sources():
            if source.content_hash == content_hash:
                return source
        return None

    def save_source(self, source: RawSourceRecord) -> None:
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
        self.paths.raw_manifest_file.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
