from __future__ import annotations

import hashlib

from src.models.wiki_models import DiffEntry, DiffReport
from src.services.manifest_service import ManifestService
from src.services.project_service import ProjectPaths


class DiffService:
    def __init__(self, paths: ProjectPaths, manifest_service: ManifestService) -> None:
        self.paths = paths
        self.manifest_service = manifest_service

    def diff(self) -> DiffReport:
        sources = (
            self.manifest_service.list_sources()
            if self.paths.raw_manifest_file.exists()
            else []
        )
        entries: list[DiffEntry] = []
        for source in sources:
            if source.compiled_at is None or source.compiled_from_hash is None:
                entries.append(
                    DiffEntry(
                        source_id=source.source_id,
                        slug=source.slug,
                        title=source.title,
                        status="new",
                        raw_path=source.raw_path,
                        details="not yet compiled",
                    )
                )
            else:
                # Recompute hash from actual normalized file on disk.
                current_hash = self._current_content_hash(source)
                if current_hash != source.compiled_from_hash:
                    reason = (
                        "source changed since last compile"
                        if current_hash == source.content_hash
                        else "normalized file changed on disk"
                    )
                    entries.append(
                        DiffEntry(
                            source_id=source.source_id,
                            slug=source.slug,
                            title=source.title,
                            status="changed",
                            raw_path=source.raw_path,
                            details=reason,
                        )
                    )
                else:
                    entries.append(
                        DiffEntry(
                            source_id=source.source_id,
                            slug=source.slug,
                            title=source.title,
                            status="up_to_date",
                            raw_path=source.raw_path,
                        )
                    )
        return DiffReport(entries=entries)

    def _current_content_hash(self, source) -> str:
        """Compute the actual SHA-256 of the normalized file on disk.

        Falls back to the manifest ``content_hash`` if the normalized file
        cannot be read (e.g. deleted).
        """
        norm_path = source.normalized_path or source.raw_path
        full_path = self.paths.root / norm_path
        try:
            text = full_path.read_text(encoding="utf-8")
            return hashlib.sha256(text.encode("utf-8")).hexdigest()
        except OSError:
            return source.content_hash
