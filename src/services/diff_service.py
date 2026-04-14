from __future__ import annotations

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
            elif source.content_hash != source.compiled_from_hash:
                entries.append(
                    DiffEntry(
                        source_id=source.source_id,
                        slug=source.slug,
                        title=source.title,
                        status="changed",
                        raw_path=source.raw_path,
                        details="source changed since last compile",
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
