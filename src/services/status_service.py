from __future__ import annotations

from src.models.wiki_models import StatusSnapshot
from src.services.manifest_service import ManifestService
from src.services.project_service import ProjectPaths


class StatusService:
    def __init__(self, paths: ProjectPaths, manifest_service: ManifestService) -> None:
        self.paths = paths
        self.manifest_service = manifest_service

    def snapshot(self, *, initialized: bool) -> StatusSnapshot:
        sources = (
            self.manifest_service.list_sources()
            if self.paths.raw_manifest_file.exists()
            else []
        )
        compiled_sources = sum(
            1 for source in sources if source.compiled_from_hash == source.content_hash
        )
        concept_page_count = (
            len(list(self.paths.wiki_concepts_dir.glob("*.md")))
            if self.paths.wiki_concepts_dir.exists()
            else 0
        )
        last_compile_at = max(
            (source.compiled_at for source in sources if source.compiled_at),
            default=None,
        )
        return StatusSnapshot(
            initialized=initialized,
            source_count=len(sources),
            compiled_source_count=compiled_sources,
            concept_page_count=concept_page_count,
            last_compile_at=last_compile_at,
        )
