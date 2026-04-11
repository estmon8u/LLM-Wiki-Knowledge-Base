from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
from typing import Optional
import uuid

from src.models.source_models import RawSourceRecord
from src.services.manifest_service import ManifestService
from src.services.normalization_service import NormalizationService
from src.services.project_service import ProjectPaths, slugify, utc_now_iso


@dataclass
class IngestResult:
    created: bool
    source: Optional[RawSourceRecord]
    message: str
    duplicate_of: Optional[RawSourceRecord] = None


class IngestService:
    def __init__(
        self,
        paths: ProjectPaths,
        manifest_service: ManifestService,
        normalization_service: Optional[NormalizationService] = None,
    ) -> None:
        self.paths = paths
        self.manifest_service = manifest_service
        self.normalization_service = normalization_service or NormalizationService()

    def ingest_path(self, raw_input_path: Path) -> IngestResult:
        source_path = raw_input_path.resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        normalized = self.normalization_service.normalize_path(source_path)
        content_hash = hashlib.sha256(
            normalized.normalized_text.encode("utf-8")
        ).hexdigest()

        duplicate = self.manifest_service.find_by_hash(content_hash)
        if duplicate is not None:
            return IngestResult(
                created=False,
                source=duplicate,
                duplicate_of=duplicate,
                message=f"Duplicate source skipped: {duplicate.title}",
            )

        title = normalized.title
        slug = self._unique_slug(slugify(title))
        raw_destination = (
            self.paths.raw_sources_dir / f"{slug}{source_path.suffix.lower()}"
        )
        raw_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, raw_destination)

        normalized_destination = (
            self.paths.raw_normalized_dir / f"{slug}{normalized.normalized_suffix}"
        )
        normalized_destination.parent.mkdir(parents=True, exist_ok=True)
        normalized_destination.write_text(normalized.normalized_text, encoding="utf-8")

        source = RawSourceRecord(
            source_id=str(uuid.uuid4()),
            slug=slug,
            title=title,
            origin=str(source_path),
            source_type="file",
            raw_path=raw_destination.relative_to(self.paths.root).as_posix(),
            normalized_path=normalized_destination.relative_to(
                self.paths.root
            ).as_posix(),
            content_hash=content_hash,
            ingested_at=utc_now_iso(),
            metadata={
                "original_name": source_path.name,
                "source_extension": source_path.suffix.lower(),
                **normalized.metadata,
            },
        )
        self.manifest_service.save_source(source)
        return IngestResult(
            created=True,
            source=source,
            message=f"Ingested {source.title} as {source.slug}",
        )

    def _unique_slug(self, base_slug: str) -> str:
        existing = {source.slug for source in self.manifest_service.list_sources()}
        if base_slug not in existing:
            return base_slug
        index = 2
        while f"{base_slug}-{index}" in existing:
            index += 1
        return f"{base_slug}-{index}"
