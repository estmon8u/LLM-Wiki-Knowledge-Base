from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil
from typing import Callable, Optional
import uuid

from src.models.source_models import RawSourceRecord
from src.services.manifest_service import ManifestService
from src.services.normalization_service import (
    NormalizationService,
    is_supported_source_path,
)
from src.services.project_service import ProjectPaths, slugify, utc_now_iso


@dataclass
class IngestResult:
    created: bool
    source: Optional[RawSourceRecord]
    message: str
    duplicate_of: Optional[RawSourceRecord] = None


@dataclass
class IngestDirectoryResult:
    directory_path: Path
    scanned_file_count: int
    results: tuple[IngestResult, ...]

    @property
    def created_results(self) -> tuple[IngestResult, ...]:
        return tuple(result for result in self.results if result.created)

    @property
    def duplicate_results(self) -> tuple[IngestResult, ...]:
        return tuple(result for result in self.results if not result.created)

    @property
    def created_count(self) -> int:
        return len(self.created_results)

    @property
    def duplicate_count(self) -> int:
        return len(self.duplicate_results)


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
        if source_path.is_dir():
            raise ValueError(f"Directory ingest requires --recursive: {source_path}")

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

    def discover_source_paths(self, raw_input_path: Path) -> tuple[Path, ...]:
        directory_path = raw_input_path.resolve()
        if not directory_path.exists():
            raise FileNotFoundError(f"Source directory not found: {directory_path}")
        if not directory_path.is_dir():
            raise ValueError(f"Source path is not a directory: {directory_path}")

        return self._supported_source_paths(directory_path)

    def ingest_directory(
        self,
        raw_input_path: Path,
        progress_callback: Optional[Callable[[Path], None]] = None,
    ) -> IngestDirectoryResult:
        directory_path = raw_input_path.resolve()
        candidate_paths = self.discover_source_paths(directory_path)

        if not candidate_paths:
            raise ValueError(
                f"No supported source files found under directory: {directory_path}"
            )

        results = []
        for path in candidate_paths:
            results.append(self.ingest_path(path))
            if progress_callback is not None:
                progress_callback(path)

        return IngestDirectoryResult(
            directory_path=directory_path,
            scanned_file_count=len(candidate_paths),
            results=tuple(results),
        )

    def _unique_slug(self, base_slug: str) -> str:
        existing = {source.slug for source in self.manifest_service.list_sources()}
        if base_slug not in existing:
            return base_slug
        index = 2
        while f"{base_slug}-{index}" in existing:
            index += 1
        return f"{base_slug}-{index}"

    def _supported_source_paths(self, directory_path: Path) -> tuple[Path, ...]:
        candidates = [
            path
            for path in directory_path.rglob("*")
            if path.is_file() and is_supported_source_path(path)
        ]
        return tuple(
            sorted(
                candidates,
                key=lambda path: path.relative_to(directory_path).as_posix().lower(),
            )
        )
