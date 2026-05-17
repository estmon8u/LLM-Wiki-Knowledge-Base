"""Source ingestion, duplicate detection, and normalized artifact writing."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.services.manifest_service import ManifestService
from graphwiki_kb.services.normalization_service import (
    NormalizationService,
    is_supported_source_path,
)
from graphwiki_kb.services.project_service import (
    ProjectPaths,
    atomic_copy_file,
    atomic_write_text,
    slugify,
    utc_now_iso,
)

_TOOLING_DIRECTORY_NAMES = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
}
_PROJECT_MANAGED_DIRECTORY_NAMES = {
    "graph",
    "raw",
    "vault",
    "wiki",
}
_PROJECT_MANAGED_FILE_NAMES = {
    "kb.config.yaml",
    "kb.schema.md",
}


@dataclass
class IngestResult:
    """Outcome for a single source-file ingest attempt."""

    created: bool
    source: RawSourceRecord | None
    message: str
    duplicate_of: RawSourceRecord | None = None


@dataclass
class IngestDirectoryResult:
    """Outcome for scanning and ingesting one source directory."""

    directory_path: Path
    scanned_file_count: int
    results: tuple[IngestResult, ...]

    @property
    def created_results(self) -> tuple[IngestResult, ...]:
        """Return results that created new manifest entries."""
        return tuple(result for result in self.results if result.created)

    @property
    def duplicate_results(self) -> tuple[IngestResult, ...]:
        """Return results skipped as duplicate sources."""
        return tuple(result for result in self.results if not result.created)

    @property
    def created_count(self) -> int:
        """Return the number of newly ingested sources."""
        return len(self.created_results)

    @property
    def duplicate_count(self) -> int:
        """Return the number of duplicate sources."""
        return len(self.duplicate_results)


class IngestService:
    """Copies source files, normalizes content, and records manifest entries."""

    def __init__(
        self,
        paths: ProjectPaths,
        manifest_service: ManifestService,
        normalization_service: NormalizationService | None = None,
        config: dict[str, object] | None = None,
    ) -> None:
        self.paths = paths
        self.manifest_service = manifest_service
        self.normalization_service = normalization_service or NormalizationService(
            config
        )

    def ingest_path(self, raw_input_path: Path) -> IngestResult:
        """Ingest one supported source file into raw and normalized storage."""
        source_path = raw_input_path.resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        if source_path.is_dir():
            raise ValueError(f"Directory ingest requires --recursive: {source_path}")

        origin_hash = _file_sha256(source_path)
        duplicate = self.manifest_service.find_by_origin_hash(origin_hash)
        if duplicate is not None:
            return IngestResult(
                created=False,
                source=duplicate,
                duplicate_of=duplicate,
                message=f"Duplicate source skipped: {duplicate.title}",
            )

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
        atomic_copy_file(source_path, raw_destination)

        normalized_destination = (
            self.paths.raw_normalized_dir / f"{slug}{normalized.normalized_suffix}"
        )
        atomic_write_text(normalized_destination, normalized.normalized_text)

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
            origin_hash=origin_hash,
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
        """Return supported source files under a directory in stable order."""
        directory_path = raw_input_path.resolve()
        if not directory_path.exists():
            raise FileNotFoundError(f"Source directory not found: {directory_path}")
        if not directory_path.is_dir():
            raise ValueError(f"Source path is not a directory: {directory_path}")

        return self._supported_source_paths(directory_path)

    def ingest_directory(
        self,
        raw_input_path: Path,
        progress_callback: Callable[[Path], None] | None = None,
    ) -> IngestDirectoryResult:
        """Ingest every supported source file under a directory."""
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
            if path.is_file()
            and is_supported_source_path(path)
            and not self._is_excluded_directory_candidate(path)
        ]
        return tuple(
            sorted(
                candidates,
                key=lambda path: path.relative_to(directory_path).as_posix().lower(),
            )
        )

    def _is_excluded_directory_candidate(self, path: Path) -> bool:
        if any(part in _TOOLING_DIRECTORY_NAMES for part in path.parts):
            return True
        try:
            relative = path.resolve().relative_to(self.paths.root.resolve())
        except ValueError:
            return False
        if relative.as_posix() in _PROJECT_MANAGED_FILE_NAMES:
            return True
        return any(part in _PROJECT_MANAGED_DIRECTORY_NAMES for part in relative.parts)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
