from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from src.providers import ProviderError
from src.services.compile_service import CompileResult, CompileService
from src.services.concept_service import ConceptGenerationResult, ConceptService
from src.services.ingest_service import IngestResult, IngestService
from src.services.search_service import SearchService


@dataclass
class UpdateOptions:
    source_paths: tuple[Path, ...] = ()
    force: bool = False
    resume: bool = False


@dataclass
class IngestSummary:
    path: Path
    is_dir: bool
    created_count: int = 0
    message: str = ""


@dataclass
class UpdateResult:
    ingest_summaries: list[IngestSummary] = field(default_factory=list)
    compile_result: Optional[CompileResult] = None
    concept_result: Optional[ConceptGenerationResult] = None
    search_refreshed: bool = False

    @property
    def ok(self) -> bool:
        return self.compile_result is not None


class UpdateService:
    def __init__(
        self,
        *,
        ingest_service: IngestService,
        compile_service: CompileService,
        concept_service: ConceptService,
        search_service: SearchService,
        config: dict[str, Any],
    ) -> None:
        self._ingest = ingest_service
        self._compile = compile_service
        self._concepts = concept_service
        self._search = search_service
        self._config = config

    def preflight(self) -> None:
        """Raise if provider is missing or broken."""
        provider_name = self._config.get("provider", {}).get("name")
        if not provider_name:
            raise UpdatePreflightError(
                "Provider is not configured, so the KB cannot be updated yet.\n"
                "Next: add a provider section to kb.config.yaml and set the "
                "matching API key environment variable."
            )
        actual_provider = getattr(self._compile, "provider", None)
        if actual_provider is not None and hasattr(actual_provider, "ensure_available"):
            actual_provider.ensure_available()

    def run(
        self,
        options: UpdateOptions,
        *,
        ingest_progress: Callable[[Path], None] | None = None,
        compile_progress: Callable[[str], None] | None = None,
    ) -> UpdateResult:
        if options.force and options.resume:
            raise ValueError("--resume cannot be combined with --force.")

        self.preflight()
        result = UpdateResult()

        # Ingest phase
        for source_path in options.source_paths:
            summary = self._ingest_one(source_path, progress=ingest_progress)
            result.ingest_summaries.append(summary)

        # Compile phase
        self._compile.plan(force=options.force, resume=options.resume)
        try:
            result.compile_result = self._compile.compile(
                force=options.force,
                resume=options.resume,
                progress_callback=compile_progress,
            )
        except (ProviderError, ValueError):
            raise

        # Concepts phase
        result.concept_result = self._concepts.generate()

        # Search refresh
        self._search.refresh(force=True)
        result.search_refreshed = True

        return result

    # ------------------------------------------------------------------

    def _ingest_one(
        self,
        source_path: Path,
        *,
        progress: Callable[[Path], None] | None = None,
    ) -> IngestSummary:
        if source_path.is_dir():
            dir_result = self._ingest.ingest_directory(
                source_path,
                progress_callback=progress,
            )
            return IngestSummary(
                path=source_path,
                is_dir=True,
                created_count=dir_result.created_count,
            )
        else:
            file_result = self._ingest.ingest_path(source_path)
            return IngestSummary(
                path=source_path,
                is_dir=False,
                created_count=1 if file_result.created else 0,
                message=""
                if file_result.created
                else f"Already present: {source_path.name}",
            )


class UpdatePreflightError(Exception):
    pass
