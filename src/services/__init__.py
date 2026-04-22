from __future__ import annotations

"""Service construction helpers for the capstone CLI."""

from typing import Any

from src.providers import build_provider
from src.services.compile_service import CompileService
from src.services.concept_service import ConceptService
from src.services.config_service import ConfigService
from src.services.diff_service import DiffService
from src.services.doctor_service import DoctorService
from src.services.export_service import ExportService
from src.services.ingest_service import IngestService
from src.services.lint_service import LintService
from src.services.manifest_service import ManifestService
from src.services.project_service import ProjectPaths, ProjectService
from src.services.query_service import QueryService
from src.services.review_service import ReviewService
from src.services.search_service import SearchService
from src.services.status_service import StatusService
from src.storage.compile_run_store import CompileRunStore


def build_services(
    paths: ProjectPaths,
    config: dict[str, Any],
) -> dict[str, Any]:
    config_service = ConfigService(paths)
    manifest_service = ManifestService(paths)
    search_service = SearchService(paths)
    provider = build_provider(config)
    schema_text = config_service.load_schema()
    compile_run_store = CompileRunStore(paths.graph_exports_dir / "compile_runs.json")
    compile_service = CompileService(
        paths,
        config,
        manifest_service,
        provider=provider,
        compile_run_store=compile_run_store,
        schema_text=schema_text,
    )
    return {
        "project": ProjectService(paths),
        "config": config_service,
        "manifest": manifest_service,
        "ingest": IngestService(paths, manifest_service),
        "compile": compile_service,
        "concepts": ConceptService(paths),
        "diff": DiffService(paths, manifest_service),
        "doctor": DoctorService(paths, config, provider=provider),
        "lint": LintService(paths, config, manifest_service),
        "search": search_service,
        "status": StatusService(paths, manifest_service, config=config),
        "query": QueryService(
            paths,
            search_service,
            provider=provider,
            refresh_index=compile_service.refresh_index,
            schema_text=schema_text,
        ),
        "export": ExportService(paths),
        "review": ReviewService(paths, provider=provider),
        "compile_run_store": compile_run_store,
    }
