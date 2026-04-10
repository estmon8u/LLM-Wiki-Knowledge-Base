from __future__ import annotations

"""Service construction helpers for the capstone CLI."""

from typing import Any

from src.services.compile_service import CompileService
from src.services.config_service import ConfigService
from src.services.export_service import ExportService
from src.services.ingest_service import IngestService
from src.services.lint_service import LintService
from src.services.manifest_service import ManifestService
from src.services.project_service import ProjectPaths, ProjectService
from src.services.query_service import QueryService
from src.services.search_service import SearchService
from src.services.status_service import StatusService


def build_services(paths: ProjectPaths, config: dict[str, Any]) -> dict[str, Any]:
    config_service = ConfigService(paths)
    manifest_service = ManifestService(paths)
    search_service = SearchService(paths)
    return {
        "project": ProjectService(paths),
        "config": config_service,
        "manifest": manifest_service,
        "ingest": IngestService(paths, manifest_service),
        "compile": CompileService(paths, config, manifest_service),
        "lint": LintService(paths, config, manifest_service),
        "search": search_service,
        "status": StatusService(paths, manifest_service),
        "query": QueryService(search_service),
        "export": ExportService(paths),
    }
