from __future__ import annotations

"""Service construction helpers for the capstone CLI."""

from typing import Any

from kb.services.compile_service import CompileService
from kb.services.config_service import ConfigService
from kb.services.export_service import ExportService
from kb.services.ingest_service import IngestService
from kb.services.lint_service import LintService
from kb.services.manifest_service import ManifestService
from kb.services.project_service import ProjectPaths, ProjectService
from kb.services.query_service import QueryService
from kb.services.search_service import SearchService
from kb.services.status_service import StatusService


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
