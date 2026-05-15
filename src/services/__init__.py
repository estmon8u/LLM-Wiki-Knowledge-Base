"""Service construction helpers for the capstone CLI."""

from __future__ import annotations

from typing import Any

from src.providers import build_lazy_provider, build_provider
from src.services.compile_service import CompileService
from src.services.concept_service import ConceptService
from src.services.config_service import ConfigService
from src.services.diff_service import DiffService
from src.services.doctor_service import DoctorService
from src.services.export_service import ExportService
from src.services.graph_ask_controller_service import GraphAskControllerService
from src.services.graphrag_command_service import GraphRAGCommandService
from src.services.container import ServiceContainer
from src.services.graphrag_find_service import GraphRAGFindService
from src.services.graphrag_input_sync_service import GraphRAGInputSyncService
from src.services.graphrag_query_service import GraphRAGQueryService
from src.services.graphrag_status_service import GraphRAGStatusService
from src.services.graphrag_sync_service import GraphRAGSyncService
from src.services.graphrag_wiki_export_service import GraphRAGWikiExportService
from src.services.graphrag_workspace_service import GraphRAGWorkspaceService
from src.services.ingest_service import IngestService
from src.services.lint_service import LintService
from src.services.manifest_service import ManifestService
from src.services.project_service import ProjectPaths, ProjectService
from src.services.query_router_service import QueryRouterService
from src.services.query_service import QueryService
from src.services.review_service import ReviewService
from src.services.search_service import SearchService
from src.services.status_service import StatusService
from src.storage.compile_run_store import CompileRunStore


def build_services(
    paths: ProjectPaths,
    config: dict[str, Any],
) -> ServiceContainer:
    """Builds services.

    Args:
        paths: Resolved project paths used by the service.
        config: Loaded knowledge-base configuration mapping.

    Returns:
        dict[str, Any] produced by the operation.
    """
    config_service = ConfigService(paths)
    manifest_service = ManifestService(paths)
    search_service = SearchService(paths)
    provider = build_lazy_provider(config, provider_builder=build_provider)
    schema_text = config_service.load_schema()
    compile_run_store = CompileRunStore(paths.graph_exports_dir / "compile_runs.json")
    graphrag_command_service = GraphRAGCommandService(paths)
    graphrag_status_service = GraphRAGStatusService(paths)
    graphrag_input_sync_service = GraphRAGInputSyncService(paths, manifest_service)
    graphrag_workspace_service = GraphRAGWorkspaceService(
        paths,
        graphrag_command_service,
        config=config,
    )
    query_router_service = QueryRouterService(graphrag_status_service)
    compile_service = CompileService(
        paths,
        config,
        manifest_service,
        provider=provider,
        compile_run_store=compile_run_store,
        schema_text=schema_text,
    )
    graphrag_query_service = GraphRAGQueryService(
        paths,
        graphrag_command_service,
        graphrag_status_service,
        search_service,
        refresh_index=compile_service.refresh_index,
    )
    graphrag_find_service = GraphRAGFindService(paths, graphrag_status_service)
    return ServiceContainer(
        project=ProjectService(paths),
        config=config_service,
        manifest=manifest_service,
        ingest=IngestService(paths, manifest_service, config=config),
        compile=compile_service,
        concepts=ConceptService(paths, provider=provider),
        diff=DiffService(paths, manifest_service),
        doctor=DoctorService(
            paths,
            config,
            provider=provider,
            graphrag_status_service=graphrag_status_service,
        ),
        lint=LintService(
            paths,
            config,
            manifest_service,
            graphrag_status_service=graphrag_status_service,
        ),
        search=search_service,
        status=StatusService(
            paths,
            manifest_service,
            config=config,
            graphrag_status_service=graphrag_status_service,
        ),
        query=QueryService(
            paths,
            search_service,
            provider=provider,
            refresh_index=compile_service.refresh_index,
            schema_text=schema_text,
        ),
        export=ExportService(paths),
        graphrag_command=graphrag_command_service,
        graphrag_workspace=graphrag_workspace_service,
        graphrag_status=graphrag_status_service,
        graphrag_query=graphrag_query_service,
        graphrag_find=graphrag_find_service,
        query_router=query_router_service,
        graph_ask_controller=GraphAskControllerService(
            paths,
            config,
            graphrag_status_service,
            query_router_service,
            graphrag_query_service,
        ),
        graphrag_wiki_export=GraphRAGWikiExportService(
            paths,
            graphrag_status_service,
            search_service,
            refresh_index=compile_service.refresh_index,
        ),
        graphrag_input_sync=graphrag_input_sync_service,
        graphrag_sync=GraphRAGSyncService(
            paths,
            graphrag_workspace_service,
            graphrag_input_sync_service,
            graphrag_status_service,
            graphrag_command_service,
        ),
        review=ReviewService(paths, provider=provider),
        compile_run_store=compile_run_store,
    )
