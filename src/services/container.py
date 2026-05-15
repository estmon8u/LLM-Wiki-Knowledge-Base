"""Typed service container used by CLI command contexts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from src.services.compile_service import CompileService
    from src.services.concept_service import ConceptService
    from src.services.config_service import ConfigService
    from src.services.diff_service import DiffService
    from src.services.doctor_service import DoctorService
    from src.services.export_service import ExportService
    from src.services.graph_ask_controller_service import GraphAskControllerService
    from src.services.graphrag_command_service import GraphRAGCommandService
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
    from src.services.project_service import ProjectService
    from src.services.query_router_service import QueryRouterService
    from src.services.query_service import QueryService
    from src.services.review_service import ReviewService
    from src.services.search_service import SearchService
    from src.services.status_service import StatusService
    from src.storage.compile_run_store import CompileRunStore


@dataclass
class ServiceContainer(Mapping[str, Any]):
    """Named application services with mapping compatibility for older tests."""

    project: ProjectService
    config: ConfigService
    manifest: ManifestService
    ingest: IngestService
    compile: CompileService
    concepts: ConceptService
    diff: DiffService
    doctor: DoctorService
    lint: LintService
    search: SearchService
    status: StatusService
    query: QueryService
    export: ExportService
    graphrag_command: GraphRAGCommandService
    graphrag_workspace: GraphRAGWorkspaceService
    graphrag_status: GraphRAGStatusService
    graphrag_query: GraphRAGQueryService
    graphrag_find: GraphRAGFindService
    query_router: QueryRouterService
    graph_ask_controller: GraphAskControllerService
    graphrag_wiki_export: GraphRAGWikiExportService
    graphrag_input_sync: GraphRAGInputSyncService
    graphrag_sync: GraphRAGSyncService
    review: ReviewService
    compile_run_store: CompileRunStore

    _NAMES: ClassVar[tuple[str, ...]] = (
        "project",
        "config",
        "manifest",
        "graphrag_input_sync",
        "ingest",
        "compile",
        "concepts",
        "diff",
        "doctor",
        "lint",
        "review",
        "search",
        "status",
        "query",
        "export",
        "graphrag_command",
        "graphrag_workspace",
        "graphrag_status",
        "graphrag_sync",
        "graphrag_query",
        "graphrag_find",
        "graphrag_wiki_export",
        "query_router",
        "graph_ask_controller",
        "compile_run_store",
    )

    def __getitem__(self, key: str) -> Any:
        if key not in self._NAMES:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._NAMES)

    def __len__(self) -> int:
        return len(self._NAMES)
