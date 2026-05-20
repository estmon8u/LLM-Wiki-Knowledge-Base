"""Typed service container used by CLI command contexts."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from graphwiki_kb.services.compile_service import CompileService
    from graphwiki_kb.services.concept_service import ConceptService
    from graphwiki_kb.services.config_service import ConfigService
    from graphwiki_kb.services.diff_service import DiffService
    from graphwiki_kb.services.doctor_service import DoctorService
    from graphwiki_kb.services.export_service import ExportService
    from graphwiki_kb.services.graph_ask_controller_service import (
        GraphAskControllerService,
    )
    from graphwiki_kb.services.graphrag_command_service import GraphRAGCommandService
    from graphwiki_kb.services.graphrag_find_service import GraphRAGFindService
    from graphwiki_kb.services.graphrag_input_sync_service import (
        GraphRAGInputSyncService,
    )
    from graphwiki_kb.services.graphrag_query_service import GraphRAGQueryService
    from graphwiki_kb.services.graphrag_status_service import GraphRAGStatusService
    from graphwiki_kb.services.graphrag_sync_service import GraphRAGSyncService
    from graphwiki_kb.services.graphrag_wiki_export_service import (
        GraphRAGWikiExportService,
    )
    from graphwiki_kb.services.graphrag_workspace_service import (
        GraphRAGWorkspaceService,
    )
    from graphwiki_kb.services.ingest_service import IngestService
    from graphwiki_kb.services.lint_service import LintService
    from graphwiki_kb.services.manifest_service import ManifestService
    from graphwiki_kb.services.project_service import ProjectService
    from graphwiki_kb.services.query_router_service import QueryRouterService
    from graphwiki_kb.services.query_service import QueryService
    from graphwiki_kb.services.review_service import ReviewService
    from graphwiki_kb.services.search_service import SearchService
    from graphwiki_kb.services.status_service import StatusService
    from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService
    from graphwiki_kb.services.wikigraph_query_service import WikiGraphQueryFacade
    from graphwiki_kb.storage.compile_run_store import CompileRunStore


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
    wikigraph_index: WikiGraphIndexService
    wikigraph_query: WikiGraphQueryFacade
    compile_run_store: CompileRunStore

    _NAMES: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._NAMES = tuple(f.name for f in dataclasses.fields(cls))

    def __post_init__(self) -> None:
        if not type(self)._NAMES:
            type(self)._NAMES = tuple(f.name for f in dataclasses.fields(self))

    def __getitem__(self, key: str) -> Any:
        if key not in self._NAMES:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._NAMES)

    def __len__(self) -> int:
        return len(self._NAMES)
