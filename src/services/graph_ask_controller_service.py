from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.services.config_service import GraphRAGRuntimeConfig, resolve_graph_config
from src.services.graphrag_query_service import (
    GraphRAGQueryAnswer,
    GraphRAGQueryService,
)
from src.services.graphrag_status_service import GraphRAGStatus, GraphRAGStatusService
from src.services.project_service import ProjectPaths
from src.services.query_router_service import QueryRouterService


class GraphAskControllerError(RuntimeError):
    pass


class GraphAskControllerService:
    def __init__(
        self,
        paths: ProjectPaths,
        config: dict[str, Any],
        status_service: GraphRAGStatusService,
        router_service: QueryRouterService,
        query_service: GraphRAGQueryService,
    ) -> None:
        self.paths = paths
        self.config = config
        self.status_service = status_service
        self.router_service = router_service
        self.query_service = query_service

    def ask(
        self,
        question: str,
        *,
        method: str = "auto",
        community_level: int | None = None,
        dynamic_community_selection: bool | None = None,
        response_type: str | None = None,
        verbose: bool = False,
        save: bool = False,
        save_as: str | None = None,
    ) -> GraphRAGQueryAnswer:
        graph_config = self._resolve_graph_config()
        status = self.status_service.status()
        if _graph_ready_for_query(status):
            self._require_credentials(graph_config)

        route = self.router_service.route(question, method=method)
        answer = self.query_service.ask(
            question,
            method=route.method,
            community_level=community_level,
            dynamic_community_selection=dynamic_community_selection,
            response_type=response_type,
            verbose=verbose,
        )
        answer.retriever = "graph"
        answer.planner = route.planner
        answer.route_reason = route.reason
        answer.claim_support = "unverified"
        if save or save_as:
            self.query_service.save_answer(answer, slug=save_as)
        return answer

    def _resolve_graph_config(self) -> GraphRAGRuntimeConfig:
        try:
            return resolve_graph_config(self.config)
        except ValueError as exc:
            raise GraphAskControllerError(str(exc)) from exc

    def _require_credentials(self, graph_config: GraphRAGRuntimeConfig) -> None:
        if os.environ.get(graph_config.api_key_env):
            return
        env_file = self.paths.graph_dir / "graphrag" / ".env"
        if _env_file_has_key(env_file, graph_config.api_key_env):
            return
        raise GraphAskControllerError(
            f"GraphRAG API key is not configured. Set {graph_config.api_key_env} "
            "or add it to graph/graphrag/.env before running `kb ask`."
        )


def _graph_ready_for_query(status: GraphRAGStatus) -> bool:
    return (
        status.workspace_initialized
        and status.input_exists
        and status.input_document_count > 0
        and status.output_present
        and status.last_index_success is not False
    )


def _env_file_has_key(path: Path, key: str) -> bool:
    if not path.exists():
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == key and value.strip().strip('"').strip("'"):
            return True
    return False
