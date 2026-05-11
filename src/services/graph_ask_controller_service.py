from __future__ import annotations

import os
from typing import Any

from src.services.config_service import GraphRAGRuntimeConfig, resolve_graph_config
from src.services.graphrag_defaults import env_file_has_key
from src.services.graphrag_query_service import (
    GraphRAGQueryAnswer,
    GraphRAGQueryService,
)
from src.services.graphrag_status_service import GraphRAGStatus, GraphRAGStatusService
from src.services.manifest_service import ManifestService
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
        manifest_service: ManifestService | None = None,
    ) -> None:
        self.paths = paths
        self.config = config
        self.status_service = status_service
        self.router_service = router_service
        self.query_service = query_service
        self.manifest_service = manifest_service

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

        staleness = self._check_staleness(status)

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
        answer.claim_support = _assess_claim_support(answer, staleness)
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
        dot_env = self.paths.graph_dir / "graphrag" / ".env"
        if env_file_has_key(dot_env, graph_config.api_key_env):
            return
        raise GraphAskControllerError(
            f"GraphRAG API key is not configured. Set {graph_config.api_key_env} "
            "or add it to graph/graphrag/.env before running `kb ask`."
        )

    def _check_staleness(self, status: GraphRAGStatus) -> list[str]:
        """Return human-readable staleness warnings (empty = fresh)."""
        warnings: list[str] = []
        manifest_mtime = self._manifest_mtime()
        if manifest_mtime is not None and status.input_updated_at:
            if manifest_mtime > status.input_updated_at:
                warnings.append(
                    "Manifest is newer than graph input. Run `kb graph sync`."
                )
        if status.input_updated_at and status.output_updated_at:
            if status.input_updated_at > status.output_updated_at:
                warnings.append(
                    "Graph input is newer than index output. "
                    "Run `kb graph index --method fast`."
                )
        return warnings

    def _manifest_mtime(self) -> str | None:
        path = self.paths.raw_manifest_file
        if not path.exists():
            return None
        try:
            from datetime import datetime, timezone

            ts = path.stat().st_mtime
            return (
                datetime.fromtimestamp(ts, tz=timezone.utc)
                .replace(microsecond=0)
                .isoformat()
            )
        except OSError:
            return None


def _graph_ready_for_query(status: GraphRAGStatus) -> bool:
    return (
        status.workspace_initialized
        and status.input_exists
        and status.input_document_count > 0
        and status.output_present
        and status.last_index_success is not False
    )


def _assess_claim_support(answer: GraphRAGQueryAnswer, staleness: list[str]) -> str:
    """Simple claim-support assessment based on available evidence."""
    if staleness:
        return "stale-index"
    if not answer.answer or not answer.answer.strip():
        return "no-answer"
    source_trace = answer.source_trace or {}
    has_index = bool(source_trace.get("index_run_id"))
    has_hash = bool(source_trace.get("input_manifest_hash"))
    if has_index and has_hash:
        return "graph-grounded"
    return "unverified"
