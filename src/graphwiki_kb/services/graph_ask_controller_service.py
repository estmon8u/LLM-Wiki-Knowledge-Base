"""GraphRAG ask controller and method router boundary."""

from __future__ import annotations

import os
from typing import Any

from graphwiki_kb.services.config_service import (
    GraphRAGRuntimeConfig,
    resolve_graph_config,
)
from graphwiki_kb.services.graphrag_defaults import env_file_has_key
from graphwiki_kb.services.graphrag_query_service import (
    GraphRAGQueryAnswer,
    GraphRAGQueryService,
)
from graphwiki_kb.services.graphrag_status_service import (
    GraphRAGStatus,
    GraphRAGStatusService,
    _timestamp_iso,
    graph_not_ready_message,
    graph_ready_for_query,
    iso_timestamp_after,
)
from graphwiki_kb.services.project_service import ProjectPaths
from graphwiki_kb.services.query_router_service import QueryRouterService


class GraphAskControllerError(RuntimeError):
    """Raised when GraphRAG ask routing or preflight validation fails."""

    pass


class GraphAskControllerService:
    """Routes `kb ask` requests into GraphRAG and applies query preflight checks."""

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
        streaming: bool | None = None,
        verbose: bool = False,
        save: bool = False,
        save_as: str | None = None,
    ) -> GraphRAGQueryAnswer:
        """Answer a question through the selected or auto-routed GraphRAG method."""
        graph_config = self._resolve_graph_config()
        status = self.status_service.status()
        route = self.router_service.route(question, method=method)
        if not graph_ready_for_query(status, method=route.method):
            raise GraphAskControllerError(
                graph_not_ready_message(status, method=route.method)
            )
        self._require_credentials(graph_config)

        staleness = self._check_staleness(status)

        answer = self.query_service.ask(
            question,
            method=route.method,
            community_level=community_level,
            dynamic_community_selection=dynamic_community_selection,
            response_type=response_type,
            streaming=streaming,
            verbose=verbose,
        )
        answer.retriever = "graph"
        answer.planner = route.planner
        answer.route_reason = route.reason
        answer.staleness_warnings = staleness
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
        dot_env = self.paths.graph_dir / "graphrag" / ".env"
        missing_envs: list[str] = []
        for key in dict.fromkeys(
            (graph_config.api_key_env, graph_config.embedding_api_key_env)
        ):
            if os.environ.get(key) or env_file_has_key(dot_env, key):
                continue
            missing_envs.append(key)
        if not missing_envs:
            return
        keys = ", ".join(missing_envs)
        raise GraphAskControllerError(
            f"GraphRAG API key is not configured. Set {keys} "
            "or add it to graph/graphrag/.env before running `kb ask`."
        )

    def _check_staleness(self, status: GraphRAGStatus) -> list[str]:
        """Return human-readable staleness warnings (empty = fresh)."""
        warnings: list[str] = []
        manifest_mtime = self._manifest_mtime()
        if iso_timestamp_after(manifest_mtime, status.input_updated_at):
            warnings.append("Manifest is newer than graph input. Run `kb update`.")
        if iso_timestamp_after(status.input_updated_at, status.output_updated_at):
            warnings.append("Graph input is newer than index output. Run `kb update`.")
        return warnings

    def _manifest_mtime(self) -> str | None:
        path = self.paths.raw_manifest_file
        if not path.exists():
            return None
        try:
            return _timestamp_iso(path.stat().st_mtime)
        except OSError:
            return None


def _assess_claim_support(answer: GraphRAGQueryAnswer, staleness: list[str]) -> str:
    """Return a conservative support level from parsed citations and trace data."""
    if staleness:
        return "stale-index"
    if not answer.answer or not answer.answer.strip():
        return "no-answer"
    if "[Data:" in answer.answer:
        return "cited-graph-answer"
    source_trace = answer.source_trace or {}
    has_index = bool(source_trace.get("index_run_id"))
    has_hash = bool(source_trace.get("input_manifest_hash"))
    if has_index and has_hash:
        return "graph-index-answer"
    return "unverified"
