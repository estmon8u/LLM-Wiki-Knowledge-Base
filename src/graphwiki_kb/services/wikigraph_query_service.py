"""High-level WikiGraphRAG query and answer service."""

from __future__ import annotations

from typing import Any

from graphwiki_kb.providers.base import TextProvider
from graphwiki_kb.services.config_service import resolve_wikigraph_config
from graphwiki_kb.services.project_service import ProjectPaths
from graphwiki_kb.wikigraph.answer_service import WikiGraphAnswerService
from graphwiki_kb.wikigraph.deps import require_networkx
from graphwiki_kb.wikigraph.index_builder import WikiGraphBuildResult
from graphwiki_kb.wikigraph.models import WikiGraphAnswer
from graphwiki_kb.wikigraph.query_service import WikiGraphMethod, WikiGraphQueryService
from graphwiki_kb.wikigraph.status_service import wikigraph_status


class WikiGraphQueryFacade:
    """Facade used by CLI commands for find/ask operations."""

    def __init__(
        self,
        paths: ProjectPaths,
        config: dict[str, Any],
        *,
        provider: TextProvider | None = None,
    ) -> None:
        self.paths = paths
        self.config = config
        self.provider = provider

    def _require_build(self) -> WikiGraphBuildResult:
        from graphwiki_kb.wikigraph.index_builder import load_built_index

        build = load_built_index(self.paths)
        if build is None:
            raise FileNotFoundError(
                "WikiGraphRAG index is not built. Run `kb update` first."
            )
        return build

    def find(
        self,
        query: str,
        *,
        method: WikiGraphMethod = "auto",
        limit: int = 8,
    ) -> dict[str, Any]:
        require_networkx()
        runtime = resolve_wikigraph_config(self.config)
        build = self._require_build()
        query_service = WikiGraphQueryService(build, runtime)
        contexts, trace, warnings = query_service.retrieve(query, method=method)
        matched_entities = [
            {
                "title": context.title,
                "node_id": context.node_id,
                "score": context.score,
            }
            for context in contexts
            if context.node_kind == "entity"
        ][:limit]
        return {
            "engine": "wikigraph",
            "query": query,
            "method": next(
                (item["value"] for item in trace if item.get("step") == "method"),
                method,
            ),
            "status": wikigraph_status(self.paths).to_dict(),
            "matched_entities": matched_entities,
            "contexts": [context.model_dump() for context in contexts[:limit]],
            "trace": trace,
            "warnings": warnings,
        }

    def ask(
        self,
        question: str,
        *,
        method: WikiGraphMethod = "auto",
        save: bool = False,
    ) -> WikiGraphAnswer:
        require_networkx()
        runtime = resolve_wikigraph_config(self.config)
        build = self._require_build()
        query_service = WikiGraphQueryService(build, runtime)
        answer_service = WikiGraphAnswerService(
            self.paths,
            query_service,
            provider=self.provider,
        )
        return answer_service.answer(question, method=method, save=save)
