"""Service layer for WikiGraphRAG retrieval and answer generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphwiki_kb.providers.base import TextProvider
from graphwiki_kb.services.config_service import resolve_wikigraph_config
from graphwiki_kb.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    slugify,
    utc_now_iso,
)
from graphwiki_kb.services.wikigraph_index_service import (
    WikiGraphIndexService,
)
from graphwiki_kb.wikigraph.answer_service import WikiGraphAnswerService
from graphwiki_kb.wikigraph.context_builder import ContextBuilderConfig
from graphwiki_kb.wikigraph.light_context_builder import (
    LightContextBuilderConfig,
)
from graphwiki_kb.wikigraph.light_query_service import (
    LightAnswerService,
    LightGraphQueryEngine,
)
from graphwiki_kb.wikigraph.models import (
    QueryMethod,
    WikiGraphAnswer,
    WikiGraphFindResult,
)
from graphwiki_kb.wikigraph.query_service import WikiGraphQueryEngine


class WikiGraphQueryError(RuntimeError):
    """Raised when the WikiGraphRAG query layer cannot serve a request."""


@dataclass
class WikiGraphQueryService:
    """High-level retrieval/answer surface used by the CLI."""

    paths: ProjectPaths
    index_service: WikiGraphIndexService
    provider: TextProvider | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def _context_builder_config(self) -> ContextBuilderConfig:
        try:
            runtime = resolve_wikigraph_config(self.config or {})
        except ValueError:
            runtime = resolve_wikigraph_config({})
        return ContextBuilderConfig(
            max_context_chunks=runtime.max_context_chunks,
            max_context_tokens=runtime.max_context_tokens,
            max_hops=runtime.max_hops,
            fuzzy_entity_match_threshold=runtime.fuzzy_entity_match_threshold,
            lexical_backend=runtime.lexical_backend,
            retrieval_improvements_enabled=runtime.retrieval_improvements_enabled,
            rrf_k=runtime.rrf_k,
            alias_query_token_budget=runtime.alias_query_token_budget,
            section_title_overlap_boost=runtime.section_title_overlap_boost,
        )

    def _resolved_mode(self) -> str:
        try:
            runtime = resolve_wikigraph_config(self.config or {})
        except ValueError:
            runtime = resolve_wikigraph_config({})
        return runtime.mode

    def _light_context_builder_config(self) -> LightContextBuilderConfig:
        try:
            runtime = resolve_wikigraph_config(self.config or {})
        except ValueError:
            runtime = resolve_wikigraph_config({})
        retrieval = runtime.lightrag.retrieval
        return LightContextBuilderConfig(
            top_k_entities=retrieval.top_k_entities,
            top_k_relations=retrieval.top_k_relations,
            top_k_chunks=retrieval.top_k_chunks,
            max_entity_tokens=retrieval.max_entity_tokens,
            max_relation_tokens=retrieval.max_relation_tokens,
            max_chunk_tokens=retrieval.max_chunk_tokens,
            max_total_tokens=retrieval.max_total_tokens,
            rrf_k=retrieval.rrf_k,
        )

    def _ensure_engine(self) -> WikiGraphQueryEngine:
        index = self.index_service.load()
        if index is None:
            raise WikiGraphQueryError(
                "WikiGraphRAG index is missing. Run `kb update` to build it "
                "(it is enabled by default; pass `--no-wikigraph` to skip)."
            )
        return WikiGraphQueryEngine(
            index=index,
            config=self._context_builder_config(),
        )

    def _ensure_light_engine(self) -> LightGraphQueryEngine:
        light_index = self.index_service.load_lightgraph()
        if light_index is None:
            raise WikiGraphQueryError(
                "WikiGraphRAG LightRAG index is missing. Set "
                "`wikigraph.mode: lightrag` and run `kb update`."
            )
        return LightGraphQueryEngine(
            index=light_index,
            config=self._light_context_builder_config(),
        )

    def find(
        self, question: str, *, method: QueryMethod = "auto"
    ) -> WikiGraphFindResult:
        """Run a provider-free retrieval and return a :class:`WikiGraphFindResult`."""
        if self._resolved_mode() == "lightrag":
            light_engine = self._ensure_light_engine()
            light_method = _classic_method_to_light(method)
            return light_engine.find(question, method=light_method)
        engine = self._ensure_engine()
        return engine.find(question, method=method)

    def ask(
        self,
        question: str,
        *,
        method: QueryMethod = "auto",
        require_provider: bool = False,
        save: bool = False,
        save_as: str | None = None,
    ) -> WikiGraphAnswer:
        """Run a full WikiGraphRAG answer pipeline for ``question``."""
        if self._resolved_mode() == "lightrag":
            light_engine = self._ensure_light_engine()
            light_service = LightAnswerService(
                engine=light_engine, provider=self.provider
            )
            light_method = _classic_method_to_light(method)
            answer = light_service.ask(
                question, method=light_method, require_provider=require_provider
            )
        else:
            engine = self._ensure_engine()
            service = WikiGraphAnswerService(engine=engine, provider=self.provider)
            answer = service.ask(
                question,
                method=method,
                require_provider=require_provider,
            )
        if save or save_as:
            saved_path = self.save_answer(question, answer, slug=save_as)
            answer = answer.model_copy(update={"saved_path": saved_path})
        return answer

    def save_answer(
        self,
        question: str,
        answer: WikiGraphAnswer,
        *,
        slug: str | None = None,
    ) -> str:
        """Persist a WikiGraphRAG answer as a wiki analysis page."""
        if not answer.answer.strip():
            raise WikiGraphQueryError("Refusing to save an empty wikigraph answer.")
        safe_slug = slugify(slug or f"wikigraph-{question}")
        if not safe_slug or safe_slug == "untitled":
            safe_slug = "wikigraph-analysis"
        dest = self.paths.wiki_analysis_dir / f"{safe_slug}.md"
        timestamp = utc_now_iso()
        frontmatter_lines = [
            "---",
            f"title: {json.dumps(question)}",
            "type: analysis",
            "engine: wikigraph",
            f"method: {answer.method}",
            f"saved_at: {timestamp}",
            f"insufficient_evidence: {str(answer.insufficient_evidence).lower()}",
            "---",
            "",
        ]
        body_lines: list[str] = [
            f"# {question}",
            "",
            "## Answer",
            "",
            answer.answer.strip(),
            "",
            "## Contexts",
            "",
        ]
        for ctx in answer.contexts:
            body_lines.append(
                f"- [[{ctx.title}]] (`{ctx.citation_ref}`) score={ctx.score:.3f}"
            )
        body_lines.append("")
        body_lines.append("## Trace")
        body_lines.append("")
        for step in answer.trace:
            body_lines.append(f"- {json.dumps(step, default=str)}")
        contents = "\n".join([*frontmatter_lines, *body_lines]) + "\n"
        atomic_write_text(dest, contents)

        run_path = (
            self.runs_dir / f"query-{_safe_timestamp(timestamp)}-{safe_slug[:24]}.json"
        )
        run_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            run_path, json.dumps(answer.model_dump(), indent=2, default=str)
        )
        return dest.relative_to(self.paths.root).as_posix()

    @property
    def runs_dir(self) -> Path:
        """Directory where saved-query run JSON files are written."""
        return self.paths.graph_dir / "runs" / "wikigraph"


def _safe_timestamp(timestamp: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in timestamp).strip("-") or "unknown"


def classic_method_to_light(method: QueryMethod) -> Any:
    """Public alias for :func:`_classic_method_to_light` (for tests)."""
    return _classic_method_to_light(method)


def _classic_method_to_light(method: QueryMethod) -> Any:
    """Map a classic :class:`QueryMethod` to its LightRAG equivalent.

    The classic ``QueryMethod`` literal doesn't have a ``hybrid`` slot,
    so callers that want LightRAG's hybrid retrieval should pass it as
    ``drift-lite`` (the legacy hybrid analog). Everything else maps
    one-to-one. Method routing remains identical for ``auto``.
    """
    if method == "drift-lite":
        return "hybrid"
    return method
