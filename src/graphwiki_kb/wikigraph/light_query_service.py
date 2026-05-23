"""LightRAG-style query routing and retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from graphwiki_kb.providers.base import TextProvider
from graphwiki_kb.wikigraph.light_context_builder import (
    LightContextBuilder,
    LightRetrievalConfig,
    bundle_to_contexts,
)
from graphwiki_kb.wikigraph.light_graph_store import (
    LightGraphStore,
    LightGraphStorePaths,
)
from graphwiki_kb.wikigraph.light_keywords import extract_query_keywords
from graphwiki_kb.wikigraph.light_models import (
    LightGraphFindResult,
    LightGraphIndex,
    LightQueryMethod,
    LightRetrievedBundle,
)
from graphwiki_kb.wikigraph.models import WikiGraphAnswer, WikiGraphFindResult

_GLOBAL_KEYWORDS = (
    "main theme",
    "main themes",
    "overall",
    "across",
    "patterns",
    "landscape",
    "whole corpus",
)
_DRIFT_KEYWORDS = (
    "compare",
    "differ",
    "difference",
    "tradeoff",
    "trade-off",
    " versus ",
    " vs ",
    "contrast",
)


@dataclass
class LightGraphQueryService:
    """Provider-free and provider-assisted LightRAG retrieval."""

    index: LightGraphIndex
    store_paths: LightGraphStorePaths
    provider: TextProvider | None = None
    retrieval_config: LightRetrievalConfig | None = None

    def __post_init__(self) -> None:
        self._builder = LightContextBuilder.from_store(
            self.index,
            self.store_paths,
            config=self.retrieval_config,
        )

    def route_method(
        self, question: str, *, method: LightQueryMethod
    ) -> LightQueryMethod:
        if method != "auto":
            return method
        keywords = extract_query_keywords(
            question, provider=self.provider, entity_catalog=self.index.entities
        )
        normalized = f" {question.casefold()} "
        if any(term in normalized for term in _DRIFT_KEYWORDS):
            return "hybrid"
        if any(term in normalized for term in _GLOBAL_KEYWORDS):
            return "global"
        if keywords.low_level_keywords:
            return "local"
        return "hybrid"

    def find(
        self, question: str, *, method: LightQueryMethod = "auto"
    ) -> LightGraphFindResult:
        chosen = self.route_method(question, method=method)
        keywords = extract_query_keywords(
            question,
            provider=self.provider,
            entity_catalog=self.index.entities,
        )
        bundle = self._builder.retrieve(
            question,
            method=chosen,
            keywords=keywords,
        )
        backend = self._backend_label()
        diagnostics = [f"auto-selected {chosen}"] if method == "auto" else []
        if backend != "lightrag":
            diagnostics.append(backend)
        return LightGraphFindResult(
            query=question,
            method=chosen,
            low_level_keywords=keywords.low_level_keywords,
            high_level_keywords=keywords.high_level_keywords,
            entities=bundle.entities,
            relations=bundle.relations,
            contexts=bundle.contexts,
            trace=bundle.trace,
            diagnostics=diagnostics,
            retrieval_backend=backend,
        )

    def retrieve_bundle(
        self, question: str, *, method: LightQueryMethod = "auto"
    ) -> LightRetrievedBundle:
        chosen = self.route_method(question, method=method)
        keywords = extract_query_keywords(
            question,
            provider=self.provider,
            entity_catalog=self.index.entities,
        )
        bundle = self._builder.retrieve(
            question,
            method=chosen,
            keywords=keywords,
        )
        bundle.retrieval_backend = self._backend_label()
        return bundle

    def to_wikigraph_find(self, result: LightGraphFindResult) -> WikiGraphFindResult:
        """Convert to classic WikiGraphFindResult for CLI compatibility."""
        return WikiGraphFindResult(
            query=result.query,
            method=result.method,  # type: ignore[arg-type]
            contexts=result.contexts,
            entities=[entity.canonical_name for entity in result.entities],
            communities=[],
            trace=result.trace,
            diagnostics=result.diagnostics,
        )

    def _backend_label(self) -> str:
        meta_path = self.store_paths.entity_vectors_dir / "meta.json"
        if not meta_path.exists():
            return "BM25 fallback"
        return "lightrag"


@dataclass
class LightGraphAnswerService:
    """Compose LightRAG bundles into WikiGraph answers."""

    query_service: LightGraphQueryService
    provider: TextProvider | None = None

    def ask(
        self,
        question: str,
        *,
        method: LightQueryMethod = "auto",
        require_provider: bool = False,
    ) -> WikiGraphAnswer:
        bundle = self.query_service.retrieve_bundle(question, method=method)
        contexts = bundle_to_contexts(bundle)
        if require_provider and self.provider is None:
            return WikiGraphAnswer(
                method=bundle.method,  # type: ignore[arg-type]
                question=question,
                answer="",
                contexts=contexts,
                insufficient_evidence=True,
                warnings=["Provider required for LightRAG answer synthesis."],
            )
        if self.provider is None:
            summary = _provider_free_summary(bundle)
            return WikiGraphAnswer(
                method=bundle.method,  # type: ignore[arg-type]
                question=question,
                answer=summary,
                contexts=contexts,
                trace=bundle.trace,
                warnings=[bundle.retrieval_backend],
                insufficient_evidence=not bundle.chunks,
            )
        answer_text = _synthesize_with_provider(
            question, bundle=bundle, provider=self.provider
        )
        return WikiGraphAnswer(
            method=bundle.method,  # type: ignore[arg-type]
            question=question,
            answer=answer_text,
            contexts=contexts,
            trace=bundle.trace,
            insufficient_evidence=not answer_text.strip(),
        )


def _provider_free_summary(bundle: LightRetrievedBundle) -> str:
    lines = [
        f"Retrieved {len(bundle.entities)} entities, "
        f"{len(bundle.relations)} relations, and {len(bundle.chunks)} source chunks.",
    ]
    for chunk in bundle.chunks[:3]:
        lines.append(f"- {chunk.compiled_page_path or chunk.normalized_path}")
    return "\n".join(lines)


def _synthesize_with_provider(
    question: str,
    *,
    bundle: LightRetrievedBundle,
    provider: TextProvider,
) -> str:
    from graphwiki_kb.providers.base import ProviderRequest

    context_text = _format_bundle_for_prompt(bundle)
    prompt = (
        "Use only the retrieved entities, relationships, and source excerpts.\n"
        "Every factual claim must cite one or more source excerpts like [C1].\n"
        "If evidence is insufficient, say so.\n\n"
        f"Question: {question}\n\n"
        f"{context_text}\n"
    )
    response = provider.generate(
        ProviderRequest(
            prompt=prompt, max_tokens=1200, response_schema_name="light_answer"
        )
    )
    return response.text.strip()


def _format_bundle_for_prompt(bundle: LightRetrievedBundle) -> str:
    lines = ["# Retrieved entities"]
    for index, entity in enumerate(bundle.entities[:8], start=1):
        lines.append(f"[E{index}] {entity.canonical_name} — {entity.type}")
        lines.append(entity.description)
    lines.append("\n# Retrieved relationships")
    for index, relation in enumerate(bundle.relations[:8], start=1):
        lines.append(f"[R{index}] {relation.relation_type} ({relation.id})")
        lines.append(relation.description)
    lines.append("\n# Source excerpts")
    for index, chunk in enumerate(bundle.chunks[:8], start=1):
        ref = chunk.compiled_page_path or chunk.normalized_path
        lines.append(f"[C{index}] {ref}#chunk-{chunk.chunk_index}")
        lines.append(chunk.text[:600])
    return "\n".join(lines)


def load_light_index(store_paths: LightGraphStorePaths) -> LightGraphIndex | None:
    """Load a persisted LightGraph index."""
    return LightGraphStore(store_paths).load_or_none()
