"""High-level LightRAG query engine (keywords -> retrieval -> bundle).

Wraps :mod:`light_keywords` and :mod:`light_context_builder` and exposes a
``find`` returning the structured :class:`LightRetrievedBundle`, plus a
conversion to the classic :class:`WikiGraphFindResult` so existing CLI/JSON
surfaces keep working.
"""

from __future__ import annotations

from dataclasses import dataclass

from graphwiki_kb.providers.base import TextProvider
from graphwiki_kb.providers.embedding_base import EmbeddingProvider
from graphwiki_kb.services.config_service import LightRagRuntimeConfig
from graphwiki_kb.wikigraph.light_context_builder import LightRetriever
from graphwiki_kb.wikigraph.light_graph_store import LightGraphStore
from graphwiki_kb.wikigraph.light_keywords import extract_query_keywords
from graphwiki_kb.wikigraph.light_models import (
    LightGraphIndex,
    LightQueryMethod,
    LightRetrievedBundle,
)
from graphwiki_kb.wikigraph.light_vector_store import LightVectorStore
from graphwiki_kb.wikigraph.models import QueryMethod, WikiGraphFindResult


@dataclass
class LightQueryEngine:
    """Keyword extraction + dual-level retrieval over a LightRAG index."""

    index: LightGraphIndex
    config: LightRagRuntimeConfig
    entity_vectors: LightVectorStore | None = None
    relation_vectors: LightVectorStore | None = None
    provider: TextProvider | None = None
    embedding_provider: EmbeddingProvider | None = None

    def __post_init__(self) -> None:
        self._retriever = LightRetriever(
            entities=self.index.entities,
            relations=self.index.relations,
            chunks=self.index.chunks,
            config=self.config,
            entity_vectors=self.entity_vectors,
            relation_vectors=self.relation_vectors,
            embedding_provider=self.embedding_provider,
        )
        self._known_aliases = {
            alias
            for entity in self.index.entities
            for alias in [entity.canonical_name, *entity.aliases]
        }

    @classmethod
    def from_store(
        cls,
        store: LightGraphStore,
        *,
        config: LightRagRuntimeConfig,
        provider: TextProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> LightQueryEngine | None:
        """Build an engine from a persisted store, or ``None`` if missing."""
        index = store.load()
        if index is None:
            return None
        return cls(
            index=index,
            config=config,
            entity_vectors=store.load_entity_vectors(),
            relation_vectors=store.load_relation_vectors(),
            provider=provider,
            embedding_provider=embedding_provider,
        )

    @property
    def using_embeddings(self) -> bool:
        """Whether vector retrieval (vs BM25 fallback) is active."""
        return self._retriever.using_embeddings

    def find(
        self,
        question: str,
        *,
        method: LightQueryMethod = "auto",
        keyword_provider: TextProvider | None = None,
    ) -> LightRetrievedBundle:
        """Extract keywords and retrieve a structured bundle for ``question``."""
        provider = keyword_provider if keyword_provider is not None else self.provider
        keywords = extract_query_keywords(
            question, provider=provider, known_aliases=self._known_aliases
        )
        return self._retriever.retrieve(question, keywords, method)

    def find_result(
        self, question: str, *, method: LightQueryMethod = "auto"
    ) -> WikiGraphFindResult:
        """Return a classic :class:`WikiGraphFindResult` for CLI/JSON parity."""
        bundle = self.find(question, method=method)
        method_value: QueryMethod = bundle.method
        return WikiGraphFindResult(
            query=question,
            method=method_value,
            contexts=bundle.contexts,
            entities=[entity.canonical_name for entity in bundle.entities],
            communities=[],
            trace=bundle.trace,
            diagnostics=bundle.diagnostics,
        )
