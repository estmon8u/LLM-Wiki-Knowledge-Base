"""Embedding-provider abstraction for the LightRAG entity/relation index.

Text-generation providers (``TextProvider``) and embedding providers are kept
separate because not every completion provider exposes embeddings (Anthropic,
for example). The LightRAG backend depends on vector matching over entity and
relation profiles, so it asks :func:`graphwiki_kb.services.embedding_service`
for an :class:`EmbeddingProvider` and only falls back to BM25 when none is
configured.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable


class EmbeddingError(RuntimeError):
    """Base error for embedding-provider configuration/execution failures."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when an embedding provider is required but not configured."""


class EmbeddingExecutionError(EmbeddingError):
    """Raised when a configured embedding provider fails during embedding."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal embedding-provider interface used by the LightRAG backend."""

    name: str
    model_name: str
    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text (order preserved)."""


class UnavailableEmbeddingProvider:
    """An embedding provider placeholder that always fails with a clear message."""

    def __init__(self, message: str, *, provider_name: str = "unavailable") -> None:
        self.name = provider_name
        self.model_name = ""
        self.dimension = 0
        self._message = message

    def ensure_available(self) -> None:
        """Raise the configured unavailability message."""
        raise EmbeddingConfigurationError(self._message)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Always raise: this provider is not available."""
        raise EmbeddingConfigurationError(self._message)


class LazyEmbeddingProvider:
    """Build a concrete embedding provider only when embeddings are needed.

    ``model_name`` and ``dimension`` are known from config up front so callers
    (status, build manifests) can describe the index without importing SDKs or
    requiring API keys until a real embedding call happens.
    """

    def __init__(
        self,
        factory: Callable[[], EmbeddingProvider],
        *,
        provider_name: str,
        model_name: str,
        dimension: int,
    ) -> None:
        self._factory = factory
        self._provider: EmbeddingProvider | None = None
        self.name = provider_name
        self.model_name = model_name
        self.dimension = dimension

    def _resolve(self) -> EmbeddingProvider:
        if self._provider is None:
            provider = self._factory()
            self._provider = provider
            self.name = provider.name
            self.model_name = provider.model_name or self.model_name
            if provider.dimension:
                self.dimension = provider.dimension
        return self._provider

    def ensure_available(self) -> None:
        """Resolve the underlying provider, surfacing config errors eagerly."""
        provider = self._resolve()
        ensure = getattr(provider, "ensure_available", None)
        if callable(ensure):
            ensure()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` via the lazily-resolved underlying provider."""
        return self._resolve().embed_texts(texts)
