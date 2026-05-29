"""Factory for the LightRAG embedding-provider layer.

Reads the top-level ``embeddings`` config section and returns a lazy
:class:`EmbeddingProvider`. Anthropic is intentionally not an embedding
provider; an unsupported provider returns ``None`` so the LightRAG index/query
layers fall back to BM25 (and label the run accordingly).
"""

from __future__ import annotations

import logging
from typing import Any

from graphwiki_kb.providers.embedding_base import (
    EmbeddingProvider,
    LazyEmbeddingProvider,
    UnavailableEmbeddingProvider,
)
from graphwiki_kb.services.config_service import resolve_embeddings_config

logger = logging.getLogger(__name__)

SUPPORTED_EMBEDDING_PROVIDERS: tuple[str, ...] = ("openai", "gemini")


def build_embedding_provider(config: dict[str, Any]) -> EmbeddingProvider | None:
    """Build an embedding provider from the ``embeddings`` config section.

    Returns ``None`` when the configured provider does not support embeddings
    (for example ``anthropic``) so the caller can use a local BM25 fallback.
    Construction is lazy: API keys and SDK imports are only touched when an
    embedding call is actually made.
    """
    resolved = resolve_embeddings_config(config)
    provider = resolved.provider

    if provider == "openai":

        def _openai() -> EmbeddingProvider:
            from graphwiki_kb.providers.openai_embedding import (
                OpenAIEmbeddingProvider,
            )

            try:
                return OpenAIEmbeddingProvider(
                    model=resolved.model,
                    api_key_env=resolved.api_key_env,
                    dimension=resolved.dimension,
                )
            except ValueError as exc:
                return UnavailableEmbeddingProvider(str(exc), provider_name="openai")

        return LazyEmbeddingProvider(
            _openai,
            provider_name="openai",
            model_name=resolved.model,
            dimension=resolved.dimension,
        )

    if provider == "gemini":

        def _gemini() -> EmbeddingProvider:
            from graphwiki_kb.providers.gemini_embedding import (
                GeminiEmbeddingProvider,
            )

            try:
                return GeminiEmbeddingProvider(
                    model=resolved.model,
                    api_key_env=resolved.api_key_env,
                    dimension=resolved.dimension,
                )
            except ValueError as exc:
                return UnavailableEmbeddingProvider(str(exc), provider_name="gemini")

        return LazyEmbeddingProvider(
            _gemini,
            provider_name="gemini",
            model_name=resolved.model,
            dimension=resolved.dimension,
        )

    logger.info(
        "Embedding provider %r does not support embeddings; "
        "LightRAG will use the BM25 fallback.",
        provider,
    )
    return None
