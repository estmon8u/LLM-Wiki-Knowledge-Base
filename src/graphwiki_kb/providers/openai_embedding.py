"""OpenAI embedding provider for the LightRAG entity/relation vector index."""

from __future__ import annotations

import os

from openai import OpenAI

from graphwiki_kb.providers.embedding_base import EmbeddingExecutionError
from graphwiki_kb.providers.retry import provider_retry


class OpenAIEmbeddingProvider:
    """Embeddings via the OpenAI ``embeddings.create`` endpoint.

    ``text-embedding-3-*`` models support a ``dimensions`` parameter, so when a
    non-zero ``dimension`` is supplied it is forwarded to pin the output size
    (keeping the vector store consistent with ``embeddings.dimension`` config).
    """

    name = "openai"

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-large",
        api_key_env: str = "OPENAI_API_KEY",
        dimension: int = 0,
    ) -> None:
        self.model_name = model
        self.dimension = int(dimension)
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise ValueError(
                f"Environment variable {api_key_env} is not set. "
                "Set it to your OpenAI API key to use OpenAI embeddings."
            )
        self._client = OpenAI(api_key=api_key)

    @provider_retry()
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per text (order preserved)."""
        if not texts:
            return []
        kwargs: dict[str, object] = {"model": self.model_name, "input": list(texts)}
        if self.dimension > 0:
            kwargs["dimensions"] = self.dimension
        response = self._client.embeddings.create(**kwargs)
        data = sorted(response.data, key=lambda item: item.index)
        vectors = [list(item.embedding) for item in data]
        if len(vectors) != len(texts):
            raise EmbeddingExecutionError(
                "OpenAI returned a different number of embeddings than inputs "
                f"({len(vectors)} != {len(texts)})."
            )
        if vectors and not self.dimension:
            self.dimension = len(vectors[0])
        return vectors
