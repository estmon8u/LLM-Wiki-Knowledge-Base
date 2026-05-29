"""Google Gemini embedding provider for the LightRAG vector index."""

from __future__ import annotations

import os

from google import genai
from google.genai import types

from graphwiki_kb.providers.embedding_base import EmbeddingExecutionError
from graphwiki_kb.providers.retry import provider_retry


class GeminiEmbeddingProvider:
    """Embeddings via the google-genai ``embed_content`` endpoint."""

    name = "gemini"

    def __init__(
        self,
        *,
        model: str = "text-embedding-004",
        api_key_env: str = "GEMINI_API_KEY",
        dimension: int = 0,
    ) -> None:
        self.model_name = model
        self.dimension = int(dimension)
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise ValueError(
                f"Environment variable {api_key_env} is not set. "
                "Set it to your Gemini API key to use Gemini embeddings."
            )
        self._client = genai.Client(api_key=api_key)

    @provider_retry()
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per text (order preserved)."""
        if not texts:
            return []
        config = None
        if self.dimension > 0:
            config = types.EmbedContentConfig(output_dimensionality=self.dimension)
        response = self._client.models.embed_content(
            model=self.model_name,
            contents=list(texts),
            config=config,
        )
        vectors = [list(embedding.values) for embedding in response.embeddings]
        if len(vectors) != len(texts):
            raise EmbeddingExecutionError(
                "Gemini returned a different number of embeddings than inputs "
                f"({len(vectors)} != {len(texts)})."
            )
        if vectors and not self.dimension:
            self.dimension = len(vectors[0])
        return vectors
