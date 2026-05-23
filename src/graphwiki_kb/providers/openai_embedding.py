"""OpenAI embedding provider."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class OpenAIEmbeddingProvider:
    """Embed texts via the OpenAI embeddings API."""

    model_name: str
    dimension: int
    api_key_env: str = "OPENAI_API_KEY"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Missing API key in environment variable {self.api_key_env}"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for OpenAI embeddings. "
                "Install with: poetry install --extras openai"
            ) from exc
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(model=self.model_name, input=texts)
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = [list(item.embedding) for item in ordered]
        if vectors and self.dimension and len(vectors[0]) != self.dimension:
            raise RuntimeError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {len(vectors[0])}"
            )
        return vectors
