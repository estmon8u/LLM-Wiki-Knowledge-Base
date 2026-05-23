"""Gemini embedding provider."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class GeminiEmbeddingProvider:
    """Embed texts via Google GenAI embeddings."""

    model_name: str
    dimension: int
    api_key_env: str = "GEMINI_API_KEY"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Missing API key in environment variable {self.api_key_env}"
            )
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai package is required for Gemini embeddings. "
                "Install with: poetry install --extras gemini"
            ) from exc
        client = genai.Client(api_key=api_key)
        vectors: list[list[float]] = []
        for text in texts:
            result = client.models.embed_content(
                model=self.model_name,
                contents=text,
            )
            embedding = list(result.embeddings[0].values)
            vectors.append(embedding)
        if vectors and self.dimension and len(vectors[0]) != self.dimension:
            raise RuntimeError(
                f"Embedding dimension mismatch: expected {self.dimension}, "
                f"got {len(vectors[0])}"
            )
        return vectors
