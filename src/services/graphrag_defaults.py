"""Shared GraphRAG configuration defaults and utilities."""

from __future__ import annotations

from pathlib import Path

DEFAULT_GRAPHRAG_MODEL = "gpt-4.1-mini"
DEFAULT_GRAPHRAG_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_GRAPHRAG_PROVIDER = "openai"
DEFAULT_GRAPHRAG_API_KEY_ENV = "OPENAI_API_KEY"
LEGACY_GRAPHRAG_API_KEY_ENV = "GRAPHRAG_API_KEY"


def env_file_has_key(path: Path, key: str) -> bool:
    """Check whether a dotenv-style file defines a non-empty value for *key*."""
    if not path.exists():
        return False
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        if name.strip() == key and value.strip().strip('"').strip("'"):
            return True
    return False
