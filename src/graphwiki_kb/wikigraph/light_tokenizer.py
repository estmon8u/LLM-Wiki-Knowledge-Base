"""Lightweight token counting for LightRAG-style chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

_TOKEN_RE = re.compile(r"\S+")


class Tokenizer(Protocol):
    """Protocol for token-aware text chunking."""

    def count(self, text: str) -> int:
        """Return token count for ``text``."""
        ...

    def encode(self, text: str) -> list[str]:
        """Return token strings for ``text``."""
        ...


@dataclass(frozen=True)
class WhitespaceTokenizer:
    """Whitespace tokenizer with ~4 characters per token heuristic."""

    chars_per_token: int = 4

    def encode(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text)

    def count(self, text: str) -> int:
        tokens = self.encode(text)
        if tokens:
            return len(tokens)
        return max(1, (len(text) + self.chars_per_token - 1) // self.chars_per_token)


def chunk_text_by_tokens(
    text: str,
    *,
    tokenizer: Tokenizer,
    chunk_token_size: int,
    overlap_tokens: int,
) -> list[tuple[str, int, int]]:
    """Split ``text`` into overlapping token-bounded chunks.

    Returns tuples of ``(chunk_text, start_char, end_char)``.
    """
    if not text.strip():
        return []
    tokens = tokenizer.encode(text)
    if not tokens:
        return [(text, 0, len(text))]
    if chunk_token_size <= 0:
        return [(text, 0, len(text))]

    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(tokens):
        end = min(len(tokens), cursor + chunk_token_size)
        spans.append((cursor, end))
        if end >= len(tokens):
            break
        cursor = max(cursor + 1, end - max(0, overlap_tokens))

    chunks: list[tuple[str, int, int]] = []
    for start_tok, end_tok in spans:
        prefix = " ".join(tokens[:start_tok])
        chunk_tokens = tokens[start_tok:end_tok]
        chunk_text = " ".join(chunk_tokens)
        if prefix:
            start_char = len(prefix) + (1 if prefix else 0)
        else:
            start_char = 0
        end_char = start_char + len(chunk_text)
        chunks.append((chunk_text, start_char, end_char))
    return chunks
