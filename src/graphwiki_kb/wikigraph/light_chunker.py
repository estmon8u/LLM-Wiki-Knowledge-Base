"""Token-aware chunker for the LightRAG-style WikiGraphRAG backend.

Builds :class:`LightChunk` objects from the project's normalized source
artifacts. Tokens are approximated by whitespace-separated words; this
keeps the chunker provider-free, deterministic, and dependency-light
while staying close to the LightRAG paper's ~1200-token target.

A real ``tiktoken``-backed tokenizer can be plugged in later by
passing a custom ``tokenize`` function, but the default is intentionally
self-contained so tests and the offline ``deterministic`` extractor work
without any optional dependency.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.wikigraph.light_models import LightChunk

Tokenize = Callable[[str], list[str]]

_WORD = re.compile(r"\S+")


def whitespace_tokenize(text: str) -> list[str]:
    """Approximate token list using whitespace-separated runs.

    The LightRAG paper targets ~1200 tokens per chunk. Whitespace runs
    over-count slightly compared with BPE tokens, which is acceptable
    because LightRAG's chunk size is a soft retrieval budget rather than
    a model context constraint.
    """
    return _WORD.findall(text)


@dataclass(frozen=True)
class LightChunkerOptions:
    """Tunable knobs for :func:`build_light_chunks`."""

    chunk_token_size: int = 1200
    overlap_tokens: int = 100
    min_tokens: int = 30


def _short_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _split_chunks(
    text: str,
    *,
    tokenize: Tokenize,
    chunk_token_size: int,
    overlap_tokens: int,
) -> list[tuple[str, int, int]]:
    """Return ``(chunk_text, start_char, end_char)`` windows for ``text``.

    The chunker tokenizes first, then walks a sliding window. The
    ``start_char`` / ``end_char`` anchors are computed by re-locating
    the boundary tokens inside the original text so anchors remain valid
    for citation rendering.
    """
    tokens = tokenize(text)
    if not tokens:
        return []
    chunk_token_size = max(1, chunk_token_size)
    overlap_tokens = max(0, min(overlap_tokens, chunk_token_size - 1))
    step = chunk_token_size - overlap_tokens
    if step <= 0:
        step = chunk_token_size

    char_positions: list[tuple[int, int]] = []
    cursor = 0
    for tok in tokens:
        idx = text.find(tok, cursor)
        if idx < 0:
            idx = cursor
        char_positions.append((idx, idx + len(tok)))
        cursor = idx + len(tok)

    windows: list[tuple[str, int, int]] = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(start + chunk_token_size, n)
        first_char = char_positions[start][0]
        last_char = char_positions[end - 1][1]
        chunk_text = text[first_char:last_char].strip()
        if chunk_text:
            windows.append((chunk_text, first_char, last_char))
        if end >= n:
            break
        start += step
    return windows


def build_light_chunks(
    *,
    root: Path,
    sources: list[RawSourceRecord],
    options: LightChunkerOptions | None = None,
    tokenize: Tokenize | None = None,
    compiled_page_lookup: Callable[[RawSourceRecord], str | None] | None = None,
) -> list[LightChunk]:
    """Return :class:`LightChunk` objects built from normalized source artifacts.

    Args:
        root: Project root used to resolve relative paths.
        sources: Manifest source records. Records without a
            ``normalized_path`` (or whose file is missing) are skipped
            silently; callers can compute a "missing source" report from
            the difference.
        options: Optional chunker knobs. Defaults match the LightRAG
            paper's published evaluation parameters.
        tokenize: Optional tokenizer. Defaults to
            :func:`whitespace_tokenize`.
        compiled_page_lookup: Optional callable mapping a source record
            to its compiled wiki page (e.g. ``wiki/sources/<slug>.md``).
            When provided, the resulting chunks store that path so
            citations can render either ``raw/normalized/...`` or
            ``wiki/sources/...`` anchors.
    """
    opts = options or LightChunkerOptions()
    tokenize_fn = tokenize or whitespace_tokenize

    chunks: list[LightChunk] = []
    for source in sources:
        if not source.normalized_path:
            continue
        path = root / source.normalized_path
        if not path.exists() or not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = _normalize_text(raw)
        if not text:
            continue
        compiled = (
            compiled_page_lookup(source) if compiled_page_lookup is not None else None
        )
        windows = _split_chunks(
            text,
            tokenize=tokenize_fn,
            chunk_token_size=opts.chunk_token_size,
            overlap_tokens=opts.overlap_tokens,
        )
        for index, (chunk_text, start_char, end_char) in enumerate(windows):
            token_count = len(tokenize_fn(chunk_text))
            if token_count < opts.min_tokens and len(windows) > 1:
                continue
            content_hash = _short_hash(chunk_text, length=16)
            chunk_id = f"chunk:{source.source_id}:{index}:{_short_hash(chunk_text)}"
            chunks.append(
                LightChunk(
                    id=chunk_id,
                    source_id=source.source_id,
                    source_slug=source.slug,
                    source_title=source.title,
                    normalized_path=source.normalized_path,
                    compiled_page_path=compiled,
                    chunk_index=index,
                    token_count=token_count,
                    text=chunk_text,
                    content_hash=content_hash,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        "source_hash": source.content_hash,
                        "origin": source.origin,
                        "source_type": source.source_type,
                    },
                )
            )
    return chunks
