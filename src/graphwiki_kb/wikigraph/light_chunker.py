"""Token-aware chunking of normalized source text for the LightRAG backend.

Unlike the classic char-based ``source_text_units`` chunker, this module
produces ~``chunk_token_size``-token :class:`LightChunk`s with token overlap.
It reads ``RawSourceRecord.normalized_path`` only (never the raw PDF/HTML), uses
manifest source IDs as stable document IDs, and emits deterministic chunk IDs so
re-running a build over unchanged input yields byte-identical artifacts.

Chunk *text* is always sliced verbatim from the original normalized document so
BM25/answer evidence stays faithful regardless of the tokenizer backend; the
tokenizer is used only to size chunks (and to count tokens per word for the
greedy windowing).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.wikigraph.light_models import LightChunk
from graphwiki_kb.wikigraph.light_tokenizer import Tokenizer, get_default_tokenizer

_WORD_PATTERN = re.compile(r"\S+")


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_text(text: str) -> str:
    """Collapse trailing per-line whitespace and trim outer whitespace."""
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def build_light_chunks(
    root: Path,
    sources: list[RawSourceRecord],
    *,
    tokenizer: Tokenizer | None = None,
    chunk_token_size: int = 1200,
    overlap_tokens: int = 100,
) -> list[LightChunk]:
    """Return token-aware chunks for every source with normalized text.

    Args:
        root: Project root used to resolve relative ``normalized_path`` values
            and to detect a compiled wiki source page.
        sources: Manifest records. Sources without a normalized artifact (or
            with an unreadable/empty one) are skipped.
        tokenizer: Tokenizer used for sizing. Defaults to
            :func:`get_default_tokenizer` (tiktoken, else regex fallback).
        chunk_token_size: Target tokens per chunk (must be > 0).
        overlap_tokens: Approximate token overlap between adjacent chunks.

    Returns:
        A deterministic list of :class:`LightChunk` in source/chunk order.
    """
    if chunk_token_size <= 0:
        raise ValueError("chunk_token_size must be positive")
    overlap_tokens = max(0, min(overlap_tokens, chunk_token_size - 1))
    tok = tokenizer or get_default_tokenizer()

    chunks: list[LightChunk] = []
    for source in sources:
        if not source.normalized_path:
            continue
        path = root / source.normalized_path
        if not path.exists() or not path.is_file():
            continue
        try:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = _normalize_text(raw_text)
        if not text:
            continue
        compiled_page_path = _compiled_page_path(root, source)
        chunks.extend(
            _chunk_one_source(
                source,
                text,
                tokenizer=tok,
                chunk_token_size=chunk_token_size,
                overlap_tokens=overlap_tokens,
                compiled_page_path=compiled_page_path,
            )
        )
    return chunks


def _compiled_page_path(root: Path, source: RawSourceRecord) -> str | None:
    """Return ``wiki/sources/<slug>.md`` when that compiled page exists."""
    rel = f"wiki/sources/{source.slug}.md"
    if (root / rel).exists():
        return rel
    return None


def _chunk_one_source(
    source: RawSourceRecord,
    text: str,
    *,
    tokenizer: Tokenizer,
    chunk_token_size: int,
    overlap_tokens: int,
    compiled_page_path: str | None,
) -> list[LightChunk]:
    word_spans = [(m.start(), m.end()) for m in _WORD_PATTERN.finditer(text)]
    if not word_spans:
        return []
    # Per-word token counts (leading space approximates BPE merges).
    word_tokens = [tokenizer.count(" " + text[s:e]) for (s, e) in word_spans]

    out: list[LightChunk] = []
    n = len(word_spans)
    i = 0
    chunk_index = 0
    while i < n:
        j = i
        tok_total = 0
        while j < n and (j == i or tok_total + word_tokens[j] <= chunk_token_size):
            tok_total += word_tokens[j]
            j += 1
        start_char = word_spans[i][0]
        end_char = word_spans[j - 1][1]
        chunk_text = text[start_char:end_char]
        out.append(
            LightChunk(
                id=f"chunk:{source.source_id}:{chunk_index}:{_short_hash(chunk_text)}",
                source_id=source.source_id,
                source_slug=source.slug,
                normalized_path=source.normalized_path or "",
                compiled_page_path=compiled_page_path,
                chunk_index=chunk_index,
                token_count=tokenizer.count(chunk_text),
                text=chunk_text,
                content_hash=_content_hash(chunk_text),
                start_char=start_char,
                end_char=end_char,
                metadata={
                    "title": source.title,
                    "source_hash": source.content_hash,
                    "source_type": source.source_type,
                },
            )
        )
        chunk_index += 1
        if j >= n:
            break
        i = _next_start(i, j, word_tokens, overlap_tokens)
    return out


def _next_start(i: int, j: int, word_tokens: list[int], overlap_tokens: int) -> int:
    """Return the next window start, applying token overlap with progress."""
    if overlap_tokens <= 0:
        return j
    back = 0
    k = j
    while k > i + 1 and back < overlap_tokens:
        k -= 1
        back += word_tokens[k]
    return max(i + 1, k)
