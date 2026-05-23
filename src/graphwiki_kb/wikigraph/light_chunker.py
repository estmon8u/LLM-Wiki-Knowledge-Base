"""Build token-aware LightChunks from normalized source artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.services.project_service import slugify
from graphwiki_kb.wikigraph.light_models import LightChunk
from graphwiki_kb.wikigraph.light_tokenizer import Tokenizer, chunk_text_by_tokens


def short_hash(text: str, *, length: int = 12) -> str:
    """Return a short hex digest prefix for stable IDs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def build_light_chunks(
    root: Path,
    sources: list[RawSourceRecord],
    *,
    tokenizer: Tokenizer,
    chunk_token_size: int,
    overlap_tokens: int,
) -> list[LightChunk]:
    """Build deterministic LightChunks from normalized source text."""
    chunks: list[LightChunk] = []
    for source in sources:
        relative = (source.normalized_path or "").strip()
        if not relative:
            continue
        path = root / relative
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = _normalize_text(text)
        if not text.strip():
            continue
        source_slug = slugify(source.slug or source.source_id)
        compiled = f"wiki/sources/{source_slug}.md"
        for chunk_index, (chunk_text, start_char, end_char) in enumerate(
            chunk_text_by_tokens(
                text,
                tokenizer=tokenizer,
                chunk_token_size=chunk_token_size,
                overlap_tokens=overlap_tokens,
            )
        ):
            content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
            chunk_id = (
                f"chunk:{source.source_id}:{chunk_index}:{short_hash(chunk_text)}"
            )
            chunks.append(
                LightChunk(
                    id=chunk_id,
                    source_id=source.source_id,
                    source_slug=source_slug,
                    normalized_path=relative,
                    compiled_page_path=compiled,
                    chunk_index=chunk_index,
                    token_count=tokenizer.count(chunk_text),
                    text=chunk_text,
                    content_hash=content_hash,
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        "source_hash": source.content_hash,
                        "source_title": source.title,
                    },
                )
            )
    return chunks


def chunk_citation_ref(chunk: LightChunk) -> str:
    """Return wiki citation anchor for a LightChunk."""
    if chunk.compiled_page_path:
        return f"{chunk.compiled_page_path}#chunk-{chunk.chunk_index}"
    return f"{chunk.normalized_path}#text-unit-{chunk.chunk_index}"


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()
