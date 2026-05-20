"""Build deterministic source-derived TextUnits for WikiGraphRAG.

This module consumes the project's normalized markdown/text artifacts
(``raw/normalized/*``) and produces stable, provenance-preserving
chunks that the WikiGraphRAG index promotes to ``text_unit`` nodes.

Design constraints (per the implementation plan):

* **Provider-free and deterministic.** The chunker has no LLM
  involvement; identical inputs produce identical TextUnits and IDs.
* **Index-time only.** Source text is read once during ``kb update``;
  retrieval never re-opens the raw files. This mirrors Microsoft
  GraphRAG's "documents are converted to TextUnits at index time" model.
* **Does not duplicate normalization.** WikiGraphRAG consumes
  ``RawSourceRecord.normalized_path`` produced by the ingest pipeline;
  it does **not** read PDFs/HTML/DOCX directly.

The output is a list of :class:`SourceTextUnit` records that
:mod:`graphwiki_kb.wikigraph.index_builder` lifts into
``text_unit`` nodes plus ``contains`` edges from each source-document
node.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from graphwiki_kb.models.source_models import RawSourceRecord

# Suffixes that we accept as "already text" when ``text_unit_source`` is
# set to ``normalized_with_text_raw_fallback``. PDF/DOCX/etc. paths are
# intentionally excluded -- the ingest pipeline owns conversion.
_RAW_TEXT_SUFFIXES: frozenset[str] = frozenset({".md", ".markdown", ".txt", ".rst"})


@dataclass(frozen=True)
class SourceTextUnit:
    """A single source-derived TextUnit chunk with full provenance."""

    source_id: str
    slug: str
    title: str
    origin: str
    source_type: str
    source_hash: str
    normalized_path: str
    raw_path: str
    unit_index: int
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class _Chunk:
    """An internal chunk produced by :func:`_chunk_text`."""

    text: str
    start_char: int
    end_char: int


def build_source_text_units(
    *,
    root: Path,
    sources: list[RawSourceRecord],
    char_limit: int,
    overlap_chars: int,
    min_chars: int,
    source_mode: str = "normalized_only",
) -> list[SourceTextUnit]:
    """Return TextUnits derived from the configured source text for each record.

    Args:
        root: Project root used to resolve relative ``normalized_path`` /
            ``raw_path`` values.
        sources: Manifest records. Empty list → empty output.
        char_limit: Target character size per TextUnit.
        overlap_chars: Character overlap between adjacent units.
        min_chars: TextUnits shorter than this many non-whitespace chars
            after trimming are discarded.
        source_mode: ``"normalized_only"`` (default) reads only
            ``normalized_path``; ``"normalized_with_text_raw_fallback"``
            additionally accepts plain-text raw paths
            (``.md`` / ``.txt`` / ``.rst``).
    """
    units: list[SourceTextUnit] = []
    for source in sources:
        relative_path = _select_source_text_path(source, source_mode)
        if relative_path is None:
            continue
        path = root / relative_path
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        text = _normalize_text(text)
        for index, chunk in enumerate(
            _chunk_text(text, char_limit=char_limit, overlap=overlap_chars)
        ):
            if len(chunk.text.strip()) < min_chars:
                continue
            units.append(
                SourceTextUnit(
                    source_id=source.source_id,
                    slug=source.slug,
                    title=source.title,
                    origin=source.origin,
                    source_type=source.source_type,
                    source_hash=source.content_hash,
                    normalized_path=source.normalized_path or "",
                    raw_path=source.raw_path,
                    unit_index=index,
                    text=chunk.text,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                )
            )
    return units


def _select_source_text_path(source: RawSourceRecord, mode: str) -> str | None:
    """Return the relative path to read for ``source`` (or ``None`` to skip)."""
    if source.normalized_path:
        return source.normalized_path
    if mode != "normalized_with_text_raw_fallback":
        return None
    suffix = Path(source.raw_path).suffix.lower()
    if suffix in _RAW_TEXT_SUFFIXES:
        return source.raw_path
    return None


def _normalize_text(text: str) -> str:
    """Collapse trailing whitespace per line and trim outer whitespace."""
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _chunk_text(text: str, *, char_limit: int, overlap: int) -> list[_Chunk]:
    """Chunk ``text`` into overlapping ranges with paragraph-aware boundaries."""
    if not text:
        return []
    if char_limit <= 0:
        char_limit = 1
    overlap = max(0, min(overlap, char_limit - 1))

    chunks: list[_Chunk] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + char_limit, length)
        if end < length:
            # Prefer a paragraph boundary in the trailing half of the chunk.
            boundary = text.rfind("\n\n", start, end)
            if boundary > start + char_limit // 2:
                end = boundary
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(_Chunk(chunk_text, start, end))
        if end >= length:
            break
        next_start = max(start + 1, end - overlap)
        if next_start <= start:
            # Safety guard against pathological 0-step loops.
            next_start = start + 1
        start = next_start
    return chunks
