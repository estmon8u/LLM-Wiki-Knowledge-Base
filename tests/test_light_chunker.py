"""Unit tests for token-aware LightChunk builder."""

from __future__ import annotations

from pathlib import Path

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.wikigraph.light_chunker import build_light_chunks, chunk_citation_ref
from graphwiki_kb.wikigraph.light_tokenizer import WhitespaceTokenizer


def _source(tmp_path: Path) -> RawSourceRecord:
    normalized = tmp_path / "raw" / "normalized" / "realm.md"
    normalized.parent.mkdir(parents=True, exist_ok=True)
    normalized.write_text(
        "REALM trains a retriever with masked language modeling. "
        "RAG combines retrieval with generation.",
        encoding="utf-8",
    )
    return RawSourceRecord(
        source_id="realm",
        slug="realm",
        title="REALM",
        origin="/tmp/realm.pdf",
        source_type="pdf",
        raw_path="raw/sources/realm.pdf",
        normalized_path="raw/normalized/realm.md",
        content_hash="hash-realm",
        ingested_at="2026-01-01T00:00:00Z",
        compiled_from_hash="hash-realm",
    )


def test_build_light_chunks_deterministic_ids(tmp_path: Path) -> None:
    source = _source(tmp_path)
    tokenizer = WhitespaceTokenizer()
    chunks = build_light_chunks(
        tmp_path,
        [source],
        tokenizer=tokenizer,
        chunk_token_size=40,
        overlap_tokens=5,
    )
    assert len(chunks) >= 1
    assert chunks[0].id.startswith("chunk:realm:0:")
    assert chunks[0].source_id == "realm"
    assert chunk_citation_ref(chunks[0]) == "wiki/sources/realm.md#chunk-0"
