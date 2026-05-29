"""Tests for LightRAG models, tokenizer, and the token-aware chunker."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.wikigraph import light_tokenizer
from graphwiki_kb.wikigraph.light_chunker import build_light_chunks
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    LightChunk,
    LightGraphIndex,
    RelationProfile,
)
from graphwiki_kb.wikigraph.light_tokenizer import (
    RegexWordTokenizer,
    TiktokenTokenizer,
    get_default_tokenizer,
)

# --------------------------------------------------------------------------- #
# Models                                                                      #
# --------------------------------------------------------------------------- #


def test_light_chunk_source_ref_prefers_compiled_page() -> None:
    chunk = LightChunk(
        id="chunk:s1:3:abc",
        source_id="s1",
        source_slug="dpr",
        normalized_path="raw/normalized/dpr.md",
        compiled_page_path="wiki/sources/dpr.md",
        chunk_index=3,
        token_count=10,
        text="body",
        content_hash="hash",
    )
    assert chunk.source_ref == "wiki/sources/dpr.md#chunk-3"


def test_light_chunk_source_ref_falls_back_to_normalized() -> None:
    chunk = LightChunk(
        id="chunk:s1:2:abc",
        source_id="s1",
        source_slug="dpr",
        normalized_path="raw/normalized/dpr.md",
        compiled_page_path=None,
        chunk_index=2,
        token_count=10,
        text="body",
        content_hash="hash",
    )
    assert chunk.source_ref == "raw/normalized/dpr.md#text-unit-2"


def test_light_graph_index_count_properties() -> None:
    index = LightGraphIndex(
        chunks=[
            LightChunk(
                id="c1",
                source_id="s1",
                source_slug="s",
                normalized_path="n.md",
                chunk_index=0,
                token_count=1,
                text="x",
                content_hash="h",
            )
        ],
        entities=[EntityProfile(id="entity:a", canonical_name="A", type="MODEL")],
        relations=[
            RelationProfile(
                id="rel:a",
                source_entity_id="entity:a",
                target_entity_id="entity:b",
                relation_type="USES",
            )
        ],
    )
    assert index.chunk_count == 1
    assert index.entity_count == 1
    assert index.relation_count == 1


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValueError):
        EntityProfile(id="e", canonical_name="A", type="MODEL", bogus=1)


# --------------------------------------------------------------------------- #
# Tokenizer                                                                   #
# --------------------------------------------------------------------------- #


def test_regex_word_tokenizer_roundtrip_counts() -> None:
    tok = RegexWordTokenizer()
    ids = tok.encode("Dense Passage Retrieval, DPR.")
    assert tok.count("Dense Passage Retrieval, DPR.") == len(ids)
    # Decode is a space-joined approximation but preserves token content.
    decoded = tok.decode(ids)
    assert "Dense" in decoded and "DPR" in decoded


def test_tiktoken_tokenizer_roundtrip() -> None:
    pytest.importorskip("tiktoken")
    tok = TiktokenTokenizer()
    text = "Retrieval-Augmented Generation uses a dense retriever."
    ids = tok.encode(text)
    assert tok.count(text) == len(ids)
    assert tok.decode(ids) == text
    assert tok.name.startswith("tiktoken:")


def test_get_default_tokenizer_falls_back_when_tiktoken_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("tiktoken missing")

    monkeypatch.setattr(light_tokenizer, "TiktokenTokenizer", _boom)
    tok = get_default_tokenizer()
    assert isinstance(tok, RegexWordTokenizer)


# --------------------------------------------------------------------------- #
# Chunker                                                                     #
# --------------------------------------------------------------------------- #


def _make_source(
    root: Path,
    *,
    source_id: str = "s1",
    slug: str = "dpr",
    body: str = "",
    compiled: bool = False,
) -> RawSourceRecord:
    normalized_rel = f"raw/normalized/{slug}.md"
    norm_path = root / normalized_rel
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    norm_path.write_text(body, encoding="utf-8")
    if compiled:
        wiki_path = root / "wiki" / "sources" / f"{slug}.md"
        wiki_path.parent.mkdir(parents=True, exist_ok=True)
        wiki_path.write_text("# compiled\n", encoding="utf-8")
    return RawSourceRecord(
        source_id=source_id,
        slug=slug,
        title=slug.upper(),
        origin="upload",
        source_type="pdf",
        raw_path=f"raw/sources/{slug}.pdf",
        content_hash=f"hash-{source_id}",
        ingested_at="2026-01-01T00:00:00Z",
        normalized_path=normalized_rel,
    )


def test_chunker_empty_corpus(tmp_path: Path) -> None:
    assert build_light_chunks(tmp_path, []) == []


def test_chunker_skips_missing_and_empty(tmp_path: Path) -> None:
    no_norm = RawSourceRecord(
        source_id="x",
        slug="x",
        title="X",
        origin="upload",
        source_type="pdf",
        raw_path="raw/sources/x.pdf",
        content_hash="h",
        ingested_at="2026-01-01T00:00:00Z",
        normalized_path=None,
    )
    missing = RawSourceRecord(
        source_id="y",
        slug="y",
        title="Y",
        origin="upload",
        source_type="pdf",
        raw_path="raw/sources/y.pdf",
        content_hash="h2",
        ingested_at="2026-01-01T00:00:00Z",
        normalized_path="raw/normalized/does-not-exist.md",
    )
    empty = _make_source(tmp_path, source_id="z", slug="z", body="   \n  \n")
    assert build_light_chunks(tmp_path, [no_norm, missing, empty]) == []


def test_chunker_single_chunk_small_doc(tmp_path: Path) -> None:
    src = _make_source(
        tmp_path, body="Dense Passage Retrieval uses a dual encoder.", compiled=True
    )
    chunks = build_light_chunks(tmp_path, [src], chunk_token_size=1200)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source_id == "s1"
    assert chunk.chunk_index == 0
    assert chunk.compiled_page_path == "wiki/sources/dpr.md"
    assert chunk.token_count > 0
    assert chunk.id.startswith("chunk:s1:0:")
    assert chunk.text.startswith("Dense Passage Retrieval")


def test_chunker_multiple_chunks_with_overlap_and_determinism(tmp_path: Path) -> None:
    words = " ".join(f"word{i}" for i in range(200))
    src = _make_source(tmp_path, body=words)
    chunks_a = build_light_chunks(
        tmp_path,
        [src],
        tokenizer=RegexWordTokenizer(),
        chunk_token_size=40,
        overlap_tokens=10,
    )
    chunks_b = build_light_chunks(
        tmp_path,
        [src],
        tokenizer=RegexWordTokenizer(),
        chunk_token_size=40,
        overlap_tokens=10,
    )
    assert len(chunks_a) > 1
    # Deterministic: identical IDs and text across runs.
    assert [c.id for c in chunks_a] == [c.id for c in chunks_b]
    # Chunk indices are contiguous from zero.
    assert [c.chunk_index for c in chunks_a] == list(range(len(chunks_a)))
    # Overlap: consecutive chunks share some leading/trailing words.
    assert chunks_a[1].start_char < chunks_a[0].end_char


def test_chunker_zero_overlap(tmp_path: Path) -> None:
    words = " ".join(f"tok{i}" for i in range(120))
    src = _make_source(tmp_path, body=words)
    chunks = build_light_chunks(
        tmp_path,
        [src],
        tokenizer=RegexWordTokenizer(),
        chunk_token_size=30,
        overlap_tokens=0,
    )
    assert len(chunks) >= 2
    # No overlap: each chunk starts at or after the previous chunk's end.
    for prev, nxt in itertools.pairwise(chunks):
        assert nxt.start_char >= prev.end_char


def test_chunker_rejects_bad_chunk_size(tmp_path: Path) -> None:
    src = _make_source(tmp_path, body="hello world")
    with pytest.raises(ValueError, match="chunk_token_size"):
        build_light_chunks(tmp_path, [src], chunk_token_size=0)
