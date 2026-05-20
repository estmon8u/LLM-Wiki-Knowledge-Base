"""Tests for the source-derived TextUnit layer in WikiGraphRAG.

Covers:
* the deterministic ``source_text_units`` chunker (Phase 3),
* ``build_wikigraph_index`` materializing ``source_document`` +
  ``text_unit`` nodes (Phase 6/8),
* curated entities receiving ``text_unit -> entity`` mention edges
  (Phase 7),
* the citation_ref shape distinguishing ``#chunk-N`` from
  ``#text-unit-N`` (Phase 11),
* the configuration toggle disabling normalized TextUnits cleanly
  (Phase 16), and
* the headline fairness test: a phrase that exists **only** in the
  normalized source body must be retrievable through WikiGraphRAG.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.services.config_service import (
    DEFAULT_CONFIG,
    resolve_wikigraph_config,
)
from graphwiki_kb.services.manifest_service import ManifestService
from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService
from graphwiki_kb.services.wikigraph_query_service import WikiGraphQueryService
from graphwiki_kb.wikigraph.index_builder import (
    BuildOptions,
    build_wikigraph_index,
    source_document_node_id,
    text_unit_node_id,
)
from graphwiki_kb.wikigraph.models import EVIDENCE_NODE_KINDS, WikiGraphRetrievedContext
from graphwiki_kb.wikigraph.source_text_units import (
    SourceTextUnit,
    _chunk_text,
    build_source_text_units,
)

REALM_WIKI_PAGE = textwrap.dedent(
    """\
    ---
    title: REALM
    type: source
    source_id: src_realm
    aliases: ['Retrieval-Augmented Language Model']
    summary: REALM pretrains a retriever with masked language modeling.
    ---

    # REALM

    ## Summary

    REALM is a retrieval-augmented language model.

    ## Methods

    REALM backpropagates through retrieval. See [[RAG]].
    """
)

RARE_PHRASE = "zygomorphic-retrieval-sentinel"

REALM_NORMALIZED_BODY = textwrap.dedent(
    f"""\
    # REALM (full normalized body)

    The training signal is masked language modeling perplexity. The retriever
    is trained jointly with the language model and Maximum Inner Product
    Search keeps inference tractable.

    The rare phrase **{RARE_PHRASE}** intentionally appears only in this
    normalized body and never in the curated wiki source page. WikiGraphRAG
    must be able to surface it via the source-derived TextUnit layer.

    {"Filler retrieval paragraph. " * 200}
    """
)


# --------------------------------------------------------------------------- #
# Phase 3 -- deterministic chunker                                            #
# --------------------------------------------------------------------------- #


def test_chunk_text_returns_overlapping_chunks() -> None:
    text = "A" * 5000
    chunks = _chunk_text(text, char_limit=2000, overlap=400)
    assert len(chunks) >= 2
    # Each chunk respects the char_limit budget plus a small slack from
    # paragraph-boundary rounding.
    assert all(len(chunk.text) <= 2500 for chunk in chunks)
    # Adjacent chunks must actually overlap by ~`overlap` chars.
    assert chunks[1].start_char < chunks[0].end_char


def test_chunk_text_handles_empty_input() -> None:
    assert _chunk_text("", char_limit=1000, overlap=100) == []


def _make_source(
    tmp_path: Path,
    *,
    normalized_body: str | None = REALM_NORMALIZED_BODY,
    normalized_path: str | None = "raw/normalized/realm.md",
    raw_path: str = "raw/sources/realm.pdf",
) -> tuple[ManifestService, RawSourceRecord]:
    from graphwiki_kb.services.config_service import ConfigService
    from graphwiki_kb.services.project_service import (
        ProjectService,
        build_project_paths,
    )

    paths = build_project_paths(tmp_path)
    ProjectService(paths).ensure_structure()
    ConfigService(paths).ensure_files()
    manifest = ManifestService(paths)
    manifest.ensure_manifest()

    (paths.wiki_sources_dir / "realm.md").write_text(REALM_WIKI_PAGE)
    if normalized_path and normalized_body is not None:
        (tmp_path / normalized_path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / normalized_path).write_text(normalized_body, encoding="utf-8")

    record = RawSourceRecord(
        source_id="src_realm",
        slug="realm",
        title="REALM",
        origin="/tmp/realm.pdf",
        source_type="pdf",
        raw_path=raw_path,
        normalized_path=normalized_path,
        content_hash="abc",
        ingested_at="2026-01-01T00:00:00Z",
    )
    manifest.save_source(record)
    return manifest, record


def test_build_source_text_units_uses_normalized_path(tmp_path: Path) -> None:
    manifest, record = _make_source(tmp_path)
    units = build_source_text_units(
        root=tmp_path,
        sources=manifest.list_sources(),
        char_limit=2000,
        overlap_chars=200,
        min_chars=50,
    )
    assert units
    assert all(isinstance(unit, SourceTextUnit) for unit in units)
    assert all(unit.source_id == record.source_id for unit in units)
    assert any(RARE_PHRASE in unit.text for unit in units)


def test_build_source_text_units_has_stable_ids(tmp_path: Path) -> None:
    manifest, _record = _make_source(tmp_path)
    units = build_source_text_units(
        root=tmp_path,
        sources=manifest.list_sources(),
        char_limit=2000,
        overlap_chars=200,
        min_chars=50,
    )
    ids_first = [text_unit_node_id(unit) for unit in units]
    units_second = build_source_text_units(
        root=tmp_path,
        sources=manifest.list_sources(),
        char_limit=2000,
        overlap_chars=200,
        min_chars=50,
    )
    ids_second = [text_unit_node_id(unit) for unit in units_second]
    assert ids_first == ids_second


def test_build_source_text_units_skips_binary_raw_without_normalized_path(
    tmp_path: Path,
) -> None:
    manifest, _ = _make_source(
        tmp_path,
        normalized_body=None,
        normalized_path=None,
        raw_path="raw/sources/realm.pdf",
    )
    units = build_source_text_units(
        root=tmp_path,
        sources=manifest.list_sources(),
        char_limit=2000,
        overlap_chars=200,
        min_chars=50,
        source_mode="normalized_only",
    )
    assert units == []


# --------------------------------------------------------------------------- #
# Phase 6/8 -- build flow integrates the TextUnit layer                       #
# --------------------------------------------------------------------------- #


def _build_index(tmp_path: Path, **overrides):
    manifest, record = _make_source(tmp_path)
    defaults = {
        "include_normalized_text_units": True,
        "text_unit_char_limit": 2000,
        "text_unit_overlap_chars": 200,
        "text_unit_min_chars": 50,
    }
    defaults.update(overrides)
    options = BuildOptions(**defaults)
    from graphwiki_kb.services.project_service import build_project_paths

    paths = build_project_paths(tmp_path)
    return (
        build_wikigraph_index(paths, sources=manifest.list_sources(), options=options),
        record,
    )


def test_build_wikigraph_index_adds_source_document_and_text_unit_nodes(
    tmp_path: Path,
) -> None:
    index, record = _build_index(tmp_path)
    assert any(node.kind == "source_document" for node in index.nodes)
    assert any(node.kind == "text_unit" for node in index.nodes)
    assert index.document_count >= 1
    assert index.text_unit_count >= 1
    assert index.include_normalized_text_units is True
    document = next(node for node in index.nodes if node.kind == "source_document")
    assert document.id == source_document_node_id(record)


def test_build_wikigraph_index_links_source_page_to_document(tmp_path: Path) -> None:
    index, _record = _build_index(tmp_path)
    document_ids = {node.id for node in index.nodes if node.kind == "source_document"}
    derived_edges = [
        edge
        for edge in index.edges
        if edge.kind == "derived_from" and edge.target in document_ids
    ]
    assert derived_edges, "source_page should link to its source_document"


def test_build_wikigraph_index_links_document_to_text_units(tmp_path: Path) -> None:
    index, _record = _build_index(tmp_path)
    contains_edges = [
        edge
        for edge in index.edges
        if edge.kind == "contains" and edge.target.startswith("textunit::")
    ]
    assert contains_edges
    assert all(edge.source.startswith("document::") for edge in contains_edges)


def test_build_wikigraph_index_text_units_mention_curated_entities(
    tmp_path: Path,
) -> None:
    index, _record = _build_index(tmp_path)
    mention_edges = [
        edge
        for edge in index.edges
        if edge.kind == "mentions" and edge.source.startswith("textunit::")
    ]
    assert mention_edges
    # All mention targets must be curated entity nodes from the wiki page.
    entity_ids = {node.id for node in index.nodes if node.kind == "entity"}
    assert all(edge.target in entity_ids for edge in mention_edges)


def test_build_wikigraph_index_disabled_preserves_wiki_only_behavior(
    tmp_path: Path,
) -> None:
    index, _record = _build_index(tmp_path, include_normalized_text_units=False)
    assert all(node.kind != "text_unit" for node in index.nodes)
    assert all(node.kind != "source_document" for node in index.nodes)
    assert index.document_count == 0
    assert index.text_unit_count == 0
    assert index.include_normalized_text_units is False


def test_community_detection_excludes_text_unit_and_document_layer(
    tmp_path: Path,
) -> None:
    index, _record = _build_index(tmp_path)
    # No community member should be a TextUnit or a source_document; the
    # projection runs over entities/pages/claims/etc. only.
    for community in index.communities:
        for member_id in community.members:
            assert not member_id.startswith("textunit::")
            assert not member_id.startswith("document::")


# --------------------------------------------------------------------------- #
# Phase 11 -- citation refs                                                   #
# --------------------------------------------------------------------------- #


def test_text_unit_citation_ref_uses_text_unit_anchor() -> None:
    ctx = WikiGraphRetrievedContext(
        node_id="textunit::src_x#0003",
        node_kind="text_unit",
        title="Foo TextUnit",
        path="raw/normalized/foo.md",
        text="body",
        score=1.0,
        chunk_index=3,
        metadata={"unit_index": 3},
    )
    assert ctx.citation_ref == "raw/normalized/foo.md#text-unit-3"


def test_chunk_citation_ref_unchanged() -> None:
    ctx = WikiGraphRetrievedContext(
        node_id="chunk::wiki/sources/realm.md#chunk-0",
        node_kind="chunk",
        title="Summary",
        path="wiki/sources/realm.md",
        text="body",
        score=1.0,
        chunk_index=0,
        metadata={},
    )
    assert ctx.citation_ref == "wiki/sources/realm.md#chunk-0"


def test_evidence_node_kinds_constant_includes_text_unit() -> None:
    assert "text_unit" in EVIDENCE_NODE_KINDS
    assert "chunk" in EVIDENCE_NODE_KINDS
    assert "claim" in EVIDENCE_NODE_KINDS


# --------------------------------------------------------------------------- #
# Phase 10 -- retrieval surfaces source-only phrases                          #
# --------------------------------------------------------------------------- #


def test_query_finds_phrase_only_present_in_normalized_body(tmp_path: Path) -> None:
    """The headline fairness test: rare phrases in raw/normalized must be reachable."""
    from graphwiki_kb.services.config_service import ConfigService
    from graphwiki_kb.services.project_service import build_project_paths

    manifest, _record = _make_source(tmp_path)
    paths = build_project_paths(tmp_path)
    config = ConfigService(paths).load()
    index_service = WikiGraphIndexService(
        paths=paths, config=config, manifest_service=manifest
    )
    index_service.build()

    query_service = WikiGraphQueryService(
        paths=paths, index_service=index_service, config=config, provider=None
    )
    result = query_service.find(RARE_PHRASE, method="basic")
    assert result.contexts
    assert any(ctx.node_kind == "text_unit" for ctx in result.contexts)
    text_units = [ctx for ctx in result.contexts if ctx.node_kind == "text_unit"]
    assert any(RARE_PHRASE in ctx.text for ctx in text_units)
    assert any(
        ctx.citation_ref.endswith("#text-unit-0") or "#text-unit-" in ctx.citation_ref
        for ctx in text_units
    )


# --------------------------------------------------------------------------- #
# Phase 2 -- config validation                                                #
# --------------------------------------------------------------------------- #


def test_default_config_enables_normalized_text_units() -> None:
    runtime = resolve_wikigraph_config(DEFAULT_CONFIG)
    assert runtime.include_normalized_text_units is True
    assert runtime.text_unit_char_limit == 4800
    assert runtime.text_unit_source == "normalized_only"
    assert runtime.text_unit_entity_mode == "mentions_existing_entities"


def test_config_rejects_out_of_range_text_unit_char_limit() -> None:
    config = {
        **DEFAULT_CONFIG,
        "wikigraph": {
            **DEFAULT_CONFIG["wikigraph"],
            "text_unit_char_limit": 999_999,
        },
    }
    with pytest.raises(ValueError):
        resolve_wikigraph_config(config)


def test_index_service_cli_override_wins_over_config(tmp_path: Path) -> None:
    """Phase 16: --no-wikigraph-normalized-text must defeat config=true."""
    from graphwiki_kb.services.config_service import ConfigService
    from graphwiki_kb.services.project_service import build_project_paths

    manifest, _record = _make_source(tmp_path)
    paths = build_project_paths(tmp_path)
    config = ConfigService(paths).load()
    # Config still asks for TextUnits...
    assert resolve_wikigraph_config(config).include_normalized_text_units is True
    service = WikiGraphIndexService(
        paths=paths, config=config, manifest_service=manifest
    )
    # ...but the explicit CLI override (False) wins.
    report = service.build(include_normalized_text_units=False)
    assert report.include_normalized_text_units is False
    assert report.text_unit_count == 0


# --------------------------------------------------------------------------- #
# Phase 13 -- persisted artifacts                                             #
# --------------------------------------------------------------------------- #


def test_text_units_file_persisted_separately(tmp_path: Path) -> None:
    from graphwiki_kb.services.config_service import ConfigService
    from graphwiki_kb.services.project_service import build_project_paths

    manifest, _record = _make_source(tmp_path)
    paths = build_project_paths(tmp_path)
    config = ConfigService(paths).load()
    service = WikiGraphIndexService(
        paths=paths, config=config, manifest_service=manifest
    )
    report = service.build()
    assert report.text_unit_count >= 1
    text_units_file = paths.graph_dir / "wikigraph" / "text_units.json"
    documents_file = paths.graph_dir / "wikigraph" / "documents.json"
    assert text_units_file.exists()
    assert documents_file.exists()
    assert text_units_file.read_text(encoding="utf-8").strip().startswith("[")
