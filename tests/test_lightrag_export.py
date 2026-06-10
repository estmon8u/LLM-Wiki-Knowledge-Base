"""Tests for the LightRAG wiki card export service + dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphwiki_kb.services.project_service import build_project_paths
from graphwiki_kb.services.wikigraph_light_export_service import (
    WikiGraphLightExportService,
)
from graphwiki_kb.wikigraph.light_graph_store import (
    LightGraphStore,
    LightGraphStorePaths,
)
from graphwiki_kb.wikigraph.light_models import (
    EntityProfile,
    LightChunk,
    LightGraphIndex,
    RelationProfile,
)


def _index() -> LightGraphIndex:
    chunks = [
        LightChunk(
            id="chunk-0",
            source_id="rag",
            source_slug="rag",
            normalized_path="raw/normalized/rag.md",
            compiled_page_path="wiki/sources/rag.md",
            chunk_index=0,
            token_count=10,
            text="RAG uses Dense Passage Retrieval.",
            content_hash="h0",
            metadata={"title": "RAG"},
        )
    ]
    entities = [
        EntityProfile(
            id="entity:rag",
            canonical_name="Retrieval-Augmented Generation",
            type="MODEL",
            aliases=["RAG"],
            description="A retrieval-augmented generator.",
            chunk_ids=["chunk-0"],
            source_ids=["rag"],
            relation_ids=["relation:rag:uses:dpr"],
            updated_at="t",
        ),
        EntityProfile(
            id="entity:dpr",
            canonical_name="Dense Passage Retrieval",
            type="METHOD",
            chunk_ids=["chunk-0"],
            source_ids=["rag"],
            relation_ids=["relation:rag:uses:dpr"],
            updated_at="t",
        ),
        EntityProfile(
            id="entity:orphan",
            canonical_name="Orphan Concept",
            type="CONCEPT",
            chunk_ids=[],
            updated_at="t",
        ),
    ]
    relations = [
        RelationProfile(
            id="relation:rag:uses:dpr",
            source_entity_id="entity:rag",
            target_entity_id="entity:dpr",
            relation_type="USES",
            keywords=["retrieval"],
            description="RAG uses DPR.",
            chunk_ids=["chunk-0"],
            source_ids=["rag"],
            updated_at="t",
        )
    ]
    return LightGraphIndex(
        built_at="2026-01-01T00:00:00Z",
        chunks=chunks,
        entities=entities,
        relations=relations,
        tier="fallback+bm25",
    )


def _store(tmp_path: Path) -> LightGraphStore:
    store = LightGraphStore(LightGraphStorePaths(tmp_path / "graph/wikigraph/lightrag"))
    store.save(
        _index(),
        build_manifest={
            "missing_sources": [
                {"source_id": "gone", "status": "missing", "requires_review": True}
            ]
        },
    )
    return store


def test_export_writes_all_card_types(tmp_path: Path) -> None:
    paths = build_project_paths(tmp_path)
    store = _store(tmp_path)
    service = WikiGraphLightExportService(paths=paths, store=store)
    written = service.export()

    base = paths.wiki_dir / "wikigraph"
    relation_slug = WikiGraphLightExportService._relation_slug(
        _index().relations[0],
        {
            "entity:rag": "Retrieval-Augmented Generation",
            "entity:dpr": "Dense Passage Retrieval",
        },
    )
    assert (base / "index.md").exists()
    assert (base / "entities" / "retrieval-augmented-generation.md").exists()
    assert (base / "relations" / f"{relation_slug}.md").exists()
    assert (base / "sources" / "rag-chunks.md").exists()
    assert (base / "diagnostics" / "extraction-warnings.md").exists()
    assert (base / "diagnostics" / "stale-sources.md").exists()
    # Returned relative paths are sorted and rooted in the project.
    assert written == sorted(written)
    assert all(rel.startswith("wiki/wikigraph/") for rel in written)


def test_lightrag_relation_slug_caps_long_names() -> None:
    relation = RelationProfile(
        id="relation:long",
        source_entity_id="entity:long-source",
        target_entity_id="entity:long-target",
        relation_type="SUPPORTS",
        updated_at="t",
    )
    name_by_id = {
        "entity:long-source": "A Very Long Source Entity Name " * 8,
        "entity:long-target": "A Very Long Target Entity Name " * 8,
    }

    slug = WikiGraphLightExportService._relation_slug(relation, name_by_id)

    assert len(slug) <= 99
    assert slug.startswith("a-very-long-source-entity-name")
    suffix = slug.rsplit("-", 1)[-1]
    assert len(suffix) == 8
    assert all(char in "0123456789abcdef" for char in suffix)


def test_entity_card_content(tmp_path: Path) -> None:
    paths = build_project_paths(tmp_path)
    store = _store(tmp_path)
    WikiGraphLightExportService(paths=paths, store=store).export()
    card = (
        paths.wiki_dir / "wikigraph" / "entities" / "retrieval-augmented-generation.md"
    ).read_text()
    assert "engine: wikigraph-lightrag" in card
    assert "entity_id: entity:rag" in card
    assert "- RAG" in card  # alias
    assert "wiki/sources/rag.md#chunk-0" in card  # chunk ref
    assert "[[wikigraph/relations/" in card  # relation backlink


def test_relation_card_content(tmp_path: Path) -> None:
    paths = build_project_paths(tmp_path)
    store = _store(tmp_path)
    WikiGraphLightExportService(paths=paths, store=store).export()
    relation_path = next((paths.wiki_dir / "wikigraph" / "relations").glob("*.md"))
    card = relation_path.read_text()
    assert "relation_type: USES" in card
    assert "wiki/sources/rag.md#chunk-0" in card


def test_diagnostics_report_orphans_and_stale(tmp_path: Path) -> None:
    paths = build_project_paths(tmp_path)
    store = _store(tmp_path)
    WikiGraphLightExportService(paths=paths, store=store).export()
    warnings = (
        paths.wiki_dir / "wikigraph" / "diagnostics" / "extraction-warnings.md"
    ).read_text()
    assert "Orphan Concept" in warnings  # entity without chunks
    stale = (
        paths.wiki_dir / "wikigraph" / "diagnostics" / "stale-sources.md"
    ).read_text()
    assert "gone" in stale


def test_export_missing_index_raises(tmp_path: Path) -> None:
    paths = build_project_paths(tmp_path)
    empty_store = LightGraphStore(
        LightGraphStorePaths(tmp_path / "graph/wikigraph/lightrag-empty")
    )
    with pytest.raises(FileNotFoundError):
        WikiGraphLightExportService(paths=paths, store=empty_store).export()


def test_index_service_export_artifacts_routes_to_lightrag(tmp_path: Path) -> None:
    from graphwiki_kb.services.config_service import ConfigService
    from graphwiki_kb.services.manifest_service import ManifestService
    from graphwiki_kb.services.project_service import ProjectService
    from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService

    paths = build_project_paths(tmp_path)
    ProjectService(paths).ensure_structure()
    ConfigService(paths).ensure_files()
    config = ConfigService(paths).load()
    config["wikigraph"]["mode"] = "lightrag"
    config["embeddings"]["provider"] = "anthropic"  # force BM25 (no network)
    manifest = ManifestService(paths)
    manifest.ensure_manifest()
    normalized_rel = "raw/normalized/dpr.md"
    (tmp_path / normalized_rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / normalized_rel).write_text(
        "Dense Passage Retrieval is a dual encoder dense retriever for QA.",
        encoding="utf-8",
    )
    from graphwiki_kb.models.source_models import RawSourceRecord

    manifest.save_source(
        RawSourceRecord(
            source_id="src_dpr",
            slug="dpr",
            title="DPR",
            origin="/tmp/dpr.pdf",
            source_type="pdf",
            raw_path="raw/sources/dpr.pdf",
            normalized_path=normalized_rel,
            content_hash="abc",
            ingested_at="2026-01-01T00:00:00Z",
        )
    )
    service = WikiGraphIndexService(
        paths=paths, config=config, manifest_service=manifest
    )
    service.build()
    written = service.export_artifacts()
    assert any("wiki/wikigraph/index.md" in rel for rel in written)
    assert (paths.wiki_dir / "wikigraph" / "index.md").exists()


def test_lightrag_export_accepts_mode_separated_base(tmp_path: Path) -> None:
    paths = build_project_paths(tmp_path)
    store = _store(tmp_path)
    service = WikiGraphLightExportService(
        paths=paths,
        store=store,
        base_subdir="wikigraph/lightrag",
    )

    written = service.export()

    assert any("wiki/wikigraph/lightrag/index.md" in rel for rel in written)
    assert (paths.wiki_dir / "wikigraph" / "lightrag" / "index.md").exists()
