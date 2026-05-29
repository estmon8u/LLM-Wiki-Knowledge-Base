"""Tests for LightRAG query-engine caching in WikiGraphQueryService."""

from __future__ import annotations

import json
from pathlib import Path

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.services.config_service import ConfigService
from graphwiki_kb.services.manifest_service import ManifestService
from graphwiki_kb.services.project_service import ProjectService, build_project_paths
from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService
from graphwiki_kb.services.wikigraph_query_service import WikiGraphQueryService


def _project(tmp_path: Path):
    paths = build_project_paths(tmp_path)
    ProjectService(paths).ensure_structure()
    ConfigService(paths).ensure_files()
    config = ConfigService(paths).load()
    config["wikigraph"]["mode"] = "lightrag"
    config["embeddings"]["provider"] = "anthropic"  # force BM25, offline
    manifest = ManifestService(paths)
    manifest.ensure_manifest()
    rel = "raw/normalized/dpr.md"
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / rel).write_text(
        "Dense Passage Retrieval is a dual encoder dense retriever for QA.",
        encoding="utf-8",
    )
    manifest.save_source(
        RawSourceRecord(
            source_id="dpr",
            slug="dpr",
            title="DPR",
            origin="/tmp/dpr.pdf",
            source_type="pdf",
            raw_path="raw/sources/dpr.pdf",
            normalized_path=rel,
            content_hash="abc",
            ingested_at="2026-01-01T00:00:00Z",
        )
    )
    index_service = WikiGraphIndexService(
        paths=paths, config=config, manifest_service=manifest
    )
    index_service.build()
    qs = WikiGraphQueryService(
        paths=paths, index_service=index_service, provider=None, config=config
    )
    return paths, qs


def test_engine_is_cached_across_calls(tmp_path: Path) -> None:
    _paths, qs = _project(tmp_path)
    qs.find("dense retriever", method="basic")
    engine1 = qs._light_engine
    assert engine1 is not None
    qs.find("dual encoder", method="basic")
    # Same engine object reused (no reload / BM25 refit) when built_at is stable.
    assert qs._light_engine is engine1


def test_cache_invalidated_when_built_at_changes(tmp_path: Path) -> None:
    paths, qs = _project(tmp_path)
    qs.find("dense retriever", method="basic")
    engine1 = qs._light_engine
    # Simulate a rebuild by changing the persisted build timestamp.
    manifest_path = paths.graph_dir / "wikigraph" / "lightrag" / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["built_at"] = "2099-01-01T00:00:00+00:00"
    manifest_path.write_text(json.dumps(manifest))
    qs.find("dense retriever", method="basic")
    assert qs._light_engine is not engine1


def test_explicit_invalidate(tmp_path: Path) -> None:
    _paths, qs = _project(tmp_path)
    qs.find("dense retriever", method="basic")
    engine1 = qs._light_engine
    qs.invalidate_lightgraph_cache()
    assert qs._light_engine is None
    qs.find("dense retriever", method="basic")
    assert qs._light_engine is not None
    assert qs._light_engine is not engine1
