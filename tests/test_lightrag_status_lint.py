"""Tests for LightRAG status block and lint checks."""

from __future__ import annotations

from pathlib import Path

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.services.config_service import ConfigService
from graphwiki_kb.services.lint_service import LintService
from graphwiki_kb.services.manifest_service import ManifestService
from graphwiki_kb.services.project_service import ProjectService, build_project_paths
from graphwiki_kb.services.status_service import StatusService
from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService


def _setup(tmp_path: Path, *, build: bool = True):
    paths = build_project_paths(tmp_path)
    ProjectService(paths).ensure_structure()
    ConfigService(paths).ensure_files()
    config = ConfigService(paths).load()
    config["wikigraph"]["mode"] = "lightrag"
    config["embeddings"]["provider"] = "anthropic"  # force BM25, no network
    manifest = ManifestService(paths)
    manifest.ensure_manifest()
    normalized_rel = "raw/normalized/dpr.md"
    (tmp_path / normalized_rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / normalized_rel).write_text(
        "Dense Passage Retrieval is a dual encoder dense retriever for open domain QA.",
        encoding="utf-8",
    )
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
    index_service = WikiGraphIndexService(
        paths=paths, config=config, manifest_service=manifest
    )
    if build:
        index_service.build()
    return paths, config, manifest, index_service


def test_status_reports_fresh_lightrag_index(tmp_path: Path) -> None:
    _paths, _config, _manifest, index_service = _setup(tmp_path)
    status = index_service.status()
    assert status["mode"] == "lightrag"
    assert status["initialized"] is True
    assert status["fresh"] is True
    assert status["chunk_count"] >= 1
    assert status["embedding_model"] == "bm25-fallback"
    assert status["embedding_tier"] == "fallback"
    assert status["provider_required"] is True


def test_status_not_built(tmp_path: Path) -> None:
    _paths, _config, _manifest, index_service = _setup(tmp_path, build=False)
    status = index_service.status()
    assert status["initialized"] is False
    assert status["stale_reasons"] == ["index not built"]


def test_status_detects_new_source_as_stale(tmp_path: Path) -> None:
    _paths, _config, manifest, index_service = _setup(tmp_path)
    # Add a new source AFTER building -> index becomes stale.
    manifest.save_source(
        RawSourceRecord(
            source_id="src_new",
            slug="new",
            title="NEW",
            origin="/tmp/new.pdf",
            source_type="pdf",
            raw_path="raw/sources/new.pdf",
            normalized_path="raw/normalized/new.md",
            content_hash="xyz",
            ingested_at="2026-01-02T00:00:00Z",
        )
    )
    status = index_service.status()
    assert status["fresh"] is False
    assert any("new source" in reason for reason in status["stale_reasons"])


def test_status_service_snapshot_includes_wikigraph(tmp_path: Path) -> None:
    paths, config, manifest, _index_service = _setup(tmp_path)
    status_service = StatusService(paths, manifest, config=config)
    snapshot = status_service.snapshot(initialized=True)
    assert snapshot.wikigraph_status.get("mode") == "lightrag"
    assert snapshot.wikigraph_status.get("initialized") is True


def test_lint_clean_when_index_fresh(tmp_path: Path) -> None:
    paths, config, manifest, _index_service = _setup(tmp_path)
    report = LintService(paths, config, manifest).lint()
    codes = {issue.code for issue in report.issues}
    assert "wikigraph-lightrag-missing-index" not in codes
    assert "wikigraph-relation-endpoint-missing" not in codes


def test_lint_flags_missing_index(tmp_path: Path) -> None:
    paths, config, manifest, _index_service = _setup(tmp_path, build=False)
    report = LintService(paths, config, manifest).lint()
    codes = {issue.code for issue in report.issues}
    assert "wikigraph-lightrag-missing-index" in codes


def test_lint_flags_stale_index(tmp_path: Path) -> None:
    paths, config, manifest, _index_service = _setup(tmp_path)
    manifest.save_source(
        RawSourceRecord(
            source_id="src_new",
            slug="new2",
            title="NEW2",
            origin="/tmp/new2.pdf",
            source_type="pdf",
            raw_path="raw/sources/new2.pdf",
            normalized_path="raw/normalized/new2.md",
            content_hash="zzz",
            ingested_at="2026-01-02T00:00:00Z",
        )
    )
    report = LintService(paths, config, manifest).lint()
    codes = {issue.code for issue in report.issues}
    assert "wikigraph-lightrag-stale" in codes


def test_lint_skipped_for_classic_mode(tmp_path: Path) -> None:
    paths, config, manifest, _index_service = _setup(tmp_path, build=False)
    config["wikigraph"]["mode"] = "classic"
    report = LintService(paths, config, manifest).lint()
    codes = {issue.code for issue in report.issues}
    assert "wikigraph-lightrag-missing-index" not in codes
