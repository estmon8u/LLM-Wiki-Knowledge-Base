"""Tests for the v8 → v9 config migration that introduces wikigraph.lightrag."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from graphwiki_kb.services.config_service import (
    CURRENT_CONFIG_VERSION,
    DEFAULT_CONFIG,
    ConfigService,
    resolve_wikigraph_config,
)
from graphwiki_kb.services.project_service import build_project_paths


def _legacy_v8_config() -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    # Drop the lightrag-only fields that v8 did not know about.
    config["wikigraph"].pop("lightrag", None)
    config["wikigraph"].pop("mode", None)
    config["version"] = 8
    return config


def test_v8_to_v9_migration_adds_mode_and_lightrag(tmp_path: Path):
    paths = build_project_paths(tmp_path)
    legacy = _legacy_v8_config()
    paths.config_file.write_text(yaml.safe_dump(legacy), encoding="utf-8")
    service = ConfigService(paths)
    loaded = service.load()
    assert loaded["version"] == CURRENT_CONFIG_VERSION
    assert loaded["wikigraph"]["mode"] == "classic"
    assert "lightrag" in loaded["wikigraph"]
    lr = loaded["wikigraph"]["lightrag"]
    assert lr["chunk_token_size"] == 1200
    assert lr["overlap_tokens"] == 100
    assert lr["retrieval"]["default_method"] == "hybrid"
    assert lr["embeddings"]["provider"] == "bm25"
    # The file on disk was updated to v9 by the migration.
    persisted = yaml.safe_load(paths.config_file.read_text(encoding="utf-8"))
    assert persisted["version"] == CURRENT_CONFIG_VERSION


def test_v8_to_v9_migration_preserves_user_mode_when_present(tmp_path: Path):
    paths = build_project_paths(tmp_path)
    legacy = _legacy_v8_config()
    legacy["wikigraph"]["mode"] = "lightrag"
    paths.config_file.write_text(yaml.safe_dump(legacy), encoding="utf-8")
    service = ConfigService(paths)
    loaded = service.load()
    assert loaded["wikigraph"]["mode"] == "lightrag"


def test_resolve_wikigraph_config_exposes_lightrag_runtime():
    runtime = resolve_wikigraph_config(DEFAULT_CONFIG)
    assert runtime.mode == "classic"
    assert runtime.lightrag.chunk_token_size == 1200
    assert runtime.lightrag.retrieval.default_method == "hybrid"
    assert runtime.lightrag.embeddings.provider == "bm25"
