"""Tests for the opt-in LightRAG LLM extractor config + provider gating."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from graphwiki_kb.services.config_service import (
    DEFAULT_CONFIG,
    resolve_wikigraph_config,
)
from graphwiki_kb.wikigraph.light_models import LightGraphBuildReport


def test_default_extraction_mode_is_deterministic() -> None:
    assert resolve_wikigraph_config(DEFAULT_CONFIG).lightrag.extraction_mode == (
        "deterministic"
    )


def test_extraction_mode_llm_resolves() -> None:
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["wikigraph"]["lightrag"]["extraction"]["extractor"] = "llm"
    assert resolve_wikigraph_config(cfg).lightrag.extraction_mode == "llm"


def test_extraction_block_missing_defaults_to_deterministic() -> None:
    # A v9 config whose lightrag block predates the `extraction` key still
    # resolves (pydantic fills the default).
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["wikigraph"]["lightrag"].pop("extraction", None)
    assert resolve_wikigraph_config(cfg).lightrag.extraction_mode == "deterministic"


def test_invalid_extractor_rejected() -> None:
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["wikigraph"]["lightrag"]["extraction"]["extractor"] = "nonsense"
    with pytest.raises(ValueError):
        resolve_wikigraph_config(cfg)


def _service(tmp_path: Path, *, extractor: str):
    from graphwiki_kb.services.config_service import ConfigService
    from graphwiki_kb.services.manifest_service import ManifestService
    from graphwiki_kb.services.project_service import (
        ProjectService,
        build_project_paths,
    )
    from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService

    paths = build_project_paths(tmp_path)
    ProjectService(paths).ensure_structure()
    ConfigService(paths).ensure_files()
    config = ConfigService(paths).load()
    config["wikigraph"]["mode"] = "lightrag"
    config["wikigraph"]["lightrag"]["extraction"]["extractor"] = extractor
    config["provider"] = {"name": "openai"}  # would be used only if extractor=llm
    manifest = ManifestService(paths)
    manifest.ensure_manifest()
    return WikiGraphIndexService(paths=paths, config=config, manifest_service=manifest)


def _patch_builder(monkeypatch):
    captured: dict = {}

    def _fake_build(root, sources, **kwargs):
        captured["provider"] = kwargs.get("provider")
        captured["provider_identity"] = kwargs.get("provider_identity")
        return LightGraphBuildReport(built_at="t", tier="x")

    import graphwiki_kb.services.wikigraph_index_service as mod

    monkeypatch.setattr(mod, "build_lightgraph_index", _fake_build)
    return captured


def test_deterministic_mode_passes_no_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _patch_builder(monkeypatch)
    service = _service(tmp_path, extractor="deterministic")
    service.build()
    assert captured["provider"] is None
    assert captured["provider_identity"] == "deterministic"


def test_llm_mode_passes_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _patch_builder(monkeypatch)
    service = _service(tmp_path, extractor="llm")
    service.build()
    assert captured["provider"] is not None
    assert captured["provider_identity"].startswith("openai:")
