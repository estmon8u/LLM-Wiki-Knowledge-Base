from __future__ import annotations

from pathlib import Path

import click

from src.services import build_services
from src.services.config_service import (
    ConfigService,
    DEFAULT_CONFIG,
    DEFAULT_SCHEMA,
    _deep_merge,
)
from src.services.project_service import (
    ProjectService,
    build_project_paths,
    discover_project_root,
    slugify,
    utc_now_iso,
)


def test_utc_now_iso_uses_utc_and_strips_microseconds() -> None:
    timestamp = utc_now_iso()

    assert timestamp.endswith("+00:00")
    assert "." not in timestamp.split("+")[0]


def test_slugify_normalizes_values_and_handles_empty() -> None:
    assert slugify("  A Complex_Title!!  ") == "a-complex-title"
    assert slugify("***") == "untitled"


def test_discover_project_root_finds_parent_marker(test_project) -> None:
    nested = test_project.root / "nested" / "folder"
    nested.mkdir(parents=True)

    assert discover_project_root(nested) == test_project.root


def test_discover_project_root_returns_start_when_no_markers(
    uninitialized_project,
) -> None:
    nested = uninitialized_project.root / "deeper"
    nested.mkdir()

    assert discover_project_root(nested) == nested.resolve()


def test_build_project_paths_uses_expected_layout(tmp_path: Path) -> None:
    paths = build_project_paths(tmp_path)

    assert paths.config_file == tmp_path / "kb.config.yaml"
    assert paths.raw_manifest_file == tmp_path / "raw" / "_manifest.json"
    assert paths.raw_normalized_dir == tmp_path / "raw" / "normalized"
    assert paths.vault_obsidian_dir == tmp_path / "vault" / "obsidian"
    assert paths.graph_exports_dir == tmp_path / "graph" / "exports"


def test_project_service_creates_structure_and_relative_paths(
    uninitialized_project,
) -> None:
    project_service = ProjectService(uninitialized_project.paths)

    assert project_service.is_initialized() is False

    created = project_service.ensure_structure()
    assert "raw" in created
    assert "raw/normalized" in created
    assert "wiki/sources" in created
    assert project_service.ensure_structure() == []

    some_file = uninitialized_project.root / "wiki" / "sources" / "item.md"
    some_file.parent.mkdir(parents=True, exist_ok=True)
    some_file.write_text("sample", encoding="utf-8")
    assert project_service.to_relative_path(some_file) == "wiki/sources/item.md"


def test_config_service_loads_defaults_and_creates_files(uninitialized_project) -> None:
    config_service = ConfigService(uninitialized_project.paths)

    assert config_service.load() == DEFAULT_CONFIG
    assert config_service.load_schema() == DEFAULT_SCHEMA

    created = config_service.ensure_files()
    assert created == ["kb.config.yaml", "kb.schema.md"]
    assert config_service.ensure_files() == []


def test_config_service_merges_custom_config(test_project) -> None:
    test_project.paths.config_file.write_text(
        "project:\n"
        "  name: Custom Project\n"
        "compile:\n"
        "  excerpt_character_limit: 120\n",
        encoding="utf-8",
    )
    config_service = ConfigService(test_project.paths)

    loaded = config_service.load()

    assert loaded["project"]["name"] == "Custom Project"
    assert loaded["project"]["description"] == DEFAULT_CONFIG["project"]["description"]
    assert loaded["compile"]["excerpt_character_limit"] == 120
    assert (
        loaded["compile"]["summary_paragraph_limit"]
        == DEFAULT_CONFIG["compile"]["summary_paragraph_limit"]
    )


def test_config_service_load_schema_reads_custom_schema(test_project) -> None:
    test_project.paths.schema_file.write_text("# custom\n", encoding="utf-8")

    assert ConfigService(test_project.paths).load_schema() == "# custom\n"


def test_deep_merge_updates_nested_values_without_losing_defaults() -> None:
    merged = _deep_merge(
        {"project": {"name": "base", "description": "desc"}, "value": 1},
        {"project": {"name": "override"}, "extra": True},
    )

    assert merged == {
        "project": {"name": "override", "description": "desc"},
        "value": 1,
        "extra": True,
    }


def test_build_services_returns_expected_keys(test_project) -> None:
    services = build_services(test_project.paths, test_project.config)

    assert set(services) == {
        "project",
        "config",
        "manifest",
        "ingest",
        "compile",
        "diff",
        "lint",
        "review",
        "search",
        "status",
        "query",
        "export",
    }
