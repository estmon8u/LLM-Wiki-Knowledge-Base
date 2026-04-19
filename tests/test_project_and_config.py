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
        "concepts",
        "diff",
        "doctor",
        "lint",
        "review",
        "search",
        "status",
        "query",
        "export",
    }


# --- P1 boundary/negative tests ---


def test_config_unknown_key_preserved_through_merge(test_project) -> None:
    test_project.paths.config_file.write_text(
        "custom_key: custom_value\n", encoding="utf-8"
    )

    loaded = ConfigService(test_project.paths).load()

    assert loaded["custom_key"] == "custom_value"
    assert loaded["project"]["name"] == DEFAULT_CONFIG["project"]["name"]


def test_config_nested_override_excerpt_character_limit(test_project) -> None:
    test_project.paths.config_file.write_text(
        "compile:\n  excerpt_character_limit: 500\n", encoding="utf-8"
    )

    loaded = ConfigService(test_project.paths).load()

    assert loaded["compile"]["excerpt_character_limit"] == 500


def test_config_invalid_yaml_raises(test_project) -> None:
    import pytest
    import yaml

    test_project.paths.config_file.write_text(
        "invalid: yaml: [broken\n", encoding="utf-8"
    )

    with pytest.raises(yaml.YAMLError):
        ConfigService(test_project.paths).load()


def test_discover_project_root_finds_schema_only(tmp_path: Path) -> None:
    (tmp_path / "kb.schema.md").write_text("# Schema\n", encoding="utf-8")
    nested = tmp_path / "sub" / "deep"
    nested.mkdir(parents=True)

    assert discover_project_root(nested) == tmp_path.resolve()


def test_ensure_structure_returns_only_newly_created(test_project) -> None:
    project_service = ProjectService(test_project.paths)

    first_run = project_service.ensure_structure()
    assert first_run == []

    new_dir = test_project.paths.graph_exports_dir
    if new_dir.exists():
        new_dir.rmdir()

    second_run = project_service.ensure_structure()
    assert "graph/exports" in second_run
    assert len(second_run) == 1


def test_slugify_all_special_characters() -> None:
    assert slugify("!!!???") == "untitled"
    assert slugify("@#$%^&*") == "untitled"
    assert slugify("   ") == "untitled"


# --- P5 data model tests: computed properties ---


def test_lint_report_zero_issues_all_counts_zero() -> None:
    from src.models.wiki_models import LintReport

    report = LintReport(issues=[])

    assert report.error_count == 0
    assert report.warning_count == 0
    assert report.suggestion_count == 0


def test_diff_report_all_three_states_simultaneously() -> None:
    from src.models.wiki_models import DiffEntry, DiffReport

    report = DiffReport(
        entries=[
            DiffEntry("id-1", "a", "A", "new", "raw/a.md", "not yet compiled"),
            DiffEntry("id-2", "b", "B", "changed", "raw/b.md", "source changed"),
            DiffEntry("id-3", "c", "C", "up_to_date", "raw/c.md"),
        ]
    )

    assert report.new_count == 1
    assert report.changed_count == 1
    assert report.up_to_date_count == 1


def test_review_report_issue_count_matches_len() -> None:
    from src.models.wiki_models import ReviewIssue, ReviewReport

    issues = [
        ReviewIssue("suggestion", "overlapping-topics", ["a.md", "b.md"], "msg1"),
        ReviewIssue("suggestion", "terminology-variant", ["c.md"], "msg2"),
        ReviewIssue("suggestion", "overlapping-topics", ["d.md", "e.md"], "msg3"),
    ]
    report = ReviewReport(issues=issues, mode="provider:stub-1")

    assert report.issue_count == len(report.issues) == 3


def test_raw_source_record_from_dict_missing_optional_fields() -> None:
    from src.models.source_models import RawSourceRecord

    minimal = {
        "source_id": "id-1",
        "slug": "minimal",
        "title": "Minimal",
        "origin": "origin.md",
        "source_type": "file",
        "raw_path": "raw/sources/minimal.md",
        "content_hash": "hash-1",
        "ingested_at": "2026-04-14T00:00:00+00:00",
    }

    record = RawSourceRecord.from_dict(minimal)

    assert record.compiled_at is None
    assert record.compiled_from_hash is None
    assert record.normalized_path is None
    assert record.metadata == {}


def test_status_snapshot_none_compile_prints_na() -> None:
    from click.testing import CliRunner
    from src.cli import main

    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["show", "status"])

        assert result.exit_code == 0
        assert "last_compile_at: n/a" in result.output
