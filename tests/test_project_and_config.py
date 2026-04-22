from __future__ import annotations

from pathlib import Path

import click
import pytest

from src.services import build_services
from src.services.config_service import (
    CURRENT_CONFIG_VERSION,
    ConfigService,
    DEFAULT_CONFIG,
    DEFAULT_SCHEMA,
    _apply_config_migrations,
    _config_version,
    _deep_merge,
)
from src.services.project_service import (
    ProjectService,
    atomic_copy_file,
    atomic_write_text,
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


def test_atomic_write_text_overwrites_without_leaving_temp_files(
    tmp_path: Path,
) -> None:
    target = tmp_path / "output.md"

    atomic_write_text(target, "first")
    atomic_write_text(target, "second")

    assert target.read_text(encoding="utf-8") == "second"
    assert not any(path.suffix == ".tmp" for path in tmp_path.iterdir())


def test_atomic_copy_file_copies_without_leaving_temp_files(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "dest.txt"
    source.write_text("payload", encoding="utf-8")

    atomic_copy_file(source, destination)

    assert destination.read_text(encoding="utf-8") == "payload"
    assert not any(path.suffix == ".tmp" for path in tmp_path.iterdir())


def test_project_service_creates_structure_and_relative_paths(
    uninitialized_project,
) -> None:
    project_service = ProjectService(uninitialized_project.paths)

    assert project_service.is_initialized() is False

    created = project_service.ensure_structure()
    assert "raw" in created
    assert "raw/normalized" in created
    assert "wiki/sources" in created
    assert "wiki/analysis" in created
    assert "wiki/log.md" in created
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


def test_config_service_loads_custom_provider_settings(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 3\n"
        "provider:\n"
        "  name: gemini\n"
        "providers:\n"
        "  openai:\n"
        "    model: gpt-5.4\n"
        "    api_key_env: OPENAI_ALT\n"
        "    reasoning_effort: medium\n"
        "  anthropic:\n"
        "    model: claude-opus-4-6\n"
        "    api_key_env: ANTHROPIC_ALT\n"
        "    thinking_budget: 2048\n"
        "  gemini:\n"
        "    model: gemini-2.5-flash\n"
        "    api_key_env: GEMINI_ALT\n"
        "    reasoning_effort: low\n",
        encoding="utf-8",
    )

    loaded = ConfigService(test_project.paths).load()

    assert loaded["provider"]["name"] == "gemini"
    assert loaded["providers"]["openai"]["model"] == "gpt-5.4"
    assert loaded["providers"]["openai"]["api_key_env"] == "OPENAI_ALT"
    assert loaded["providers"]["anthropic"]["thinking_budget"] == 2048
    assert loaded["providers"]["gemini"]["reasoning_effort"] == "low"


def test_config_service_invalid_provider_settings_raise(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 3\n"
        "providers:\n"
        "  openai:\n"
        "    model: gpt-5.4-mini\n"
        "    api_key_env: OPENAI_API_KEY\n"
        "    reasoning_effort: 7\n"
        "  anthropic:\n"
        "    model: claude-sonnet-4-6\n"
        "    api_key_env: ANTHROPIC_API_KEY\n"
        "    thinking_budget: 10000\n"
        "  gemini:\n"
        "    model: gemini-3.1-flash-lite-preview\n"
        "    api_key_env: GEMINI_API_KEY\n"
        "    reasoning_effort: high\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="providers.openai.reasoning_effort"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_provider_settings_in_active_provider_block(
    test_project,
) -> None:
    test_project.paths.config_file.write_text(
        "version: 3\n" "provider:\n" "  name: openai\n" "  model: gpt-5.4\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only supports 'name'"):
        ConfigService(test_project.paths).load()


def test_config_version_helper_uses_legacy_default_and_validates_values() -> None:
    assert _config_version({}) == 1
    assert (
        _config_version({"version": CURRENT_CONFIG_VERSION}) == CURRENT_CONFIG_VERSION
    )

    with pytest.raises(ValueError, match="must be an integer"):
        _config_version({"version": "two"})

    with pytest.raises(ValueError, match=">= 1"):
        _config_version({"version": 0})


def test_apply_config_migrations_upgrades_version_one_payload() -> None:
    migrated, changed = _apply_config_migrations(
        {
            "version": 1,
            "project": {"name": "Legacy Project"},
            "storage": {"wiki_sources_dir": "wiki/sources"},
            "compile": {
                "summary_paragraph_limit": 2,
                "excerpt_character_limit": 180,
            },
        }
    )

    assert changed is True
    assert migrated["version"] == CURRENT_CONFIG_VERSION
    assert migrated["project"]["name"] == "Legacy Project"
    assert migrated["compile"]["excerpt_character_limit"] == 180
    assert "summary_paragraph_limit" not in migrated["compile"]
    assert migrated["storage"]["raw_normalized_dir"] == "raw/normalized"
    assert migrated["provider"] == {}
    assert migrated["providers"]["openai"]["model"] == "gpt-5.4-mini"


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


def test_config_service_migrates_legacy_file_and_rewrites_disk(test_project) -> None:
    import yaml

    test_project.paths.config_file.write_text(
        "version: 1\n"
        "project:\n"
        "  name: Legacy Project\n"
        "compile:\n"
        "  summary_paragraph_limit: 2\n"
        "  excerpt_character_limit: 120\n",
        encoding="utf-8",
    )

    loaded = ConfigService(test_project.paths).load()

    assert loaded["version"] == CURRENT_CONFIG_VERSION
    assert loaded["project"]["name"] == "Legacy Project"
    assert loaded["storage"]["raw_normalized_dir"] == "raw/normalized"
    assert loaded["provider"] == {}
    assert loaded["providers"]["anthropic"]["model"] == "claude-sonnet-4-6"
    assert "summary_paragraph_limit" not in loaded["compile"]

    persisted = yaml.safe_load(
        test_project.paths.config_file.read_text(encoding="utf-8")
    )
    assert persisted["version"] == CURRENT_CONFIG_VERSION
    assert persisted["provider"] == {}
    assert persisted["providers"]["gemini"]["model"] == "gemini-3.1-flash-lite-preview"
    assert persisted["storage"]["raw_normalized_dir"] == "raw/normalized"
    assert "summary_paragraph_limit" not in persisted["compile"]


def test_config_service_migrates_provider_overrides_into_providers_section(
    test_project,
) -> None:
    test_project.paths.config_file.write_text(
        "version: 2\n"
        "provider:\n"
        "  name: openai\n"
        "  model: gpt-5.4\n"
        "  api_key_env: MY_OPENAI_KEY\n"
        "  reasoning_effort: low\n",
        encoding="utf-8",
    )

    loaded = ConfigService(test_project.paths).load()

    assert loaded["version"] == CURRENT_CONFIG_VERSION
    assert loaded["provider"] == {"name": "openai"}
    assert loaded["providers"]["openai"]["model"] == "gpt-5.4"
    assert loaded["providers"]["openai"]["api_key_env"] == "MY_OPENAI_KEY"
    assert loaded["providers"]["openai"]["reasoning_effort"] == "low"


def test_config_service_migrates_v3_file_to_v4_with_conversion_defaults(
    test_project,
) -> None:
    test_project.paths.config_file.write_text(
        "version: 3\nprovider:\n  name: openai\n",
        encoding="utf-8",
    )

    loaded = ConfigService(test_project.paths).load()

    assert loaded["version"] == CURRENT_CONFIG_VERSION
    assert loaded["conversion"]["mistral_ocr"]["model"] == "mistral-ocr-latest"
    assert loaded["conversion"]["html"]["renderer"] == "wkhtmltopdf"
    assert loaded["conversion"]["fallbacks"]["pdf"] == "docling"


def test_config_service_rejects_invalid_conversion_table_format(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 4\n"
        "conversion:\n"
        "  mistral_ocr:\n"
        "    model: mistral-ocr-latest\n"
        "    api_key_env: MISTRAL_API_KEY\n"
        "    table_format: text\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="table_format"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_invalid_html_renderer_settings(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 4\n"
        "conversion:\n"
        "  html:\n"
        "    renderer: playwright\n"
        "    wkhtmltopdf_path: ''\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conversion.html.renderer"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_invalid_conversion_fallbacks(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 4\n"
        "conversion:\n"
        "  fallbacks:\n"
        "    pdf: markitdown\n"
        "    docx: markitdown\n"
        "    pptx: markitdown\n"
        "    html: markitdown\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conversion.fallbacks.pdf"):
        ConfigService(test_project.paths).load()


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
        "compile_run_store",
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
    import yaml

    test_project.paths.config_file.write_text(
        "invalid: yaml: [broken\n", encoding="utf-8"
    )

    with pytest.raises(yaml.YAMLError):
        ConfigService(test_project.paths).load()


def test_config_future_version_raises(test_project) -> None:
    test_project.paths.config_file.write_text("version: 99\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported kb.config.yaml version"):
        ConfigService(test_project.paths).load()


def test_config_non_mapping_yaml_raises(test_project) -> None:
    test_project.paths.config_file.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a YAML mapping"):
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


def test_ensure_structure_seeds_wiki_log_file(uninitialized_project) -> None:
    project_service = ProjectService(uninitialized_project.paths)

    project_service.ensure_structure()

    log_file = uninitialized_project.paths.wiki_log_file
    assert log_file.exists()
    assert log_file.read_text(encoding="utf-8") == "# Activity Log\n"

    # Idempotent — second call does not overwrite
    log_file.write_text("# Activity Log\n\n- existing entry\n", encoding="utf-8")
    project_service.ensure_structure()
    assert "existing entry" in log_file.read_text(encoding="utf-8")


# --- schema_excerpt tests ---


def test_schema_excerpt_extracts_matching_section() -> None:
    from src.services.config_service import DEFAULT_SCHEMA, schema_excerpt

    result = schema_excerpt(DEFAULT_SCHEMA, ["Source Pages"])
    assert "## Source Pages" in result
    assert "Create one source page" in result
    assert "## Query Behavior" not in result


def test_schema_excerpt_returns_multiple_sections() -> None:
    from src.services.config_service import DEFAULT_SCHEMA, schema_excerpt

    result = schema_excerpt(DEFAULT_SCHEMA, ["Source Pages", "Query Behavior"])
    assert "## Source Pages" in result
    assert "## Query Behavior" in result


def test_schema_excerpt_missing_heading_returns_empty() -> None:
    from src.services.config_service import schema_excerpt

    result = schema_excerpt("# Just a title\n\nNo sections.\n", ["Missing Heading"])
    assert result == ""


def test_schema_excerpt_empty_schema_returns_empty() -> None:
    from src.services.config_service import schema_excerpt

    assert schema_excerpt("", ["Source Pages"]) == ""


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

        result = runner.invoke(main, ["status"])

        assert result.exit_code == 0
        # When no compile has happened, last_compile_at is not shown
        assert "last_compile_at" not in result.output
        assert "0 total" in result.output


def test_config_service_rejects_non_dict_conversion_block(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 4\nconversion: invalid\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="conversion.*mapping"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_unknown_conversion_section(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 4\nconversion:\n  extra_section:\n    key: value\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown sections"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_non_dict_mistral_ocr(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 4\nconversion:\n  mistral_ocr: invalid\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mistral_ocr.*mapping"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_unknown_mistral_ocr_keys(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 4\nconversion:\n  mistral_ocr:\n"
        "    model: mistral-ocr-latest\n    api_key_env: MISTRAL_API_KEY\n"
        "    table_format: markdown\n    extra: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown keys"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_empty_mistral_ocr_model(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 4\nconversion:\n  mistral_ocr:\n"
        "    model: ''\n    api_key_env: MISTRAL_API_KEY\n"
        "    table_format: markdown\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mistral_ocr.model"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_non_dict_html_block(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 4\nconversion:\n  html: invalid\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="html.*mapping"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_unknown_html_keys(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 4\nconversion:\n  html:\n"
        "    renderer: wkhtmltopdf\n    extra: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="html.*unknown keys"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_empty_wkhtmltopdf_path_string(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 4\nconversion:\n  html:\n"
        "    renderer: wkhtmltopdf\n    wkhtmltopdf_path: ' '\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="wkhtmltopdf_path"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_non_dict_fallbacks(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 4\nconversion:\n  fallbacks: invalid\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fallbacks.*mapping"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_unknown_fallback_keys(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 4\nconversion:\n  fallbacks:\n"
        "    pdf: docling\n    docx: markitdown\n"
        "    pptx: markitdown\n    html: markitdown\n    extra: foo\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fallbacks.*unknown keys"):
        ConfigService(test_project.paths).load()
