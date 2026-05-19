"""Tests for test project and config.

This module belongs to `tests.test_project_and_config` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graphwiki_kb.services import build_services
from graphwiki_kb.services.config_service import (
    CURRENT_CONFIG_VERSION,
    DEFAULT_CONFIG,
    DEFAULT_SCHEMA,
    ConfigService,
    _apply_config_migrations,
    _config_version,
    _deep_merge,
    resolve_graph_config,
)
from graphwiki_kb.services.graphrag_defaults import (
    DEFAULT_GRAPHRAG_EMBEDDING_MODEL,
    DEFAULT_GRAPHRAG_MODEL,
    DEFAULT_GRAPHRAG_PROVIDER,
)
from graphwiki_kb.services.project_service import (
    ProjectService,
    _replace_with_retry,
    atomic_copy_file,
    atomic_write_text,
    build_project_paths,
    discover_project_root,
    slugify,
    unique_markdown_heading,
    utc_now_iso,
)


def test_utc_now_iso_uses_utc_and_strips_microseconds() -> None:
    """Verifies that utc now iso uses utc and strips microseconds."""
    timestamp = utc_now_iso()

    assert timestamp.endswith("+00:00")
    assert "." not in timestamp.split("+")[0]


def test_atomic_write_text_preserves_exact_newline_bytes(tmp_path: Path) -> None:
    """Verifies that atomic text writes do not apply platform newline conversion."""
    output_path = tmp_path / "nested" / "payload.json"

    atomic_write_text(output_path, "one\ntwo\n")

    assert output_path.read_bytes() == b"one\ntwo\n"


def test_slugify_normalizes_values_and_handles_empty() -> None:
    """Verifies that slugify normalizes values and handles empty."""
    assert slugify("  A Complex_Title!!  ") == "a-complex-title"
    assert slugify("***") == "untitled"


def test_unique_markdown_heading_skips_existing_suffixes() -> None:
    """Verifies that unique markdown heading skips existing suffixes."""
    existing = "# Title\n\n## Entry\n\n## Entry (2)\n"

    assert unique_markdown_heading(existing, "## Entry") == "## Entry (3)"


def test_discover_project_root_finds_parent_marker(test_project) -> None:
    """Verifies that discover project root finds parent marker.

    Args:
        test_project: Test project value used by the operation.
    """
    nested = test_project.root / "nested" / "folder"
    nested.mkdir(parents=True)

    assert discover_project_root(nested) == test_project.root


def test_discover_project_root_returns_start_when_no_markers(
    uninitialized_project,
) -> None:
    """Verifies that discover project root returns start when no markers.

    Args:
        uninitialized_project: Uninitialized project value used by the operation.
    """
    nested = uninitialized_project.root / "deeper"
    nested.mkdir()

    assert discover_project_root(nested) == nested.resolve()


def test_build_project_paths_uses_expected_layout(tmp_path: Path) -> None:
    """Verifies that build project paths uses expected layout.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    paths = build_project_paths(tmp_path)

    assert paths.config_file == tmp_path / "kb.config.yaml"
    assert paths.raw_manifest_file == tmp_path / "raw" / "_manifest.json"
    assert paths.raw_normalized_dir == tmp_path / "raw" / "normalized"
    assert paths.vault_obsidian_dir == tmp_path / "vault" / "obsidian"
    assert paths.graph_exports_dir == tmp_path / "graph" / "exports"


def test_atomic_write_text_overwrites_without_leaving_temp_files(
    tmp_path: Path,
) -> None:
    """Verifies that atomic write text overwrites without leaving temp files.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    target = tmp_path / "output.md"

    atomic_write_text(target, "first")
    atomic_write_text(target, "second")

    assert target.read_text(encoding="utf-8") == "second"
    assert not any(path.suffix == ".tmp" for path in tmp_path.iterdir())


def test_atomic_copy_file_copies_without_leaving_temp_files(tmp_path: Path) -> None:
    """Verifies that atomic copy file copies without leaving temp files.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source = tmp_path / "source.txt"
    destination = tmp_path / "dest.txt"
    source.write_text("payload", encoding="utf-8")

    atomic_copy_file(source, destination)

    assert destination.read_text(encoding="utf-8") == "payload"
    assert not any(path.suffix == ".tmp" for path in tmp_path.iterdir())


def test_replace_with_retry_retries_transient_permission_error(monkeypatch) -> None:
    """Verifies that replace with retry retries transient permission error.

    Args:
        monkeypatch: Monkeypatch value used by the operation.
    """
    calls = []

    def fake_replace(source, destination):
        """Fake replace.

        Args:
            source: Source record or path being processed.
            destination: Destination value used by the operation.
        """
        calls.append((source, destination))
        if len(calls) < 2:
            raise PermissionError("locked")

    monkeypatch.setattr(
        "graphwiki_kb.services.project_service.os.replace", fake_replace
    )
    monkeypatch.setattr(
        "graphwiki_kb.services.project_service.time.sleep", lambda _secs: None
    )

    _replace_with_retry(Path("source.tmp"), Path("dest.txt"))

    assert calls == [
        (Path("source.tmp"), Path("dest.txt")),
        (Path("source.tmp"), Path("dest.txt")),
    ]


def test_replace_with_retry_raises_last_permission_error(monkeypatch) -> None:
    """Verifies that replace with retry raises last permission error.

    Args:
        monkeypatch: Monkeypatch value used by the operation.
    """
    calls = []

    def fake_replace(source, destination):
        """Fake replace.

        Args:
            source: Source record or path being processed.
            destination: Destination value used by the operation.
        """
        calls.append((source, destination))
        raise PermissionError("still locked")

    monkeypatch.setattr(
        "graphwiki_kb.services.project_service.os.replace", fake_replace
    )
    monkeypatch.setattr(
        "graphwiki_kb.services.project_service.time.sleep", lambda _secs: None
    )

    with pytest.raises(PermissionError, match="still locked"):
        _replace_with_retry(Path("source.tmp"), Path("dest.txt"))

    assert len(calls) == 10


def test_project_service_creates_structure_and_relative_paths(
    uninitialized_project,
) -> None:
    """Verifies that project service creates structure and relative paths.

    Args:
        uninitialized_project: Uninitialized project value used by the operation.
    """
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


def test_project_service_to_relative_path_handles_paths_outside_project(
    test_project,
    tmp_path: Path,
) -> None:
    """Regression: diagnostics for outside paths should not raise ValueError."""
    outside_dir = tmp_path.parent / f"{tmp_path.name}-outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "outside.md"
    outside_file.write_text("outside", encoding="utf-8")

    assert (
        ProjectService(test_project.paths).to_relative_path(outside_file)
        == outside_file.resolve().as_posix()
    )


def test_config_service_loads_defaults_and_creates_files(uninitialized_project) -> None:
    """Verifies that config service loads defaults and creates files.

    Args:
        uninitialized_project: Uninitialized project value used by the operation.
    """
    config_service = ConfigService(uninitialized_project.paths)

    assert config_service.load() == DEFAULT_CONFIG
    graph = config_service.load()["graph"]
    assert graph["provider"] == DEFAULT_GRAPHRAG_PROVIDER
    assert graph["model"] == DEFAULT_GRAPHRAG_MODEL
    assert graph["embedding_provider"] == DEFAULT_GRAPHRAG_PROVIDER
    assert graph["embedding_model"] == DEFAULT_GRAPHRAG_EMBEDDING_MODEL
    assert graph["api_key_env"] is None
    assert graph["embedding_api_key_env"] is None
    assert graph["chunking"] == {"size": 1200, "overlap": 150}
    assert "api" in graph["extraction"]["entity_types"]
    assert graph["extraction"]["max_gleanings"] == 2
    assert graph["input"]["max_source_bytes"] > 0
    assert graph["routing"] == {"aliases": {}}
    assert config_service.load()["providers"]["openai"]["api"] == "responses"
    assert config_service.load()["providers"]["openai"]["store_responses"] is False
    assert (
        config_service.load()["conversion"]["html"]["allow_local_file_access"] is False
    )
    assert resolve_graph_config(config_service.load()).api_key_env == "OPENAI_API_KEY"
    assert config_service.load_schema() == DEFAULT_SCHEMA

    created = config_service.ensure_files()
    assert created == ["kb.config.yaml", "kb.schema.md"]
    assert config_service.ensure_files() == []


def test_config_service_loads_custom_provider_settings(test_project) -> None:
    """Verifies that config service loads custom provider settings.

    Args:
        test_project: Test project value used by the operation.
    """
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
    """Verifies that config service invalid provider settings raise.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 3\n"
        "providers:\n"
        "  openai:\n"
        "    model: gpt-5.4-nano\n"
        "    api_key_env: OPENAI_API_KEY\n"
        "    reasoning_effort: 7\n"
        "  anthropic:\n"
        "    model: claude-sonnet-4-6\n"
        "    api_key_env: ANTHROPIC_API_KEY\n"
        "    thinking_budget: 10000\n"
        "  gemini:\n"
        "    model: gemini-2.5-flash\n"
        "    api_key_env: GEMINI_API_KEY\n"
        "    reasoning_effort: high\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"providers.openai.reasoning_effort"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_unknown_reasoning_effort_names(test_project) -> None:
    """Verifies provider reasoning/thinking effort names are validated."""
    test_project.paths.config_file.write_text(
        "version: 7\n"
        "providers:\n"
        "  openai:\n"
        "    model: gpt-5.4-nano\n"
        "    api_key_env: OPENAI_API_KEY\n"
        "    reasoning_effort: extreme\n"
        "  anthropic:\n"
        "    model: claude-sonnet-4-6\n"
        "    api_key_env: ANTHROPIC_API_KEY\n"
        "    thinking_effort: medium\n"
        "  gemini:\n"
        "    model: gemini-2.5-flash\n"
        "    api_key_env: GEMINI_API_KEY\n"
        "    reasoning_effort: high\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"providers.openai.reasoning_effort"):
        ConfigService(test_project.paths).load()

    test_project.paths.config_file.write_text(
        "version: 7\n"
        "providers:\n"
        "  openai:\n"
        "    model: gpt-5.4-nano\n"
        "    api_key_env: OPENAI_API_KEY\n"
        "    reasoning_effort: low\n"
        "  anthropic:\n"
        "    model: claude-sonnet-4-6\n"
        "    api_key_env: ANTHROPIC_API_KEY\n"
        "    thinking_effort: extreme\n"
        "  gemini:\n"
        "    model: gemini-2.5-flash\n"
        "    api_key_env: GEMINI_API_KEY\n"
        "    reasoning_effort: high\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"providers.anthropic.thinking_effort"):
        ConfigService(test_project.paths).load()

    test_project.paths.config_file.write_text(
        "version: 7\n"
        "providers:\n"
        "  openai:\n"
        "    model: gpt-5.4-nano\n"
        "    api_key_env: OPENAI_API_KEY\n"
        "    reasoning_effort: xhigh\n"
        "  anthropic:\n"
        "    model: claude-sonnet-4-6\n"
        "    api_key_env: ANTHROPIC_API_KEY\n"
        "    thinking_effort: high\n"
        "  gemini:\n"
        "    model: gemini-2.5-flash\n"
        "    api_key_env: GEMINI_API_KEY\n"
        "    reasoning_effort: xhigh\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"providers.gemini.reasoning_effort"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_provider_settings_in_active_provider_block(
    test_project,
) -> None:
    """Verifies that config service rejects provider settings in active provider block.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 3\n" "provider:\n" "  name: openai\n" "  model: gpt-5.4\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only supports 'name'"):
        ConfigService(test_project.paths).load()


def test_config_version_helper_uses_legacy_default_and_validates_values() -> None:
    """Verifies that config version helper uses legacy default and validates values."""
    assert _config_version({}) == 1
    assert (
        _config_version({"version": CURRENT_CONFIG_VERSION}) == CURRENT_CONFIG_VERSION
    )

    with pytest.raises(ValueError, match="must be an integer"):
        _config_version({"version": "two"})

    with pytest.raises(ValueError, match=">= 1"):
        _config_version({"version": 0})


def test_apply_config_migrations_upgrades_version_one_payload() -> None:
    """Verifies that apply config migrations upgrades version one payload."""
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
    assert migrated["providers"]["openai"]["model"] == "gpt-5.4-nano"


def test_config_service_merges_custom_config(test_project) -> None:
    """Verifies that config service merges custom config.

    Args:
        test_project: Test project value used by the operation.
    """
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
    """Verifies that config service migrates legacy file and rewrites disk.

    Args:
        test_project: Test project value used by the operation.
    """
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
    assert persisted["providers"]["gemini"]["model"] == "gemini-2.5-flash"
    assert persisted["storage"]["raw_normalized_dir"] == "raw/normalized"
    assert "summary_paragraph_limit" not in persisted["compile"]


def test_config_service_does_not_persist_failed_migration(test_project) -> None:
    """Verifies migration rewrites happen only after the migrated config validates."""
    original = (
        "version: 5\n"
        "graph:\n"
        "  provider: ''\n"
        "  model: gpt-4.1\n"
        "  embedding_model: text-embedding-3-small\n"
    )
    test_project.paths.config_file.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match=r"graph.provider"):
        ConfigService(test_project.paths).load()

    assert test_project.paths.config_file.read_text(encoding="utf-8") == original


def test_config_service_migrates_provider_overrides_into_providers_section(
    test_project,
) -> None:
    """Verifies that config service migrates provider overrides into providers section.

    Args:
        test_project: Test project value used by the operation.
    """
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
    """Verifies that config service migrates v3 file to v4 with conversion defaults.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 3\nprovider:\n  name: openai\n",
        encoding="utf-8",
    )

    loaded = ConfigService(test_project.paths).load()

    assert loaded["version"] == CURRENT_CONFIG_VERSION
    assert loaded["conversion"]["mistral_ocr"]["model"] == "mistral-ocr-latest"
    assert loaded["conversion"]["html"]["renderer"] == "wkhtmltopdf"
    assert loaded["conversion"]["fallbacks"]["pdf"] == ["docling", "markitdown"]
    assert loaded["graph"]["model"] == DEFAULT_GRAPHRAG_MODEL
    assert loaded["graph"]["embedding_model"] == DEFAULT_GRAPHRAG_EMBEDDING_MODEL


def test_config_service_migrates_v4_file_to_v5_with_graph_defaults(
    test_project,
) -> None:
    """Verifies that config service migrates v4 file to v5 with graph defaults.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 4\n"
        "conversion:\n"
        "  mistral_ocr:\n"
        "    model: mistral-ocr-latest\n"
        "    api_key_env: MISTRAL_API_KEY\n"
        "    table_format: markdown\n"
        "  html:\n"
        "    renderer: wkhtmltopdf\n"
        "    wkhtmltopdf_path:\n"
        "  fallbacks:\n"
        "    pdf: docling\n"
        "    docx: markitdown\n"
        "    pptx: markitdown\n"
        "    html: markitdown\n",
        encoding="utf-8",
    )

    loaded = ConfigService(test_project.paths).load()

    assert loaded["version"] == CURRENT_CONFIG_VERSION
    assert loaded["graph"] == DEFAULT_CONFIG["graph"]
    assert loaded["conversion"]["fallbacks"]["pdf"] == ["docling"]


def test_config_service_loads_custom_graph_config(test_project) -> None:
    """Verifies that config service loads custom graph config.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 6\n"
        "graph:\n"
        "  provider: openai\n"
        "  model: gpt-4.1\n"
        "  embedding_provider: gemini\n"
        "  embedding_model: gemini-embedding-001\n"
        "  api_key_env: OPENAI_GRAPH_KEY\n",
        encoding="utf-8",
    )

    loaded = ConfigService(test_project.paths).load()
    graph_config = resolve_graph_config(loaded)

    assert graph_config.provider == "openai"
    assert graph_config.model == "gpt-4.1"
    assert graph_config.embedding_provider == "gemini"
    assert graph_config.embedding_model == "gemini-embedding-001"
    assert graph_config.api_key_env == "OPENAI_GRAPH_KEY"
    assert graph_config.embedding_api_key_env == "GEMINI_API_KEY"
    assert graph_config.chunk_size == 1200
    assert graph_config.chunk_overlap == 150
    assert graph_config.entity_types
    assert graph_config.max_gleanings == 2
    assert graph_config.max_source_bytes > 0


def test_config_service_loads_graph_tuning_and_input_limits(test_project) -> None:
    """Verifies GraphRAG extraction, chunking, and sync limits are configurable."""
    test_project.paths.config_file.write_text(
        "version: 7\n"
        "graph:\n"
        "  provider: openai\n"
        "  model: gpt-4.1\n"
        "  embedding_model: text-embedding-3-small\n"
        "  chunking:\n"
        "    size: 900\n"
        "    overlap: 90\n"
        "  extraction:\n"
        "    entity_types: [concept, api]\n"
        "    max_gleanings: 3\n"
        "  input:\n"
        "    max_source_bytes: 2048\n",
        encoding="utf-8",
    )

    graph_config = resolve_graph_config(ConfigService(test_project.paths).load())

    assert graph_config.chunk_size == 900
    assert graph_config.chunk_overlap == 90
    assert graph_config.entity_types == ("concept", "api")
    assert graph_config.max_gleanings == 3
    assert graph_config.max_source_bytes == 2048


def test_graph_config_resolves_api_keys_from_provider_catalog(test_project) -> None:
    """Verifies that graph config resolves api keys from provider catalog.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 6\n"
        "providers:\n"
        "  openai:\n"
        "    model: gpt-5.4\n"
        "    api_key_env: OPENAI_ALT\n"
        "    reasoning_effort: medium\n"
        "  anthropic:\n"
        "    model: claude-sonnet-4-6\n"
        "    api_key_env: ANTHROPIC_ALT\n"
        "    thinking_budget: 2048\n"
        "  gemini:\n"
        "    model: gemini-2.5-flash\n"
        "    api_key_env: GEMINI_ALT\n"
        "    reasoning_effort: low\n"
        "graph:\n"
        "  provider: openai\n"
        "  model: gpt-4.1\n"
        "  embedding_provider: gemini\n"
        "  embedding_model: gemini-embedding-001\n",
        encoding="utf-8",
    )

    graph_config = resolve_graph_config(ConfigService(test_project.paths).load())

    assert graph_config.api_key_env == "OPENAI_ALT"
    assert graph_config.embedding_api_key_env == "GEMINI_ALT"


def test_graph_config_uses_explicit_embedding_api_key_override(test_project) -> None:
    """Verifies that graph config uses explicit embedding api key override.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.config["graph"] = {
        "provider": "openai",
        "model": "gpt-4.1-mini",
        "embedding_provider": "gemini",
        "embedding_model": "gemini-embedding-001",
        "embedding_api_key_env": "GEMINI_GRAPH_KEY",
    }

    graph_config = resolve_graph_config(test_project.config)

    assert graph_config.api_key_env == "OPENAI_API_KEY"
    assert graph_config.embedding_api_key_env == "GEMINI_GRAPH_KEY"


def test_graph_config_defaults_openai_api_key_when_catalog_missing() -> None:
    """Verifies that graph config defaults openai api key when catalog missing."""
    graph_config = resolve_graph_config(
        {
            "graph": DEFAULT_CONFIG["graph"],
            "providers": {},
        }
    )

    assert graph_config.api_key_env == "OPENAI_API_KEY"
    assert graph_config.embedding_api_key_env == "OPENAI_API_KEY"


def test_graph_config_requires_api_key_for_unknown_provider() -> None:
    """Verifies that graph config requires api key for unknown provider."""
    with pytest.raises(ValueError, match=r"graph.*api_key_env"):
        resolve_graph_config(
            {
                "graph": {
                    "provider": "voyage",
                    "model": "custom-chat",
                    "embedding_model": "custom-embedding",
                },
                "providers": {},
            }
        )


def test_config_service_migrates_legacy_graph_api_key_env(test_project) -> None:
    """Verifies that config service migrates legacy graph api key env.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 5\n"
        "graph:\n"
        "  provider: openai\n"
        "  model: gpt-4.1\n"
        "  embedding_model: text-embedding-3-small\n"
        "  api_key_env: GRAPHRAG_API_KEY\n",
        encoding="utf-8",
    )

    loaded = ConfigService(test_project.paths).load()
    graph_config = resolve_graph_config(loaded)

    assert loaded["version"] == CURRENT_CONFIG_VERSION
    assert loaded["graph"]["api_key_env"] is None
    assert graph_config.api_key_env == "OPENAI_API_KEY"
    assert graph_config.embedding_api_key_env == "OPENAI_API_KEY"


def test_config_service_migrates_version_six_runtime_options(test_project) -> None:
    """Verifies version 7 adds runtime API/routing/security defaults."""
    test_project.paths.config_file.write_text(
        "version: 6\n"
        "providers:\n"
        "  openai:\n"
        "    model: gpt-5.4-nano\n"
        "    api_key_env: OPENAI_API_KEY\n"
        "    reasoning_effort: high\n"
        "graph:\n"
        "  provider: openai\n"
        "  model: gpt-5.4-nano\n"
        "  embedding_provider: openai\n"
        "  embedding_model: text-embedding-3-small\n"
        "conversion:\n"
        "  html:\n"
        "    renderer: wkhtmltopdf\n"
        "    wkhtmltopdf_path:\n",
        encoding="utf-8",
    )

    loaded = ConfigService(test_project.paths).load()

    assert loaded["version"] == CURRENT_CONFIG_VERSION
    assert loaded["providers"]["openai"]["api"] == "responses"
    assert loaded["graph"]["routing"] == {"aliases": {}}
    assert loaded["conversion"]["html"]["allow_local_file_access"] is False


def test_config_service_rejects_invalid_graph_config(test_project) -> None:
    """Verifies that config service rejects invalid graph config.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 6\ngraph:\n  provider: ''\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"graph.provider"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_blank_optional_graph_config(test_project) -> None:
    """Verifies that config service rejects blank optional graph config.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 6\n" "graph:\n" "  provider: openai\n" "  embedding_provider: ''\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"graph.embedding_provider"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_unknown_graph_keys(test_project) -> None:
    """Verifies that config service rejects unknown graph keys.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 6\ngraph:\n  provider: openai\n  extra: nope\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"graph.*unknown keys"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_unknown_top_level_keys(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 7\nunexpected: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown top-level keys"):
        ConfigService(test_project.paths).load()


def test_config_service_allows_custom_values_under_extensions(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 7\nextensions:\n  local_note: keep\n",
        encoding="utf-8",
    )

    loaded = ConfigService(test_project.paths).load()

    assert loaded["extensions"] == {"local_note": "keep"}


def test_config_service_rejects_invalid_conversion_table_format(test_project) -> None:
    """Verifies that config service rejects invalid conversion table format.

    Args:
        test_project: Test project value used by the operation.
    """
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
    """Verifies that config service rejects invalid html renderer settings.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 4\n"
        "conversion:\n"
        "  html:\n"
        "    renderer: playwright\n"
        "    wkhtmltopdf_path: ''\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"conversion.html.renderer"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_invalid_conversion_fallbacks(test_project) -> None:
    """Verifies that config service rejects invalid conversion fallbacks.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 4\n"
        "conversion:\n"
        "  fallbacks:\n"
        "    pdf: tika\n"
        "    docx: markitdown\n"
        "    pptx: markitdown\n"
        "    html: markitdown\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"conversion.fallbacks.pdf"):
        ConfigService(test_project.paths).load()


def test_config_service_accepts_ordered_pdf_fallback_chain(test_project) -> None:
    """PDF fallbacks can be ordered after Mistral while legacy scalars still load."""
    test_project.paths.config_file.write_text(
        "version: 4\n"
        "conversion:\n"
        "  fallbacks:\n"
        "    pdf:\n"
        "      - docling\n"
        "      - markitdown\n"
        "    docx: markitdown\n"
        "    pptx: markitdown\n"
        "    html: markitdown\n",
        encoding="utf-8",
    )

    loaded = ConfigService(test_project.paths).load()

    assert loaded["conversion"]["fallbacks"]["pdf"] == ["docling", "markitdown"]
    assert loaded["conversion"]["fallbacks"]["docx"] == ["markitdown"]


def test_config_service_load_schema_reads_custom_schema(test_project) -> None:
    """Verifies that config service load schema reads custom schema.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.schema_file.write_text("# custom\n", encoding="utf-8")

    assert ConfigService(test_project.paths).load_schema() == "# custom\n"


def test_deep_merge_updates_nested_values_without_losing_defaults() -> None:
    """Verifies that deep merge updates nested values without losing defaults."""
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
    """Verifies that build services returns expected keys.

    Args:
        test_project: Test project value used by the operation.
    """
    services = build_services(test_project.paths, test_project.config)

    assert set(services) == {
        "project",
        "config",
        "manifest",
        "graphrag_input_sync",
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
        "graphrag_command",
        "graphrag_workspace",
        "graphrag_status",
        "graphrag_sync",
        "graphrag_query",
        "graphrag_find",
        "graphrag_wiki_export",
        "query_router",
        "graph_ask_controller",
        "compile_run_store",
    }


# --- P1 boundary/negative tests ---


def test_config_unknown_key_rejected_through_merge(test_project) -> None:
    """Verifies that config unknown key rejected through merge.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "custom_key: custom_value\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unknown top-level keys: custom_key"):
        ConfigService(test_project.paths).load()


def test_config_nested_override_excerpt_character_limit(test_project) -> None:
    """Verifies that config nested override excerpt character limit.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "compile:\n  excerpt_character_limit: 500\n", encoding="utf-8"
    )

    loaded = ConfigService(test_project.paths).load()

    assert loaded["compile"]["excerpt_character_limit"] == 500


def test_config_invalid_yaml_raises(test_project) -> None:
    """Verifies that config invalid yaml raises.

    Args:
        test_project: Test project value used by the operation.
    """
    import yaml

    test_project.paths.config_file.write_text(
        "invalid: yaml: [broken\n", encoding="utf-8"
    )

    with pytest.raises(yaml.YAMLError):
        ConfigService(test_project.paths).load()


def test_config_future_version_raises(test_project) -> None:
    """Verifies that config future version raises.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text("version: 99\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Unsupported kb.config.yaml version"):
        ConfigService(test_project.paths).load()


def test_config_non_mapping_yaml_raises(test_project) -> None:
    """Verifies that config non mapping yaml raises.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        ConfigService(test_project.paths).load()


def test_discover_project_root_finds_schema_only(tmp_path: Path) -> None:
    """Verifies that discover project root finds schema only.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    (tmp_path / "kb.schema.md").write_text("# Schema\n", encoding="utf-8")
    nested = tmp_path / "sub" / "deep"
    nested.mkdir(parents=True)

    assert discover_project_root(nested) == tmp_path.resolve()


def test_ensure_structure_returns_only_newly_created(test_project) -> None:
    """Verifies that ensure structure returns only newly created.

    Args:
        test_project: Test project value used by the operation.
    """
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
    """Verifies that ensure structure seeds wiki log file.

    Args:
        uninitialized_project: Uninitialized project value used by the operation.
    """
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
    """Verifies that schema excerpt extracts matching section."""
    from graphwiki_kb.services.config_service import DEFAULT_SCHEMA, schema_excerpt

    result = schema_excerpt(DEFAULT_SCHEMA, ["Source Pages"])
    assert "## Source Pages" in result
    assert "Create one source page" in result
    assert "## Query Behavior" not in result


def test_schema_excerpt_returns_multiple_sections() -> None:
    """Verifies that schema excerpt returns multiple sections."""
    from graphwiki_kb.services.config_service import DEFAULT_SCHEMA, schema_excerpt

    result = schema_excerpt(DEFAULT_SCHEMA, ["Source Pages", "Query Behavior"])
    assert "## Source Pages" in result
    assert "## Query Behavior" in result


def test_schema_excerpt_missing_heading_returns_empty() -> None:
    """Verifies that schema excerpt missing heading returns empty."""
    from graphwiki_kb.services.config_service import schema_excerpt

    result = schema_excerpt("# Just a title\n\nNo sections.\n", ["Missing Heading"])
    assert result == ""


def test_schema_excerpt_empty_schema_returns_empty() -> None:
    """Verifies that schema excerpt empty schema returns empty."""
    from graphwiki_kb.services.config_service import schema_excerpt

    assert schema_excerpt("", ["Source Pages"]) == ""


def test_slugify_all_special_characters() -> None:
    """Verifies that slugify all special characters."""
    assert slugify("!!!???") == "untitled"
    assert slugify("@#$%^&*") == "untitled"
    assert slugify("   ") == "untitled"


# --- P5 data model tests: computed properties ---


def test_lint_report_zero_issues_all_counts_zero() -> None:
    """Verifies that lint report zero issues all counts zero."""
    from graphwiki_kb.models.wiki_models import LintReport

    report = LintReport(issues=[])

    assert report.error_count == 0
    assert report.warning_count == 0
    assert report.suggestion_count == 0


def test_diff_report_counts_all_states_simultaneously() -> None:
    """Verifies that diff report counts all states simultaneously."""
    from graphwiki_kb.models.wiki_models import DiffEntry, DiffReport

    report = DiffReport(
        entries=[
            DiffEntry("id-1", "a", "A", "new", "raw/a.md", "not yet compiled"),
            DiffEntry("id-2", "b", "B", "changed", "raw/b.md", "source changed"),
            DiffEntry("id-3", "c", "C", "missing", "raw/c.md"),
            DiffEntry("id-4", "d", "D", "up_to_date", "raw/d.md"),
        ]
    )

    assert report.new_count == 1
    assert report.changed_count == 1
    assert report.missing_count == 1
    assert report.up_to_date_count == 1


def test_review_report_issue_count_matches_len() -> None:
    """Verifies that review report issue count matches len."""
    from graphwiki_kb.models.wiki_models import ReviewIssue, ReviewReport

    issues = [
        ReviewIssue("suggestion", "overlapping-topics", ["a.md", "b.md"], "msg1"),
        ReviewIssue("suggestion", "terminology-variant", ["c.md"], "msg2"),
        ReviewIssue("suggestion", "overlapping-topics", ["d.md", "e.md"], "msg3"),
    ]
    report = ReviewReport(issues=issues, mode="provider:stub-1")

    assert report.issue_count == len(report.issues) == 3


def test_raw_source_record_from_dict_missing_optional_fields() -> None:
    """Verifies that raw source record from dict missing optional fields."""
    from graphwiki_kb.models.source_models import RawSourceRecord

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
    """Verifies that status snapshot none compile prints na."""
    from click.testing import CliRunner

    from graphwiki_kb.cli import main

    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["status"])

        assert result.exit_code == 0
        # When no compile has happened, last_compile_at is not shown
        assert "last_compile_at" not in result.output
        assert "0 total" in result.output


def test_config_service_rejects_non_dict_conversion_block(test_project) -> None:
    """Verifies that config service rejects non dict conversion block.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 4\nconversion: invalid\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"conversion.*mapping"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_unknown_conversion_section(test_project) -> None:
    """Verifies that config service rejects unknown conversion section.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 4\nconversion:\n  extra_section:\n    key: value\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown sections"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_non_dict_mistral_ocr(test_project) -> None:
    """Verifies that config service rejects non dict mistral ocr.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 4\nconversion:\n  mistral_ocr: invalid\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"mistral_ocr.*mapping"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_unknown_mistral_ocr_keys(test_project) -> None:
    """Verifies that config service rejects unknown mistral ocr keys.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 4\nconversion:\n  mistral_ocr:\n"
        "    model: mistral-ocr-latest\n    api_key_env: MISTRAL_API_KEY\n"
        "    table_format: markdown\n    extra: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown keys"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_empty_mistral_ocr_model(test_project) -> None:
    """Verifies that config service rejects empty mistral ocr model.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 4\nconversion:\n  mistral_ocr:\n"
        "    model: ''\n    api_key_env: MISTRAL_API_KEY\n"
        "    table_format: markdown\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"mistral_ocr.model"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_non_dict_html_block(test_project) -> None:
    """Verifies that config service rejects non dict html block.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 4\nconversion:\n  html: invalid\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"html.*mapping"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_unknown_html_keys(test_project) -> None:
    """Verifies that config service rejects unknown html keys.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 4\nconversion:\n  html:\n"
        "    renderer: wkhtmltopdf\n    extra: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"html.*unknown keys"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_empty_wkhtmltopdf_path_string(test_project) -> None:
    """Verifies that config service rejects empty wkhtmltopdf path string.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 4\nconversion:\n  html:\n"
        "    renderer: wkhtmltopdf\n    wkhtmltopdf_path: ' '\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="wkhtmltopdf_path"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_non_dict_fallbacks(test_project) -> None:
    """Verifies that config service rejects non dict fallbacks.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 4\nconversion:\n  fallbacks: invalid\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"fallbacks.*mapping"):
        ConfigService(test_project.paths).load()


def test_config_service_rejects_unknown_fallback_keys(test_project) -> None:
    """Verifies that config service rejects unknown fallback keys.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.config_file.write_text(
        "version: 4\nconversion:\n  fallbacks:\n"
        "    pdf: docling\n    docx: markitdown\n"
        "    pptx: markitdown\n    html: markitdown\n    extra: foo\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"fallbacks.*unknown keys"):
        ConfigService(test_project.paths).load()
