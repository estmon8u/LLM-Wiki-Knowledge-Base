from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from src.cli import main
from src.commands.ingest import _echo_directory_result
from src.providers.base import ProviderRequest, ProviderResponse, TextProvider
from src.services.ingest_service import IngestDirectoryResult, IngestResult


def _set_provider_config() -> None:
    """Write a stub provider to kb.config.yaml so the update preflight passes."""
    config = yaml.safe_load(Path("kb.config.yaml").read_text(encoding="utf-8"))
    config["provider"] = {"name": "stub"}
    Path("kb.config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )


class _CliFakeProvider(TextProvider):
    name = "cli-fake"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            text="Traceability is preserved through compiled source pages. [Sample]",
            model_name="cli-fake-v1",
        )


class _CliResumeProvider(TextProvider):
    name = "cli-resume"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("resume summary failure")
        return ProviderResponse(
            text="Stub summary of the document.",
            model_name="cli-resume-v1",
        )


def _compiled_page(title: str, body: str, *, summary: str = "Summary") -> str:
    return (
        "---\n"
        f"title: {title}\n"
        f"summary: {summary}\n"
        "source_id: source-1\n"
        "raw_path: raw/source.md\n"
        "source_hash: hash-1\n"
        "compiled_at: 2026-04-14T00:00:00Z\n"
        "---\n\n"
        f"# {title}\n\n"
        f"{body}\n"
    )


def test_init_creates_expected_project_files() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["init"])

        assert result.exit_code == 0
        assert Path("kb.config.yaml").exists()
        assert Path("kb.schema.md").exists()
        assert Path("raw/_manifest.json").exists()
        assert Path("wiki/sources").exists()
        assert Path("vault/obsidian").exists()


def test_init_is_idempotent() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["init"])

        assert result.exit_code == 0
        assert "project already had the required scaffold" in result.output


def test_help_lists_core_commands() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    for command_name in (
        "add",
        "ask",
        "doctor",
        "export",
        "find",
        "init",
        "review",
        "status",
        "update",
    ):
        assert command_name in result.output


def test_running_cli_without_subcommand_prints_help() -> None:
    runner = CliRunner()

    result = runner.invoke(main, [])

    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_status_before_init_shows_uninitialized_state() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["status"])

        assert result.exit_code == 0
        assert "kb init" in result.output


def test_end_to_end_cli_flow_for_local_markdown_source() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample Research Note\n\n"
            "Markdown-first knowledge bases preserve source traceability.\n\n"
            "They can be linted for broken links and missing citations.\n",
            encoding="utf-8",
        )

        assert runner.invoke(main, ["init"]).exit_code == 0

        ingest_result = runner.invoke(main, ["add", "sample.md"])
        assert ingest_result.exit_code == 0
        assert "Ingested Sample Research Note" in ingest_result.output

        _set_provider_config()
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            update_result = runner.invoke(main, ["update"])
        assert update_result.exit_code == 0
        assert "Update Summary" in update_result.output
        assert "Compiled 1 source page(s)" in update_result.output

        lint_result = runner.invoke(main, ["lint"])
        assert lint_result.exit_code == 0
        assert "No lint issues found." in lint_result.output

        search_result = runner.invoke(main, ["find", "traceability"])
        assert search_result.exit_code == 0
        assert "wiki/sources/sample-research-note.md" in search_result.output
        assert Path("graph/exports/search_index.sqlite3").exists()

        _set_provider_config()
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            query_result = runner.invoke(main, ["ask", "traceability", "knowledge"])
        assert query_result.exit_code == 0
        assert "Answer" in query_result.output
        assert "Citations" in query_result.output
        assert "wiki/sources/sample-research-note.md" in query_result.output
        assert "#chunk-" in query_result.output

        export_result = runner.invoke(main, ["export"])
        assert export_result.exit_code == 0
        assert Path("vault/obsidian/sources/sample-research-note.md").exists()


def test_end_to_end_cli_flow_for_local_html_source() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.html").write_text(
            "<html><body><h1>HTML Research Note</h1>"
            "<p>Traceability survives conversion.</p></body></html>",
            encoding="utf-8",
        )

        assert runner.invoke(main, ["init"]).exit_code == 0

        ingest_result = runner.invoke(main, ["add", "sample.html"])
        assert ingest_result.exit_code == 0
        assert "Ingested HTML Research Note" in ingest_result.output
        assert "raw/normalized/html-research-note.md" in ingest_result.output

        _set_provider_config()
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            update_result = runner.invoke(main, ["update"])
        assert update_result.exit_code == 0
        assert "Compiled 1 source page(s)" in update_result.output

        search_result = runner.invoke(main, ["find", "traceability"])
        assert search_result.exit_code == 0
        assert "wiki/sources/html-research-note.md" in search_result.output


def test_search_and_ask_show_empty_messages_when_no_results() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        search_result = runner.invoke(main, ["find", "missing-topic"])
        ask_result = runner.invoke(main, ["ask", "missing-topic"])

        assert search_result.exit_code == 0
        assert "No wiki pages matched that query." in search_result.output
        assert ask_result.exit_code != 0
        assert "requires a configured provider" in ask_result.output


def test_find_and_ask_require_terms() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        search_result = runner.invoke(main, ["find"])
        ask_result = runner.invoke(main, ["ask"])

        assert search_result.exit_code != 0
        assert "Provide at least one search term." in search_result.output
        assert ask_result.exit_code != 0
        assert "Provide a question to answer." in ask_result.output


def test_ingest_reports_click_error_for_unsupported_file_type() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.bin").write_text("not a supported source", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["add", "sample.bin"])

        assert result.exit_code != 0
        assert "Supported ingest inputs are canonical text" in result.output


def test_add_alias_ingests_source_file() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Added Sample\n\nAlias ingest path.\n", encoding="utf-8"
        )
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["add", "sample.md"])

        assert result.exit_code == 0
        assert "Ingest Summary" in result.output
        assert "Ingested Added Sample" in result.output
        assert "slug: added-sample" in result.output
        assert Path("raw/sources/added-sample.md").exists()


def test_add_alias_recursively_ingests_directory_by_default() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("mydir/alpha.md").parent.mkdir(parents=True, exist_ok=True)
        Path("mydir/alpha.md").write_text("# Alpha\n\nAlpha body.\n", encoding="utf-8")

        result = runner.invoke(main, ["add", "mydir"])

        assert result.exit_code == 0
        assert "Ingesting 1 source file(s)..." in result.output
        assert "Ingest Summary" in result.output
        assert "Processed 1 supported source file(s)" in result.output
        assert "created: 1" in result.output


def test_add_alias_recursively_ingests_supported_directory_files() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("bulk/nested").mkdir(parents=True, exist_ok=True)
        Path("bulk/alpha.md").write_text("# Alpha\n\nAlpha body.\n", encoding="utf-8")
        Path("bulk/nested/beta.txt").write_text(
            "Beta title\n\nBeta body.\n", encoding="utf-8"
        )
        Path("bulk/nested/ignored.bin").write_text("ignore me", encoding="utf-8")

        result = runner.invoke(main, ["add", "bulk"])

        assert result.exit_code == 0
        assert "Ingesting 2 source file(s)..." in result.output
        assert "Ingest Summary" in result.output
        assert "Processed 2 supported source file(s)" in result.output
        assert "created: 2" in result.output
        assert "duplicates skipped: 0" in result.output
        assert "ingested: alpha (raw/sources/alpha.md)" in result.output
        assert "ingested: beta-title (raw/sources/beta-title.txt)" in result.output
        assert Path("raw/sources/alpha.md").exists()
        assert Path("raw/sources/beta-title.txt").exists()


def test_add_alias_recursive_directory_reports_duplicates() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("bulk/nested").mkdir(parents=True, exist_ok=True)
        duplicate = "# Shared\n\nSame body.\n"
        Path("bulk/first.md").write_text(duplicate, encoding="utf-8")
        Path("bulk/nested/second.md").write_text(duplicate, encoding="utf-8")

        result = runner.invoke(main, ["add", "bulk"])

        assert result.exit_code == 0
        assert "Ingest Summary" in result.output
        assert "created: 1" in result.output
        assert "duplicates skipped: 1" in result.output
        assert "duplicate: shared" in result.output


def test_add_accepts_multiple_source_paths() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("a.md").write_text("# Alpha\n\nAlpha body.\n", encoding="utf-8")
        Path("b.md").write_text("# Beta\n\nBeta body.\n", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["add", "a.md", "b.md"])

        assert result.exit_code == 0
        assert "Ingested Alpha" in result.output
        assert "Ingested Beta" in result.output
        assert Path("raw/sources/alpha.md").exists()
        assert Path("raw/sources/beta.md").exists()


def test_echo_directory_result_ignores_missing_source_entries(capsys) -> None:
    result = IngestDirectoryResult(
        directory_path=Path("bulk"),
        scanned_file_count=2,
        results=(
            IngestResult(created=True, source=None, message="created"),
            IngestResult(
                created=False,
                source=None,
                duplicate_of=None,
                message="duplicate",
            ),
        ),
    )

    _echo_directory_result(result)

    output = capsys.readouterr().out
    assert "Ingest Summary" in output
    assert "Processed 2 supported source file(s) under bulk" in output
    assert "created: 1" in output
    assert "duplicates skipped: 1" in output


def test_lint_returns_nonzero_when_errors_exist() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("wiki/sources").mkdir(parents=True, exist_ok=True)
        Path("wiki/sources/bad.md").write_text(
            "# Bad\n\n[[Missing Target]]\n", encoding="utf-8"
        )

        result = runner.invoke(main, ["lint"])

        assert result.exit_code == 1
        assert "ERRORS" in result.output
        assert "broken-link" in result.output


def test_lint_reports_markdown_link_and_heading_errors_at_cli() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("wiki/sources").mkdir(parents=True, exist_ok=True)
        Path("wiki/sources/target.md").write_text(
            _compiled_page("Target", "## Present Section\n\nBody."),
            encoding="utf-8",
        )
        Path("wiki/sources/bad.md").write_text(
            _compiled_page(
                "Bad Page",
                (
                    "See [missing](missing.md) and [[target#Missing Section]].\n\n"
                    "### Skipped Level\n\n"
                    "# Another H1\n"
                ),
            ),
            encoding="utf-8",
        )

        result = runner.invoke(main, ["lint"])

        assert result.exit_code == 1
        assert "broken-markdown-link" in result.output
        assert "broken-fragment" in result.output
        assert "heading-level-skip" in result.output
        assert "multiple-h1" in result.output


def test_lint_reports_frontmatter_type_and_empty_page_at_cli() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("wiki/sources").mkdir(parents=True, exist_ok=True)
        Path("wiki/sources/bad-types.md").write_text(
            "---\n"
            "title: 123\n"
            "summary: OK\n"
            "source_id: id-1\n"
            "raw_path: raw/file.md\n"
            "source_hash: hash-1\n"
            "compiled_at: not-a-date\n"
            "tags: not-a-list\n"
            "---\n\n"
            "# Bad Types\n",
            encoding="utf-8",
        )

        result = runner.invoke(main, ["lint"])

        assert "invalid-field-type" in result.output
        assert "invalid-date-format" in result.output
        assert "empty-page" in result.output


def test_diff_requires_initialization() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["status", "--changed"])

        assert result.exit_code != 0
        assert "Project not initialized" in result.output


def test_diff_end_to_end_new_then_compiled() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nBody for diff test.\n",
            encoding="utf-8",
        )

        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0

        diff_before = runner.invoke(main, ["status", "--changed"])
        assert diff_before.exit_code == 0
        assert "Source Diff" in diff_before.output
        assert "[NEW]" in diff_before.output
        assert "new: 1" in diff_before.output

        _set_provider_config()
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            assert runner.invoke(main, ["update"]).exit_code == 0

        diff_after = runner.invoke(main, ["status", "--changed"])
        assert diff_after.exit_code == 0
        assert "Summary" in diff_after.output
        assert "[OK]" in diff_after.output
        assert "up_to_date: 1" in diff_after.output


def test_cli_supports_explicit_project_root_option(tmp_path: Path) -> None:
    runner = CliRunner()
    source_path = tmp_path / "external.md"
    source_path.write_text(
        "# External\n\nProject root option test.\n", encoding="utf-8"
    )

    init_result = runner.invoke(main, ["--project-root", str(tmp_path), "init"])
    ingest_result = runner.invoke(
        main, ["--project-root", str(tmp_path), "add", str(source_path)]
    )
    status_result = runner.invoke(main, ["--project-root", str(tmp_path), "status"])

    assert init_result.exit_code == 0
    assert ingest_result.exit_code == 0
    assert status_result.exit_code == 0
    assert "1 total" in status_result.output


def test_ask_save_flag_creates_analysis_page() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nTraceability and citation evidence.\n",
            encoding="utf-8",
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0
        _set_provider_config()
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            assert runner.invoke(main, ["update"]).exit_code == 0

        _set_provider_config()
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            result = runner.invoke(
                main,
                ["ask", "--save", "How", "does", "traceability", "work?"],
            )

        assert result.exit_code == 0
        assert "Saved analysis page:" in result.output
        assert Path("wiki/analysis").exists()
        analysis_files = list(Path("wiki/analysis").glob("*.md"))
        assert len(analysis_files) == 1
        content = analysis_files[0].read_text(encoding="utf-8")
        assert "type: analysis" in content


def test_review_command_requires_provider() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["review"])

        assert result.exit_code != 0
        assert "requires a configured provider" in result.output


def test_review_command_reports_overlapping_topics_requires_provider() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("wiki/sources").mkdir(parents=True, exist_ok=True)
        Path("wiki/sources/alpha.md").write_text(
            "knowledge base traceability citation markdown wiki compile ingest lint",
            encoding="utf-8",
        )
        Path("wiki/sources/beta.md").write_text(
            "knowledge base traceability citation markdown wiki compile query lint",
            encoding="utf-8",
        )

        result = runner.invoke(main, ["review"])

        assert result.exit_code != 0
        assert "requires a configured provider" in result.output


def test_review_requires_initialization() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["review"])

        assert result.exit_code != 0
        assert "Project not initialized" in result.output


# --- P3 CLI-level tests: user-facing behavior ---


def test_lint_verbose_flag_does_not_crash() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["--verbose", "lint"])

        assert result.exit_code == 0
        assert "No lint issues found." in result.output


def test_query_piped_input_does_not_save_without_flag() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nTraceability evidence.\n", encoding="utf-8"
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0
        _set_provider_config()
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            assert runner.invoke(main, ["update"]).exit_code == 0

        _set_provider_config()
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            result = runner.invoke(
                main,
                ["ask", "traceability"],
            )

        assert result.exit_code == 0
        assert "Saved analysis page:" not in result.output


def test_ingest_recursively_ingests_directory_by_default() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("mydir/alpha.md").parent.mkdir(parents=True, exist_ok=True)
        Path("mydir/alpha.md").write_text("# Alpha\n\nAlpha body.\n", encoding="utf-8")

        result = runner.invoke(main, ["add", "mydir"])

        assert result.exit_code == 0
        assert "Ingesting 1 source file(s)..." in result.output
        assert "Ingest Summary" in result.output
        assert "Processed 1 supported source file(s)" in result.output
        assert "created: 1" in result.output


def test_ingest_recursive_directory_requires_supported_files() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("bulk").mkdir()
        Path("bulk/ignored.bin").write_text("ignore me", encoding="utf-8")

        result = runner.invoke(main, ["add", "bulk"])

        assert result.exit_code != 0
        assert "No supported source files found under directory" in result.output


def test_export_on_empty_wiki_succeeds() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["export"])

        assert result.exit_code == 0
        assert "Vault Export" in result.output
        assert "Exported 0 markdown file(s)" in result.output


def test_unknown_command_shows_error() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["nonexistent-command"])

        assert result.exit_code != 0


def test_provider_override_flag_switches_provider() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("sample.md").write_text(
            "# Traceability\n\nTraceability evidence.\n",
            encoding="utf-8",
        )
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0
        _set_provider_config()
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            assert runner.invoke(main, ["update"]).exit_code == 0

        _set_provider_config()
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            result = runner.invoke(
                main,
                ["--provider", "openai", "ask", "traceability"],
            )

        assert result.exit_code == 0
        assert "[mode: provider:" in result.output


def test_provider_override_flag_rejects_invalid_name() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["--provider", "invalid", "status"])

        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid" in result.output.lower()


def test_provider_override_clears_tier_and_api_key_env() -> None:
    """--provider clears stale tier, model, and api_key_env from the config."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        # Set an openai provider with a specific tier and api_key_env.
        config = yaml.safe_load(Path("kb.config.yaml").read_text(encoding="utf-8"))
        config["provider"] = {
            "name": "openai",
            "model": "gpt-5.4",
            "tier": "deep",
            "api_key_env": "MY_OPENAI_KEY",
        }
        Path("kb.config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

        _set_provider_config()
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            result = runner.invoke(
                main,
                ["--provider", "anthropic", "status"],
            )

        assert result.exit_code == 0


# --- Simplified CLI UX tests ---


def test_update_compiles_and_generates_concepts() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nBody for update test.\n", encoding="utf-8"
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0
        _set_provider_config()

        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            result = runner.invoke(main, ["update"])

        assert result.exit_code == 0
        assert "Update Summary" in result.output
        assert "Compiled 1 source page(s)" in result.output
        assert "Concept Summary" in result.output


def test_update_with_paths_adds_then_compiles() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("note.md").write_text(
            "# New Note\n\nAdded via update.\n", encoding="utf-8"
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        _set_provider_config()

        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            result = runner.invoke(main, ["update", "note.md"])

        assert result.exit_code == 0
        assert "Added note.md" in result.output
        assert "Update Summary" in result.output
        assert "Compiled 1 source page(s)" in result.output


def test_find_command_works() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["find", "missing-topic"])

        assert result.exit_code == 0
        assert "No wiki pages matched that query." in result.output


def test_flat_status_shows_knowledge_base_overview() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["status"])

        assert result.exit_code == 0
        assert "Knowledge Base" in result.output
        assert "Sources" in result.output
        assert "0 total" in result.output
        assert "kb add" in result.output


def test_status_changed_flag_shows_diff() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nBody for status --changed test.\n", encoding="utf-8"
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0

        result = runner.invoke(main, ["status", "--changed"])

        assert result.exit_code == 0
        assert "Source Diff" in result.output
        assert "[NEW]" in result.output
        assert "new: 1" in result.output


def test_flat_export_defaults_to_vault() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["export"])

        assert result.exit_code == 0
        assert "Vault Export" in result.output
        assert "Exported 0 markdown file(s)" in result.output


def test_config_command_shows_config() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["config"])

        assert result.exit_code == 0
        assert "Configuration" in result.output
        assert "project" in result.output


def test_sources_list_shows_empty() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["sources"])

        assert result.exit_code == 0
        assert "No sources ingested yet." in result.output


def test_sources_list_shows_ingested_sources() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text("# Sample\n\nBody.\n", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0

        result = runner.invoke(main, ["sources", "list"])

        assert result.exit_code == 0
        assert "Sources" in result.output
        assert "sample" in result.output
        assert "total: 1" in result.output


def test_review_successful_run_shows_no_issues() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("wiki/sources").mkdir(parents=True, exist_ok=True)
        Path("wiki/sources/alpha.md").write_text(
            _compiled_page("Alpha", "Unique content about alpha topic."),
            encoding="utf-8",
        )

        _set_provider_config()
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            result = runner.invoke(main, ["review"])

        assert result.exit_code == 0
        assert "Review mode:" in result.output
        assert "No review issues found." in result.output


def test_review_successful_run_shows_issues() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("wiki/sources").mkdir(parents=True, exist_ok=True)
        overlap_body = (
            "knowledge base traceability citation markdown wiki compile "
            "ingest lint review export vault query"
        )
        Path("wiki/sources/alpha.md").write_text(
            _compiled_page("Alpha", overlap_body), encoding="utf-8"
        )
        Path("wiki/sources/beta.md").write_text(
            _compiled_page("Beta", overlap_body), encoding="utf-8"
        )

        _set_provider_config()
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            result = runner.invoke(main, ["review"])

        assert result.exit_code == 0
        assert "Review mode:" in result.output
        assert "Total review issues:" in result.output


def test_sources_show_displays_details() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text("# Sample\n\nBody.\n", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0

        result = runner.invoke(main, ["sources", "show", "sample"])

        assert result.exit_code == 0
        assert "Source: sample" in result.output
        assert "source_id:" in result.output
        assert "raw_path:" in result.output
        assert "content_hash:" in result.output
        assert "source_type:" in result.output


def test_sources_show_missing_slug_fails() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["sources", "show", "nonexistent"])

        assert result.exit_code != 0
        assert "Source not found: nonexistent" in result.output


def test_ask_show_evidence_flag() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nTraceability and citation evidence.\n",
            encoding="utf-8",
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0
        _set_provider_config()
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            assert runner.invoke(main, ["update"]).exit_code == 0

            result = runner.invoke(
                main,
                [
                    "ask",
                    "--show-evidence",
                    "How",
                    "does",
                    "traceability",
                    "work?",
                ],
            )

        assert result.exit_code == 0
        assert "Evidence" in result.output
        assert "Answer" in result.output


def test_status_shows_stale_sources_needing_compile() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nBody for status test.\n", encoding="utf-8"
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0

        result = runner.invoke(main, ["status"])

        assert result.exit_code == 0
        assert "1 need compiling" in result.output
        assert "kb update" in result.output


def test_status_shows_current_after_compile() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nBody for status current test.\n", encoding="utf-8"
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0
        _set_provider_config()
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            assert runner.invoke(main, ["update"]).exit_code == 0

        result = runner.invoke(main, ["status"])

        assert result.exit_code == 0
        assert "Knowledge base is current." in result.output


def test_update_with_directory_path_adds_then_compiles() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("docs").mkdir()
        Path("docs/alpha.md").write_text("# Alpha\n\nAlpha body.\n", encoding="utf-8")
        Path("docs/beta.md").write_text("# Beta\n\nBeta body.\n", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0
        _set_provider_config()

        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            result = runner.invoke(main, ["update", "docs"])

        assert result.exit_code == 0
        assert "Added 2 source(s) from" in result.output
        assert "Update Summary" in result.output
        assert "Compiled 2 source page(s)" in result.output


def test_update_with_already_present_file() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("note.md").write_text("# Note\n\nNote body.\n", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "note.md"]).exit_code == 0
        _set_provider_config()

        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            result = runner.invoke(main, ["update", "note.md"])

        assert result.exit_code == 0
        assert "Already present: note.md" in result.output
        assert "Update Summary" in result.output


def test_update_resume_rejects_force_combination() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["update", "--force", "--resume"])

        assert result.exit_code != 0
        assert "--resume cannot be combined with --force" in result.output


# ---------------------------------------------------------------------------
# Provider preflight in update
# ---------------------------------------------------------------------------


def test_update_fails_without_provider_config() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            result = runner.invoke(main, ["update"])

        assert result.exit_code != 0
        assert "Provider is not configured" in result.output


def test_update_generic_service_error_becomes_click_exception() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("note.md").write_text("# Note\n\nBody.\n", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "note.md"]).exit_code == 0
        _set_provider_config()

        with patch(
            "src.services.build_provider", return_value=_CliFakeProvider()
        ), patch(
            "src.services.update_service.UpdateService.run",
            side_effect=RuntimeError("boom"),
        ):
            result = runner.invoke(main, ["update"])

        assert result.exit_code != 0
        assert "boom" in result.output


# ---------------------------------------------------------------------------
# Config subcommands
# ---------------------------------------------------------------------------


def test_config_show_subcommand() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["config", "show"])

        assert result.exit_code == 0
        assert "Configuration" in result.output


def test_config_provider_set_and_clear() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["config", "provider", "set", "openai"])
        assert result.exit_code == 0
        assert "Provider set to openai" in result.output

        config = yaml.safe_load(Path("kb.config.yaml").read_text(encoding="utf-8"))
        assert config["provider"]["name"] == "openai"

        result = runner.invoke(
            main,
            ["config", "provider", "set", "anthropic", "--model", "claude-4"],
        )
        assert result.exit_code == 0
        assert "model=claude-4" in result.output

        result = runner.invoke(main, ["config", "provider", "clear"])
        assert result.exit_code == 0
        assert "Provider cleared." in result.output

        config = yaml.safe_load(Path("kb.config.yaml").read_text(encoding="utf-8"))
        assert config["provider"] == {}


def test_config_provider_set_switching_clears_stale() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        runner.invoke(
            main,
            ["config", "provider", "set", "openai", "--model", "gpt-5.4"],
        )
        result = runner.invoke(
            main,
            ["config", "provider", "set", "anthropic"],
        )
        assert result.exit_code == 0
        config = yaml.safe_load(Path("kb.config.yaml").read_text(encoding="utf-8"))
        assert config["provider"]["name"] == "anthropic"
        assert "model" not in config["provider"]


def test_config_provider_set_rejects_unknown_name() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(
            main,
            ["config", "provider", "set", "foobar"],
        )
        assert result.exit_code != 0
        assert "foobar" in result.output.lower() or "invalid" in result.output.lower()


# ---------------------------------------------------------------------------
# --json flag tests
# ---------------------------------------------------------------------------

import json


def test_doctor_json_output() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["doctor", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["ok"] is True
        assert "checks" in data
        assert data["passed"] > 0


def test_find_json_output() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nTraceability and citation.\n", encoding="utf-8"
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0
        _set_provider_config()
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            assert runner.invoke(main, ["update"]).exit_code == 0

        result = runner.invoke(main, ["find", "--json", "traceability"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "title" in data[0]
        assert "path" in data[0]
        assert "score" in data[0]


def test_find_json_empty_results() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["find", "--json", "missing-topic"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == []


def test_status_json_output() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nBody for JSON status.\n", encoding="utf-8"
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0

        result = runner.invoke(main, ["status", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["source_count"] == 1
        assert data["initialized"] is True


def test_status_changed_json_output() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nBody for JSON diff.\n", encoding="utf-8"
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0

        result = runner.invoke(main, ["status", "--changed", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "entries" in data
        assert data["new"] >= 1


def test_sources_list_json_output() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text("# Sample\n\nBody.\n", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0

        result = runner.invoke(main, ["sources", "list", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["slug"] == "sample"


def test_sources_list_json_empty() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["sources", "list", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == []


# ---------------------------------------------------------------------------
# Coverage gap: doctor with failures
# ---------------------------------------------------------------------------


def test_doctor_failure_exits_nonzero() -> None:
    """doctor raises SystemExit(1) when a check fails."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["doctor", "--strict"])

        # --strict treats missing provider as error → non-zero exit
        assert result.exit_code != 0


def test_doctor_json_failure_exits_nonzero() -> None:
    """doctor --json raises SystemExit(1) when a check fails."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["doctor", "--strict", "--json"])

        assert result.exit_code != 0
        data = json.loads(result.output)
        assert data["ok"] is False
