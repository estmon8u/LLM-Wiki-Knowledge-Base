from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from src.cli import main
from src.commands.ingest import _echo_directory_result
from src.providers.base import ProviderRequest, ProviderResponse, TextProvider
from src.services.ingest_service import IngestDirectoryResult, IngestResult


class _CliFakeProvider(TextProvider):
    name = "cli-fake"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            text="Traceability is preserved through compiled source pages. [Sample]",
            model_name="cli-fake-v1",
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
        "doctor",
        "init",
        "ingest",
        "compile",
        "check",
        "show",
        "query",
        "export",
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
        result = runner.invoke(main, ["show", "status"])

        assert result.exit_code == 0
        assert "initialized: false" in result.output


def test_compile_requires_initialization() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["compile"])

        assert result.exit_code != 0
        assert "Project not initialized" in result.output


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

        ingest_result = runner.invoke(main, ["ingest", "sample.md"])
        assert ingest_result.exit_code == 0
        assert "Ingested Sample Research Note" in ingest_result.output

        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            compile_result = runner.invoke(main, ["compile"])
        assert compile_result.exit_code == 0
        assert "Compiled 1 source page(s)" in compile_result.output

        lint_result = runner.invoke(main, ["check", "lint"])
        assert lint_result.exit_code == 0
        assert "No lint issues found." in lint_result.output

        search_result = runner.invoke(main, ["query", "search", "traceability"])
        assert search_result.exit_code == 0
        assert "wiki/sources/sample-research-note.md" in search_result.output

        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            query_result = runner.invoke(
                main, ["query", "ask", "How", "does", "the", "wiki", "help?"]
            )
        assert query_result.exit_code == 0
        assert "Citations:" in query_result.output
        assert "wiki/sources/sample-research-note.md" in query_result.output

        export_result = runner.invoke(main, ["export", "vault"])
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

        ingest_result = runner.invoke(main, ["ingest", "sample.html"])
        assert ingest_result.exit_code == 0
        assert "Ingested HTML Research Note" in ingest_result.output
        assert "raw/normalized/html-research-note.md" in ingest_result.output

        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            compile_result = runner.invoke(main, ["compile"])
        assert compile_result.exit_code == 0
        assert "Compiled 1 source page(s)" in compile_result.output

        search_result = runner.invoke(main, ["query", "search", "traceability"])
        assert search_result.exit_code == 0
        assert "wiki/sources/html-research-note.md" in search_result.output


def test_search_and_query_show_empty_messages_when_no_results() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        search_result = runner.invoke(main, ["query", "search", "missing-topic"])
        query_result = runner.invoke(main, ["query", "ask", "missing-topic"])

        assert search_result.exit_code == 0
        assert "No wiki pages matched that query." in search_result.output
        assert query_result.exit_code != 0
        assert "requires a configured provider" in query_result.output


def test_search_and_query_require_terms() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        search_result = runner.invoke(main, ["query", "search"])
        query_result = runner.invoke(main, ["query", "ask"])

        assert search_result.exit_code != 0
        assert "Provide at least one search term." in search_result.output
        assert query_result.exit_code != 0
        assert "Provide a question to answer." in query_result.output


def test_ingest_reports_click_error_for_unsupported_file_type() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.bin").write_text("not a supported source", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["ingest", "sample.bin"])

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
        assert "Ingested Added Sample" in result.output
        assert "- slug: added-sample" in result.output
        assert Path("raw/sources/added-sample.md").exists()


def test_add_alias_recursively_ingests_directory_by_default() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("mydir/alpha.md").parent.mkdir(parents=True, exist_ok=True)
        Path("mydir/alpha.md").write_text("# Alpha\n\nAlpha body.\n", encoding="utf-8")

        result = runner.invoke(main, ["add", "mydir"])

        assert result.exit_code == 0
        assert "Processed 1 supported source file(s)" in result.output
        assert "- created: 1" in result.output


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
        assert "Processed 2 supported source file(s)" in result.output
        assert "- created: 2" in result.output
        assert "- duplicates skipped: 0" in result.output
        assert "- ingested: alpha (raw/sources/alpha.md)" in result.output
        assert "- ingested: beta-title (raw/sources/beta-title.txt)" in result.output
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
        assert "- created: 1" in result.output
        assert "- duplicates skipped: 1" in result.output
        assert "- duplicate: shared" in result.output


def test_echo_directory_result_ignores_missing_source_entries(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr("click.echo", captured.append)
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

    assert captured == [
        "Processed 2 supported source file(s) under bulk",
        "- created: 1",
        "- duplicates skipped: 1",
    ]


def test_lint_returns_nonzero_when_errors_exist() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("wiki/sources").mkdir(parents=True, exist_ok=True)
        Path("wiki/sources/bad.md").write_text(
            "# Bad\n\n[[Missing Target]]\n", encoding="utf-8"
        )

        result = runner.invoke(main, ["check", "lint"])

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

        result = runner.invoke(main, ["check", "lint"])

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

        result = runner.invoke(main, ["check", "lint"])

        assert "invalid-field-type" in result.output
        assert "invalid-date-format" in result.output
        assert "empty-page" in result.output


def test_diff_requires_initialization() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["show", "diff"])

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
        assert runner.invoke(main, ["ingest", "sample.md"]).exit_code == 0

        diff_before = runner.invoke(main, ["show", "diff"])
        assert diff_before.exit_code == 0
        assert "[NEW]" in diff_before.output
        assert "new: 1" in diff_before.output

        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            assert runner.invoke(main, ["compile"]).exit_code == 0

        diff_after = runner.invoke(main, ["show", "diff"])
        assert diff_after.exit_code == 0
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
        main, ["--project-root", str(tmp_path), "ingest", str(source_path)]
    )
    status_result = runner.invoke(
        main, ["--project-root", str(tmp_path), "show", "status"]
    )

    assert init_result.exit_code == 0
    assert ingest_result.exit_code == 0
    assert status_result.exit_code == 0
    assert "source_count: 1" in status_result.output


def test_query_save_prompt_creates_analysis_page() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nTraceability and citation evidence.\n",
            encoding="utf-8",
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["ingest", "sample.md"]).exit_code == 0
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            assert runner.invoke(main, ["compile"]).exit_code == 0

        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            result = runner.invoke(
                main,
                ["query", "ask", "How", "does", "traceability", "work?"],
                input="y\n",
            )

        assert result.exit_code == 0
        assert "Saved analysis page:" in result.output
        assert Path("wiki/concepts").exists()
        analysis_files = list(Path("wiki/concepts").glob("*.md"))
        assert len(analysis_files) == 1
        content = analysis_files[0].read_text(encoding="utf-8")
        assert "type: analysis" in content


def test_review_command_requires_provider() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["check", "review"])

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

        result = runner.invoke(main, ["check", "review"])

        assert result.exit_code != 0
        assert "requires a configured provider" in result.output


def test_review_adversarial_flag_without_provider_fails() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("wiki/sources").mkdir(parents=True, exist_ok=True)
        Path("wiki/sources/alpha.md").write_text(
            "# Alpha\n\n## Timeline\n\nIn 2026 the workflow stores source hashes.\n",
            encoding="utf-8",
        )
        Path("wiki/sources/beta.md").write_text(
            "# Beta\n\n## Timeline\n\nIn 2026 the workflow exports vault files.\n",
            encoding="utf-8",
        )

        result = runner.invoke(main, ["check", "review", "--adversarial"])

        assert result.exit_code != 0
        assert "requires a configured provider" in result.output


def test_review_requires_initialization() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["check", "review"])

        assert result.exit_code != 0
        assert "Project not initialized" in result.output


# --- P3 CLI-level tests: user-facing behavior ---


def test_lint_verbose_flag_does_not_crash() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["--verbose", "check", "lint"])

        assert result.exit_code == 0
        assert "No lint issues found." in result.output


def test_query_piped_input_skips_save_confirm() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nTraceability evidence.\n", encoding="utf-8"
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["ingest", "sample.md"]).exit_code == 0
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            assert runner.invoke(main, ["compile"]).exit_code == 0

        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            result = runner.invoke(
                main,
                ["query", "ask", "traceability"],
                input="\n",
            )

        assert result.exit_code == 0
        assert "Saved analysis page:" not in result.output


def test_query_self_consistency_flag_requires_provider() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nTraceability evidence.\n", encoding="utf-8"
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["ingest", "sample.md"]).exit_code == 0
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            assert runner.invoke(main, ["compile"]).exit_code == 0

        result = runner.invoke(
            main,
            ["query", "ask", "--self-consistency", "3", "traceability"],
            input="\n",
        )

        assert result.exit_code != 0
        assert "requires a configured provider" in result.output


def test_ingest_recursively_ingests_directory_by_default() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("mydir/alpha.md").parent.mkdir(parents=True, exist_ok=True)
        Path("mydir/alpha.md").write_text("# Alpha\n\nAlpha body.\n", encoding="utf-8")

        result = runner.invoke(main, ["ingest", "mydir"])

        assert result.exit_code == 0
        assert "Processed 1 supported source file(s)" in result.output
        assert "- created: 1" in result.output


def test_ingest_recursive_directory_requires_supported_files() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("bulk").mkdir()
        Path("bulk/ignored.bin").write_text("ignore me", encoding="utf-8")

        result = runner.invoke(main, ["ingest", "bulk"])

        assert result.exit_code != 0
        assert "No supported source files found under directory" in result.output


def test_export_vault_on_empty_wiki_succeeds() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["export", "vault"])

        assert result.exit_code == 0
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
        assert runner.invoke(main, ["ingest", "sample.md"]).exit_code == 0
        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            assert runner.invoke(main, ["compile"]).exit_code == 0

        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            result = runner.invoke(
                main,
                ["--provider", "openai", "query", "ask", "traceability"],
            )

        assert result.exit_code == 0
        assert "[mode: provider:" in result.output


def test_provider_override_flag_rejects_invalid_name() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["--provider", "invalid", "show", "status"])

        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid" in result.output.lower()
