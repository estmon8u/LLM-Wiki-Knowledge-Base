from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from kb.cli import main


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
        "init",
        "ingest",
        "compile",
        "lint",
        "search",
        "query",
        "status",
        "export-vault",
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

        compile_result = runner.invoke(main, ["compile"])
        assert compile_result.exit_code == 0
        assert "Compiled 1 source page(s)" in compile_result.output

        lint_result = runner.invoke(main, ["lint"])
        assert lint_result.exit_code == 0
        assert "No lint issues found." in lint_result.output

        search_result = runner.invoke(main, ["search", "traceability"])
        assert search_result.exit_code == 0
        assert "wiki/sources/sample-research-note.md" in search_result.output

        query_result = runner.invoke(
            main, ["query", "How", "does", "the", "wiki", "help?"]
        )
        assert query_result.exit_code == 0
        assert "Citations:" in query_result.output
        assert "wiki/sources/sample-research-note.md" in query_result.output

        export_result = runner.invoke(main, ["export-vault"])
        assert export_result.exit_code == 0
        assert Path("vault/obsidian/sources/sample-research-note.md").exists()


def test_search_and_query_show_empty_messages_when_no_results() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        search_result = runner.invoke(main, ["search", "missing-topic"])
        query_result = runner.invoke(main, ["query", "missing-topic"])

        assert search_result.exit_code == 0
        assert "No wiki pages matched that query." in search_result.output
        assert query_result.exit_code == 0
        assert (
            "No compiled wiki pages matched that question yet." in query_result.output
        )


def test_search_and_query_require_terms() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        search_result = runner.invoke(main, ["search"])
        query_result = runner.invoke(main, ["query"])

        assert search_result.exit_code != 0
        assert "Provide at least one search term." in search_result.output
        assert query_result.exit_code != 0
        assert "Provide a question to answer." in query_result.output


def test_ingest_reports_click_error_for_unsupported_file_type() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.pdf").write_text("not really a pdf", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["ingest", "sample.pdf"])

        assert result.exit_code != 0
        assert "Only markdown and text files are supported" in result.output


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
    status_result = runner.invoke(main, ["--project-root", str(tmp_path), "status"])

    assert init_result.exit_code == 0
    assert ingest_result.exit_code == 0
    assert status_result.exit_code == 0
    assert "source_count: 1" in status_result.output
