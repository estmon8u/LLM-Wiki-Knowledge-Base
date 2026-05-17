"""Tests for test cli.

This module belongs to `tests.test_cli` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import yaml
from click.testing import CliRunner

from graphwiki_kb.cli import main
from graphwiki_kb.commands.ingest import _echo_directory_result
from graphwiki_kb.providers.base import ProviderRequest, ProviderResponse, TextProvider
from graphwiki_kb.services.ingest_service import IngestDirectoryResult, IngestResult


def _set_provider_config() -> None:
    """Write a stub provider to kb.config.yaml so the update preflight passes."""
    config = yaml.safe_load(Path("kb.config.yaml").read_text(encoding="utf-8"))
    config["provider"] = {"name": "stub"}
    Path("kb.config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )


class _CliFakeProvider(TextProvider):
    """Represents cli fake provider behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    name = "cli-fake"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate.

        Args:
            request: Request value used by the operation.

        Returns:
            ProviderResponse produced by the operation.
        """
        if request.response_schema_name == "kb_review_report":
            return ProviderResponse(text='{"issues": []}', model_name="cli-fake-v1")
        if request.response_schema_name == "kb_query_answer":
            match = re.search(r"^citation_ref:\s*(.+)$", request.prompt, re.MULTILINE)
            ref = match.group(1).strip() if match else "wiki/sources/sample.md#chunk-0"
            return ProviderResponse(
                text=json.dumps(
                    {
                        "answer_markdown": "Traceability is preserved through compiled source pages.",
                        "claims": [
                            {
                                "text": "Traceability is preserved through compiled source pages.",
                                "citation_refs": [ref],
                            }
                        ],
                        "citations": [{"ref": ref, "title": "Sample"}],
                        "insufficient_evidence": False,
                    }
                ),
                model_name="cli-fake-v1",
            )
        return ProviderResponse(
            text="Traceability is preserved through compiled source pages. [Sample]",
            model_name="cli-fake-v1",
        )


class _CliResumeProvider(TextProvider):
    """Represents cli resume provider behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    name = "cli-resume"

    def __init__(self) -> None:
        """Initializes the instance."""
        self.calls = 0

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate.

        Args:
            request: Request value used by the operation.

        Returns:
            ProviderResponse produced by the operation.
        """
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("resume summary failure")
        return ProviderResponse(
            text="Stub summary of the document.",
            model_name="cli-resume-v1",
        )


def _compiled_page(title: str, body: str, *, summary: str = "Summary") -> str:
    """Handles compiled page.

    Args:
        title: Title value used by the operation.
        body: Body value used by the operation.
        summary: Summary value used by the operation.

    Returns:
        str produced by the operation.
    """
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
    """Verifies that init creates expected project files."""
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
    """Verifies that init is idempotent."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["init"])

        assert result.exit_code == 0
        assert "project already had the required scaffold" in result.output


def test_init_regenerates_malformed_config() -> None:
    """Verifies init can recover a malformed config file."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("kb.config.yaml").write_text("version: [\n", encoding="utf-8")

        result = runner.invoke(main, ["init"])

        assert result.exit_code == 0
        assert "kb.config.yaml (regenerated" in result.output
        assert "backup: kb.config.yaml.bak." in result.output
        assert list(Path().glob("kb.config.yaml.bak.*"))
        assert isinstance(
            yaml.safe_load(Path("kb.config.yaml").read_text(encoding="utf-8")),
            dict,
        )


def test_help_lists_core_commands() -> None:
    """Verifies that help lists core commands."""
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
        "legacy",
        "review",
        "status",
        "update",
    ):
        assert command_name in result.output


def test_subcommand_help_works_with_malformed_config() -> None:
    """Verifies command help does not require loading a valid project config."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("kb.config.yaml").write_text("version: [\n", encoding="utf-8")

        result = runner.invoke(main, ["update", "--help"])

        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "--graph-only" in result.output


def test_running_cli_without_subcommand_prints_help() -> None:
    """Verifies that running cli without subcommand prints help."""
    runner = CliRunner()

    result = runner.invoke(main, [])

    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_status_before_init_shows_uninitialized_state() -> None:
    """Verifies that status before init shows uninitialized state."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["status"])

        assert result.exit_code == 0
        assert "kb init" in result.output


def test_end_to_end_cli_flow_for_local_markdown_source() -> None:
    """Verifies that end to end cli flow for local markdown source."""
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
        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            update_result = runner.invoke(main, ["update", "--no-graph"])
        assert update_result.exit_code == 0
        assert "Update Summary" in update_result.output
        assert "Mode: wiki-only" in update_result.output
        assert "Compiled 1 source page(s)" in update_result.output

        lint_result = runner.invoke(main, ["lint"])
        assert lint_result.exit_code == 0
        assert "No lint issues found." in lint_result.output

        search_result = runner.invoke(main, ["legacy", "find", "traceability"])
        assert search_result.exit_code == 0
        assert "wiki/sources/sample-research-note.md" in search_result.output
        assert Path("graph/exports/search_index.sqlite3").exists()

        _set_provider_config()
        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            query_result = runner.invoke(
                main, ["legacy", "ask", "traceability", "knowledge"]
            )
        assert query_result.exit_code == 0
        assert "Answer" in query_result.output
        assert "Citations" in query_result.output
        assert "wiki/sources/sample-research-note.md" in query_result.output
        assert "#chunk-" in query_result.output
        assert "retriever: legacy-fts" in query_result.output

        export_result = runner.invoke(main, ["export"])
        assert export_result.exit_code == 0
        assert Path("vault/obsidian/sources/sample-research-note.md").exists()


def test_end_to_end_cli_flow_for_local_html_source() -> None:
    """Verifies that end to end cli flow for local html source."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        offline_env = {"MISTRAL_API_KEY": ""}
        Path("sample.html").write_text(
            "<html><body><h1>HTML Research Note</h1>"
            "<p>Traceability survives conversion.</p></body></html>",
            encoding="utf-8",
        )

        assert runner.invoke(main, ["init"]).exit_code == 0

        ingest_result = runner.invoke(main, ["add", "sample.html"], env=offline_env)
        assert ingest_result.exit_code == 0
        assert "Ingested HTML Research Note" in ingest_result.output
        assert "raw/normalized/html-research-note.md" in ingest_result.output

        _set_provider_config()
        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            update_result = runner.invoke(main, ["update", "--no-graph"])
        assert update_result.exit_code == 0
        assert "Mode: wiki-only" in update_result.output
        assert "Compiled 1 source page(s)" in update_result.output

        search_result = runner.invoke(main, ["legacy", "find", "traceability"])
        assert search_result.exit_code == 0
        assert "wiki/sources/html-research-note.md" in search_result.output

        graph_page = Path("wiki/graph/entities/generated.md")
        graph_page.parent.mkdir(parents=True, exist_ok=True)
        graph_page.write_text(
            "---\ntitle: Generated Graph\ntype: graph_entity\n---\n\n"
            "# Generated Graph\n\nuniquegraphtoken appears only here.\n",
            encoding="utf-8",
        )
        find_result = runner.invoke(main, ["find", "uniquegraphtoken"])
        legacy_find_result = runner.invoke(main, ["legacy", "find", "uniquegraphtoken"])
        assert find_result.exit_code == 0
        assert "wiki/graph/entities/generated.md" in find_result.output
        assert legacy_find_result.exit_code == 0
        assert "No wiki pages matched that query." in legacy_find_result.output


def test_legacy_search_empty_and_top_level_find_searches_wiki() -> None:
    """Verifies legacy and top-level find both report empty wiki search results."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        search_result = runner.invoke(main, ["legacy", "find", "missing-topic"])
        find_result = runner.invoke(main, ["find", "missing-topic"])
        ask_result = runner.invoke(main, ["ask", "missing-topic"])

        assert search_result.exit_code == 0
        assert "No wiki pages matched that query." in search_result.output
        assert find_result.exit_code == 0
        assert (
            "No graph artifacts or wiki pages matched that query." in find_result.output
        )
        assert ask_result.exit_code != 0
        assert "kb update" in ask_result.output
        assert "kb legacy ask" not in ask_result.output


def test_find_and_ask_require_terms() -> None:
    """Verifies that find and ask require terms."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        search_result = runner.invoke(main, ["find"])
        ask_result = runner.invoke(main, ["ask"])
        legacy_search_result = runner.invoke(main, ["legacy", "find"])
        legacy_ask_result = runner.invoke(main, ["legacy", "ask"])

        assert search_result.exit_code != 0
        assert "Provide at least one search term." in search_result.output
        assert ask_result.exit_code != 0
        assert "Provide a question to answer" in ask_result.output
        assert legacy_search_result.exit_code != 0
        assert "Provide at least one search term." in legacy_search_result.output
        assert legacy_ask_result.exit_code != 0
        assert "Provide a question to answer." in legacy_ask_result.output


def test_ingest_reports_click_error_for_unsupported_file_type() -> None:
    """Verifies that ingest reports click error for unsupported file type."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.bin").write_text("not a supported source", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["add", "sample.bin"])

        assert result.exit_code != 0
        assert "Supported ingest inputs are canonical text" in result.output


def test_add_alias_ingests_source_file() -> None:
    """Verifies that add alias ingests source file."""
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
    """Verifies that add alias recursively ingests directory by default."""
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
    """Verifies that add alias recursively ingests supported directory files."""
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
    """Verifies that add alias recursive directory reports duplicates."""
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
    """Verifies that add accepts multiple source paths."""
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
    """Verifies that echo directory result ignores missing source entries.

    Args:
        capsys: Capsys value used by the operation.
    """
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
    """Verifies that lint returns nonzero when errors exist."""
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


def test_lint_json_reports_issues_and_exits_nonzero() -> None:
    """Verifies that lint json reports errors and exits nonzero."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("wiki/sources").mkdir(parents=True, exist_ok=True)
        Path("wiki/sources/bad.md").write_text(
            "---\ntype: source\nsummary: Bad\n---\n\n# Bad\n\n[[missing]]",
            encoding="utf-8",
        )

        result = runner.invoke(main, ["lint", "--json"])

        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["ok"] is False
        assert payload["error_count"] >= 1
        assert any(issue["code"] == "broken-link" for issue in payload["issues"])


def test_lint_reports_markdown_link_and_heading_errors_at_cli() -> None:
    """Verifies that lint reports markdown link and heading errors at cli."""
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
    """Verifies that lint reports frontmatter type and empty page at cli."""
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
    """Verifies that diff requires initialization."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["status", "--changed"])

        assert result.exit_code != 0
        assert "Project not initialized" in result.output


def test_diff_end_to_end_new_then_compiled() -> None:
    """Verifies that diff end to end new then compiled."""
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
        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            assert runner.invoke(main, ["update", "--no-graph"]).exit_code == 0

        diff_after = runner.invoke(main, ["status", "--changed"])
        assert diff_after.exit_code == 0
        assert "Summary" in diff_after.output
        assert "[OK]" in diff_after.output
        assert "up_to_date: 1" in diff_after.output


def test_cli_supports_explicit_project_root_option(tmp_path: Path) -> None:
    """Verifies that cli supports explicit project root option.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
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
    """Verifies that ask save flag creates analysis page."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nTraceability and citation evidence.\n",
            encoding="utf-8",
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0
        _set_provider_config()
        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            assert runner.invoke(main, ["update", "--no-graph"]).exit_code == 0

        _set_provider_config()
        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            result = runner.invoke(
                main,
                [
                    "legacy",
                    "ask",
                    "--save",
                    "How",
                    "does",
                    "traceability",
                    "work?",
                ],
            )

        assert result.exit_code == 0
        assert "Saved analysis page:" in result.output
        assert Path("wiki/analysis").exists()
        analysis_files = list(Path("wiki/analysis").glob("*.md"))
        assert len(analysis_files) == 1
        content = analysis_files[0].read_text(encoding="utf-8")
        assert "type: analysis" in content


def test_review_command_requires_provider() -> None:
    """Verifies that review command requires provider."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["review"])

        assert result.exit_code != 0
        assert "requires a configured provider" in result.output


def test_review_command_reports_overlapping_topics_requires_provider() -> None:
    """Verifies that review command reports overlapping topics requires provider."""
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
    """Verifies that review requires initialization."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["review"])

        assert result.exit_code != 0
        assert "Project not initialized" in result.output


# --- P3 CLI-level tests: user-facing behavior ---


def test_lint_verbose_flag_does_not_crash() -> None:
    """Verifies that lint verbose flag does not crash."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["--verbose", "lint"])

        assert result.exit_code == 0
        assert "No lint issues found." in result.output


def test_query_piped_input_does_not_save_without_flag() -> None:
    """Verifies that query piped input does not save without flag."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nTraceability evidence.\n", encoding="utf-8"
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0
        _set_provider_config()
        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            assert runner.invoke(main, ["update", "--no-graph"]).exit_code == 0

        _set_provider_config()
        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            result = runner.invoke(
                main,
                ["legacy", "ask", "traceability"],
            )

        assert result.exit_code == 0
        assert "Saved analysis page:" not in result.output


def test_ingest_recursively_ingests_directory_by_default() -> None:
    """Verifies that ingest recursively ingests directory by default."""
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
    """Verifies that ingest recursive directory requires supported files."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("bulk").mkdir()
        Path("bulk/ignored.bin").write_text("ignore me", encoding="utf-8")

        result = runner.invoke(main, ["add", "bulk"])

        assert result.exit_code != 0
        assert "No supported source files found under directory" in result.output


def test_export_on_empty_wiki_succeeds() -> None:
    """Verifies that export on empty wiki succeeds."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["export"])

        assert result.exit_code == 0
        assert "Vault Export" in result.output
        assert "Exported 1 markdown file(s)" in result.output


def test_unknown_command_shows_error() -> None:
    """Verifies that unknown command shows error."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["nonexistent-command"])

        assert result.exit_code != 0


def test_provider_override_flag_switches_provider() -> None:
    """Verifies that provider override flag switches provider."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("sample.md").write_text(
            "# Traceability\n\nTraceability evidence.\n",
            encoding="utf-8",
        )
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0
        _set_provider_config()
        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            assert runner.invoke(main, ["update", "--no-graph"]).exit_code == 0

        _set_provider_config()
        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            result = runner.invoke(
                main,
                ["--provider", "openai", "legacy", "ask", "traceability"],
            )

        assert result.exit_code == 0
        assert "mode: provider:" in result.output


def test_provider_override_flag_rejects_invalid_name() -> None:
    """Verifies that provider override flag rejects invalid name."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["--provider", "invalid", "status"])

        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid" in result.output.lower()


def test_provider_override_clears_tier_and_api_key_env() -> None:
    """--provider clears stale provider-specific override fields in memory."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        config = yaml.safe_load(Path("kb.config.yaml").read_text(encoding="utf-8"))
        config["provider"] = {"name": "openai"}
        Path("kb.config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

        captured_provider: dict[str, str] = {}

        def _capture_provider_config(runtime_config, provider_catalog=None):
            """Handles capture provider config.

            Args:
                runtime_config: Runtime config value used by the operation.
                provider_catalog: Provider catalog value used by the operation.
            """
            captured_provider.clear()
            captured_provider.update(runtime_config.get("provider", {}))
            return _CliFakeProvider()

        with patch(
            "graphwiki_kb.services.build_provider", side_effect=_capture_provider_config
        ):
            result = runner.invoke(
                main,
                ["--provider", "anthropic", "update", "--no-graph"],
            )

        assert result.exit_code == 0
        assert captured_provider == {"name": "anthropic"}


def test_cli_fails_clearly_for_malformed_provider_settings() -> None:
    """Verifies that cli fails clearly for malformed provider settings."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("kb.config.yaml").write_text(
            "version: 3\nproviders:\n  openai: []\n",
            encoding="utf-8",
        )

        result = runner.invoke(main, ["status"])

        assert result.exit_code != 0
        assert "kb.config.yaml" in result.output


# --- Simplified CLI UX tests ---


def test_update_compiles_and_generates_concepts() -> None:
    """Verifies that update compiles and generates concepts."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nBody for update test.\n", encoding="utf-8"
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0
        _set_provider_config()

        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            result = runner.invoke(main, ["update", "--no-graph"])

        assert result.exit_code == 0
        assert "Update Summary" in result.output
        assert "Compiled 1 source page(s)" in result.output
        assert "Concept Summary" in result.output


def test_update_with_paths_adds_then_compiles() -> None:
    """Verifies that update with paths adds then compiles."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("note.md").write_text(
            "# New Note\n\nAdded via update.\n", encoding="utf-8"
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        _set_provider_config()

        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            result = runner.invoke(main, ["update", "--no-graph", "note.md"])

        assert result.exit_code == 0
        assert "Added note.md" in result.output
        assert "Update Summary" in result.output
        assert "Compiled 1 source page(s)" in result.output


def test_find_command_works() -> None:
    """Verifies that find command works."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["find", "missing-topic"])

        assert result.exit_code == 0
        assert "No graph artifacts or wiki pages matched that query." in result.output


def test_flat_status_shows_knowledge_base_overview() -> None:
    """Verifies that flat status shows knowledge base overview."""
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
    """Verifies that status changed flag shows diff."""
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
    """Verifies that flat export defaults to vault."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["export"])

        assert result.exit_code == 0
        assert "Vault Export" in result.output
        assert "Exported 1 markdown file(s)" in result.output


def test_config_command_shows_config() -> None:
    """Verifies that config command shows config."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["config"])

        assert result.exit_code == 0
        assert "Configuration" in result.output
        assert "project" in result.output


def test_sources_list_shows_empty() -> None:
    """Verifies that sources list shows empty."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["sources"])

        assert result.exit_code == 0
        assert "No sources ingested yet." in result.output


def test_sources_list_shows_ingested_sources() -> None:
    """Verifies that sources list shows ingested sources."""
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
    """Verifies that review successful run shows no issues."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("wiki/sources").mkdir(parents=True, exist_ok=True)
        Path("wiki/sources/alpha.md").write_text(
            _compiled_page("Alpha", "Unique content about alpha topic."),
            encoding="utf-8",
        )

        _set_provider_config()
        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            result = runner.invoke(main, ["review"])

        assert result.exit_code == 0
        assert "Review mode:" in result.output
        assert "No review issues found." in result.output


def test_review_successful_run_shows_issues() -> None:
    """Verifies that review successful run shows issues."""
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
        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            result = runner.invoke(main, ["review"])

        assert result.exit_code == 0
        assert "Review mode:" in result.output
        assert "Total review issues:" in result.output


def test_review_json_and_fail_on_warning() -> None:
    """Verifies review json output and fail-on threshold."""
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
        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            result = runner.invoke(
                main, ["review", "--json", "--fail-on", "suggestion"]
            )

        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["ok"] is False
        assert payload["issue_count"] >= 1
        assert any(issue["code"] == "overlapping-topics" for issue in payload["issues"])


def test_sources_show_displays_details() -> None:
    """Verifies that sources show displays details."""
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
    """Verifies that sources show missing slug fails."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["sources", "show", "nonexistent"])

        assert result.exit_code != 0
        assert "Source not found: nonexistent" in result.output


def test_ask_show_evidence_flag() -> None:
    """Verifies that ask show evidence flag."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nTraceability and citation evidence.\n",
            encoding="utf-8",
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0
        _set_provider_config()
        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            assert runner.invoke(main, ["update", "--no-graph"]).exit_code == 0

            result = runner.invoke(
                main,
                [
                    "legacy",
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
    """Verifies that status shows stale sources needing compile."""
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
    """Verifies that status shows current after compile."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nBody for status current test.\n", encoding="utf-8"
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0
        _set_provider_config()
        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            assert runner.invoke(main, ["update", "--no-graph"]).exit_code == 0

        result = runner.invoke(main, ["status"])

        assert result.exit_code == 0
        assert "Knowledge base is current." in result.output


def test_update_with_directory_path_adds_then_compiles() -> None:
    """Verifies that update with directory path adds then compiles."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("docs").mkdir()
        Path("docs/alpha.md").write_text("# Alpha\n\nAlpha body.\n", encoding="utf-8")
        Path("docs/beta.md").write_text("# Beta\n\nBeta body.\n", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0
        _set_provider_config()

        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            result = runner.invoke(main, ["update", "--no-graph", "docs"])

        assert result.exit_code == 0
        assert "Added 2 source(s) from" in result.output
        assert "Update Summary" in result.output
        assert "Compiled 2 source page(s)" in result.output


def test_update_with_already_present_file() -> None:
    """Verifies that update with already present file."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("note.md").write_text("# Note\n\nNote body.\n", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "note.md"]).exit_code == 0
        _set_provider_config()

        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            result = runner.invoke(main, ["update", "--no-graph", "note.md"])

        assert result.exit_code == 0
        assert "Already present: note.md" in result.output
        assert "Update Summary" in result.output


def test_update_resume_rejects_force_combination() -> None:
    """Verifies that update resume rejects force combination."""
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
    """Verifies that update fails without provider config."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            result = runner.invoke(main, ["update", "--no-graph"])

        assert result.exit_code != 0
        assert "Provider is not configured" in result.output


def test_update_generic_service_error_propagates_unexpected_exception() -> None:
    """Verifies update does not hide unexpected service errors."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("note.md").write_text("# Note\n\nBody.\n", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "note.md"]).exit_code == 0
        _set_provider_config()

        with (
            patch(
                "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
            ),
            patch(
                "graphwiki_kb.services.update_service.UpdateService.run",
                side_effect=RuntimeError("boom"),
            ),
        ):
            result = runner.invoke(main, ["update", "--no-graph"])

        assert result.exit_code != 0
        assert isinstance(result.exception, RuntimeError)
        assert str(result.exception) == "boom"


# ---------------------------------------------------------------------------
# Config subcommands
# ---------------------------------------------------------------------------


def test_config_show_subcommand() -> None:
    """Verifies that config show subcommand."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["config", "show"])

        assert result.exit_code == 0
        assert "Configuration" in result.output


def test_config_provider_set_and_clear() -> None:
    """Verifies that config provider set and clear."""
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
        config = yaml.safe_load(Path("kb.config.yaml").read_text(encoding="utf-8"))
        assert config["provider"]["name"] == "anthropic"
        assert config["providers"]["anthropic"]["model"] == "claude-4"

        result = runner.invoke(main, ["config", "provider", "clear"])
        assert result.exit_code == 0
        assert "Provider cleared." in result.output

        config = yaml.safe_load(Path("kb.config.yaml").read_text(encoding="utf-8"))
        assert config["provider"] == {}


def test_config_provider_set_switching_clears_stale() -> None:
    """Verifies that config provider set switching clears stale."""
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
        assert config["providers"]["openai"]["model"] == "gpt-5.4"


def test_config_provider_set_rejects_unknown_name() -> None:
    """Verifies that config provider set rejects unknown name."""
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


def test_doctor_json_output() -> None:
    """Verifies that doctor json output."""
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
    """Verifies that find json output."""
    runner = CliRunner(mix_stderr=False)
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nTraceability and citation.\n", encoding="utf-8"
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0
        _set_provider_config()
        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            assert runner.invoke(main, ["update", "--no-graph"]).exit_code == 0

        result = runner.invoke(main, ["find", "--json", "traceability"])

        assert result.exit_code == 0
        assert result.stderr == ""
        data = json.loads(result.output)
        assert data["retriever"] == "graph-and-wiki-index"
        assert isinstance(data["results"], list)
        assert len(data["results"]) > 0
        assert data["results"][0]["retriever"] == "wiki-index"
        assert "title" in data["results"][0]
        assert "path" in data["results"][0]
        assert "score" in data["results"][0]


def test_find_json_empty_results() -> None:
    """Verifies that find json empty results."""
    runner = CliRunner(mix_stderr=False)
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["find", "--json", "missing-topic"])

        assert result.exit_code == 0
        assert result.stderr == ""
        data = json.loads(result.output)
        assert data["retriever"] == "graph-and-wiki-index"
        assert data["results"] == []


def test_status_json_output() -> None:
    """Verifies that status json output."""
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
    """Verifies that status changed json output."""
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
        assert "missing" in data


def test_sources_list_json_output() -> None:
    """Verifies that sources list json output."""
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
    """Verifies that sources list json empty."""
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


def test_clean_citation_refs_strips_inline_citation_refs() -> None:
    """Verifies that clean citation refs strips inline citation refs."""
    from graphwiki_kb.services.citation_cleanup import clean_citation_refs

    raw = (
        "Traceability is preserved [wiki/sources/alpha.md#chunk-0] "
        "through compiled pages [wiki/sources/beta.md#chunk-2]."
    )
    cleaned = clean_citation_refs(raw)
    assert "wiki/sources/" not in cleaned
    assert "chunk-" not in cleaned
    assert "Traceability is preserved through compiled pages." in cleaned

    # Multi-ref brackets
    multi = (
        "Evidence shows [wiki/sources/a.md#chunk-0, wiki/sources/b.md#chunk-1] support."
    )
    assert "wiki/sources/" not in clean_citation_refs(multi)
    assert "Evidence shows support." in clean_citation_refs(multi)

    # Backticked refs
    backtick = "Facts [`wiki/sources/a.md#chunk-0`] confirmed."
    assert "wiki/sources/" not in clean_citation_refs(backtick)

    # Parenthesized refs
    paren = "Claims (wiki/sources/a.md#chunk-0) stand."
    assert "wiki/sources/" not in clean_citation_refs(paren)

    unbalanced = "Keep malformed [wiki/sources/a.md#chunk-0 and this bracket."
    assert clean_citation_refs(unbalanced) == unbalanced
