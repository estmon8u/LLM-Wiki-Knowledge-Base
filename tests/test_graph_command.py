from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from src.cli import main


def _write_graphrag_settings() -> None:
    Path("graph/graphrag").mkdir(parents=True, exist_ok=True)
    Path("graph/graphrag/settings.yaml").write_text(
        "input:\n"
        "  type: text\n"
        "input_storage:\n"
        "  type: file\n"
        "  base_dir: input\n"
        "chunking:\n"
        "  type: tokens\n"
        "  size: 1200\n"
        "  overlap: 100\n"
        "  encoding_model: o200k_base\n",
        encoding="utf-8",
    )


def test_graph_sync_command_writes_sources_json() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample Research Note\n\nGraph sync preserves provenance.\n",
            encoding="utf-8",
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0
        _write_graphrag_settings()

        result = runner.invoke(main, ["graph", "sync"])

        assert result.exit_code == 0
        assert "Synced 1 normalized source(s)" in result.output
        input_file = Path("graph/graphrag/input/sources.json")
        records = json.loads(input_file.read_text(encoding="utf-8"))
        assert records[0]["title"] == "Sample Research Note"
        assert records[0]["normalized_path"] == "raw/normalized/sample-research-note.md"
        assert records[0]["text"].startswith("# Sample Research Note")


def test_graph_sync_command_supports_json_output() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        _write_graphrag_settings()

        result = runner.invoke(main, ["graph", "sync", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["source_count"] == 0
        assert payload["output_path"] == "graph/graphrag/input/sources.json"
        assert payload["settings_path"] == "graph/graphrag/settings.yaml"
        assert "source_id" in payload["metadata_fields"]


def test_graph_sync_command_requires_initialized_project() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["graph", "sync"])

        assert result.exit_code != 0
        assert "Project not initialized" in result.output


def test_graph_sync_command_reports_missing_workspace_settings() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["graph", "sync"])

        assert result.exit_code != 0
        assert "GraphRAG settings not found" in result.output
