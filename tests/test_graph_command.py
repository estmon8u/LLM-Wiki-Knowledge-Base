from __future__ import annotations

import json
from pathlib import Path
import subprocess

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


def test_graph_init_command_runs_graphrag_init(monkeypatch) -> None:
    calls = []

    def fake_run(command, *, cwd, capture_output, text):
        calls.append(command)
        settings_path = Path(cwd) / "graph" / "graphrag" / "settings.yaml"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text("input:\n  type: json\n", encoding="utf-8")
        return subprocess.CompletedProcess(
            command, 0, stdout="initialized\n", stderr=""
        )

    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.run",
        fake_run,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["graph", "init", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["workspace_dir"] == "graph/graphrag"
        assert payload["settings_path"] == "graph/graphrag/settings.yaml"
        assert payload["returncode"] == 0
        assert calls[0][1:4] == ("-m", "graphrag", "init")
        assert "--force" in calls[0]


def test_graph_init_command_supports_human_output(monkeypatch) -> None:
    def fake_run(command, *, cwd, capture_output, text):
        settings_path = Path(cwd) / "graph" / "graphrag" / "settings.yaml"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text("input:\n  type: json\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.run",
        fake_run,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["graph", "init"])

        assert result.exit_code == 0
        assert "Initialized GraphRAG workspace at graph/graphrag" in result.output
        assert "Settings: graph/graphrag/settings.yaml" in result.output


def test_graph_init_command_reports_graphrag_failure(monkeypatch) -> None:
    def fake_run(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(
            command,
            2,
            stdout="",
            stderr="init failed\n",
        )

    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.run",
        fake_run,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["graph", "init"])

        assert result.exit_code != 0
        assert "GraphRAG command failed: init failed" in result.output


def test_graph_index_command_records_run(monkeypatch) -> None:
    calls = []

    def fake_run(command, *, cwd, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="dry run ok\n", stderr="")

    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.run",
        fake_run,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text("# Sample\n\nGraph indexing.\n", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0
        _write_graphrag_settings()
        assert runner.invoke(main, ["graph", "sync"]).exit_code == 0

        result = runner.invoke(
            main,
            ["graph", "index", "--method", "fast", "--dry-run", "--json"],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["run"]["method"] == "fast"
        assert payload["run"]["dry_run"] is True
        assert payload["run"]["success"] is True
        assert calls[0][1:4] == ("-m", "graphrag", "index")
        assert "--dry-run" in calls[0]
        assert Path("graph/runs/graph_index_runs.json").exists()


def test_graph_index_command_supports_human_output(monkeypatch) -> None:
    def fake_run(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(command, 0, stdout="dry run ok\n", stderr="")

    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.run",
        fake_run,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text("# Sample\n\nGraph indexing.\n", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0
        _write_graphrag_settings()
        assert runner.invoke(main, ["graph", "sync"]).exit_code == 0

        result = runner.invoke(
            main, ["graph", "index", "--method", "fast", "--dry-run"]
        )

        assert result.exit_code == 0
        assert "GraphRAG dry run completed with method fast." in result.output
        assert "Run ID:" in result.output


def test_graph_index_command_records_failed_run(monkeypatch) -> None:
    def fake_run(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(
            command,
            3,
            stdout="",
            stderr="index failed\n",
        )

    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.run",
        fake_run,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text("# Sample\n\nGraph indexing.\n", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0
        _write_graphrag_settings()
        assert runner.invoke(main, ["graph", "sync"]).exit_code == 0

        result = runner.invoke(main, ["graph", "index", "--method", "fast"])

        assert result.exit_code != 0
        assert "GraphRAG command failed: index failed" in result.output
        runs = json.loads(Path("graph/runs/graph_index_runs.json").read_text())
        assert runs[0]["success"] is False
        assert runs[0]["returncode"] == 3


def test_graph_index_command_requires_synced_input() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        _write_graphrag_settings()

        result = runner.invoke(main, ["graph", "index", "--method", "fast"])

        assert result.exit_code != 0
        assert "Run `kb graph sync` first" in result.output


def test_graph_index_command_requires_workspace_settings() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["graph", "index", "--method", "fast"])

        assert result.exit_code != 0
        assert "Run `kb graph init` first" in result.output


def test_graph_index_command_requires_non_empty_input() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        _write_graphrag_settings()
        Path("graph/graphrag/input").mkdir(parents=True, exist_ok=True)
        Path("graph/graphrag/input/sources.json").write_text("[]", encoding="utf-8")

        result = runner.invoke(main, ["graph", "index", "--method", "fast"])

        assert result.exit_code != 0
        assert "GraphRAG input has no documents" in result.output


def test_graph_status_command_reports_index_state() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        _write_graphrag_settings()
        Path("graph/graphrag/input").mkdir(parents=True, exist_ok=True)
        Path("graph/graphrag/input/sources.json").write_text(
            json.dumps([{"id": "src-1"}]),
            encoding="utf-8",
        )
        for table in ("entities", "relationships", "communities", "community_reports"):
            Path(f"graph/graphrag/output/{table}.parquet").parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            Path(f"graph/graphrag/output/{table}.parquet").write_text(
                "",
                encoding="utf-8",
            )

        result = runner.invoke(main, ["graph", "status", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["workspace_initialized"] is True
        assert payload["input_document_count"] == 1
        assert payload["entities_present"] is True
        assert payload["relationships_present"] is True
        assert payload["communities_present"] is True
        assert payload["community_reports_present"] is True


def test_graph_status_command_supports_human_output(monkeypatch) -> None:
    def fake_run(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.run",
        fake_run,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text("# Sample\n\nGraph status.\n", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "sample.md"]).exit_code == 0
        _write_graphrag_settings()
        assert runner.invoke(main, ["graph", "sync"]).exit_code == 0
        assert (
            runner.invoke(
                main, ["graph", "index", "--method", "fast", "--dry-run"]
            ).exit_code
            == 0
        )

        result = runner.invoke(main, ["graph", "status"])

        assert result.exit_code == 0
        assert "Workspace initialized: yes" in result.output
        assert "Input: present (1 document(s))" in result.output
        assert "Last index run:" in result.output
        assert "success: yes" in result.output
