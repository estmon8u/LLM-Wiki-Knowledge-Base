from __future__ import annotations

import json
from pathlib import Path
import subprocess

from click.testing import CliRunner
import pandas as pd
import yaml

from src.cli import main
from src.services.graphrag_defaults import (
    DEFAULT_GRAPHRAG_EMBEDDING_MODEL,
    DEFAULT_GRAPHRAG_MODEL,
)


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


def _write_ready_graphrag_index() -> None:
    _write_graphrag_settings()
    Path("graph/graphrag/input").mkdir(parents=True, exist_ok=True)
    Path("graph/graphrag/input/sources.json").write_text(
        json.dumps([{"id": "src-1", "text": "Graph source text"}]),
        encoding="utf-8",
    )
    Path("graph/graphrag/output").mkdir(parents=True, exist_ok=True)
    Path("graph/graphrag/output/entities.parquet").write_text("", encoding="utf-8")
    Path("graph/runs").mkdir(parents=True, exist_ok=True)
    Path("graph/runs/graph_index_runs.json").write_text(
        json.dumps(
            [
                {
                    "run_id": "20260511T000000Z",
                    "created_at": "2026-05-11T00:00:00+00:00",
                    "method": "fast",
                    "dry_run": False,
                    "success": True,
                    "returncode": 0,
                    "command": ["python", "-m", "graphrag", "index"],
                    "stdout_tail": "indexed",
                    "stderr_tail": "",
                }
            ]
        ),
        encoding="utf-8",
    )


def _write_valid_graph_export_table() -> None:
    _write_graphrag_settings()
    Path("graph/graphrag/input").mkdir(parents=True, exist_ok=True)
    Path("graph/graphrag/input/sources.json").write_text(
        json.dumps([{"id": "src-1", "text": "Graph source text"}]),
        encoding="utf-8",
    )
    output_dir = Path("graph/graphrag/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "id": "entity-1",
                "title": "Retrieval-Augmented Generation",
                "description": "A graph entity.",
                "degree": 2,
            }
        ]
    ).to_parquet(output_dir / "entities.parquet")
    Path("graph/runs").mkdir(parents=True, exist_ok=True)
    Path("graph/runs/graph_index_runs.json").write_text(
        json.dumps(
            [
                {
                    "run_id": "20260511T000000Z",
                    "created_at": "2026-05-11T00:00:00+00:00",
                    "method": "fast",
                    "dry_run": False,
                    "success": True,
                    "returncode": 0,
                    "command": ["python", "-m", "graphrag", "index"],
                    "stdout_tail": "indexed",
                    "stderr_tail": "",
                }
            ]
        ),
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
        assert payload["provider"] == "openai"
        assert payload["model"] == DEFAULT_GRAPHRAG_MODEL
        assert payload["embedding_provider"] == "openai"
        assert payload["embedding_model"] == DEFAULT_GRAPHRAG_EMBEDDING_MODEL
        assert payload["api_key_env"] == "OPENAI_API_KEY"
        assert payload["embedding_api_key_env"] == "OPENAI_API_KEY"
        assert calls[0][1:4] == ("-m", "graphrag", "init")
        assert calls[0][calls[0].index("--model") + 1] == DEFAULT_GRAPHRAG_MODEL
        assert (
            calls[0][calls[0].index("--embedding") + 1]
            == DEFAULT_GRAPHRAG_EMBEDDING_MODEL
        )
        assert "--force" in calls[0]
        settings = yaml.safe_load(Path("graph/graphrag/settings.yaml").read_text())
        assert (
            settings["completion_models"]["default_completion_model"]["model"]
            == DEFAULT_GRAPHRAG_MODEL
        )
        assert (
            settings["embedding_models"]["default_embedding_model"]["model"]
            == DEFAULT_GRAPHRAG_EMBEDDING_MODEL
        )
        assert (
            settings["completion_models"]["default_completion_model"]["api_key"]
            == "${OPENAI_API_KEY}"
        )
        assert (
            settings["embedding_models"]["default_embedding_model"]["api_key"]
            == "${OPENAI_API_KEY}"
        )


def test_graph_init_command_uses_graph_config_and_cli_overrides(monkeypatch) -> None:
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
        config = yaml.safe_load(Path("kb.config.yaml").read_text(encoding="utf-8"))
        config["graph"] = {
            "provider": "openai",
            "model": "configured-chat",
            "embedding_model": "configured-embedding",
            "api_key_env": "OPENAI_GRAPH_KEY",
        }
        Path("kb.config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )

        result = runner.invoke(
            main,
            [
                "graph",
                "init",
                "--model",
                "override-chat",
                "--embedding",
                "override-embedding",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["model"] == "override-chat"
        assert payload["embedding_model"] == "override-embedding"
        assert payload["api_key_env"] == "OPENAI_GRAPH_KEY"
        assert payload["embedding_api_key_env"] == "OPENAI_GRAPH_KEY"
        assert calls[0][calls[0].index("--model") + 1] == "override-chat"
        assert calls[0][calls[0].index("--embedding") + 1] == "override-embedding"
        settings = yaml.safe_load(Path("graph/graphrag/settings.yaml").read_text())
        completion = settings["completion_models"]["default_completion_model"]
        embedding = settings["embedding_models"]["default_embedding_model"]
        assert completion["model"] == "override-chat"
        assert completion["api_key"] == "${OPENAI_GRAPH_KEY}"
        assert embedding["model"] == "override-embedding"
        assert embedding["api_key"] == "${OPENAI_GRAPH_KEY}"


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


def test_graph_ask_command_supports_json_output(monkeypatch) -> None:
    calls = []

    def fake_run(command, *, cwd, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="GraphRAG says REALM augments pretraining.\n",
            stderr="",
        )

    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.run",
        fake_run,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        _write_ready_graphrag_index()

        result = runner.invoke(
            main,
            [
                "graph",
                "ask",
                "How does REALM differ from RAG?",
                "--method",
                "local",
                "--community-level",
                "1",
                "--no-dynamic-selection",
                "--response-type",
                "Single Sentence",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["retriever"] == "graphrag"
        assert payload["method"] == "local"
        assert payload["answer"] == "GraphRAG says REALM augments pretraining."
        assert payload["index_run_id"] == "20260511T000000Z"
        assert payload["input_manifest_hash"]
        assert calls[0][1:4] == ("-m", "graphrag", "query")
        assert calls[0][-1] == "How does REALM differ from RAG?"


def test_graph_ask_command_supports_human_output(monkeypatch) -> None:
    def fake_run(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="GraphRAG answer body.\n",
            stderr="",
        )

    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.run",
        fake_run,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        _write_ready_graphrag_index()

        result = runner.invoke(
            main,
            ["graph", "ask", "What is RAG?", "--method", "basic"],
        )

        assert result.exit_code == 0
        assert "retriever: graphrag, method: basic" in result.output
        assert "GraphRAG answer body." in result.output


def test_graph_ask_command_saves_analysis_page(monkeypatch) -> None:
    def fake_run(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="GraphRAG saved answer.\n",
            stderr="",
        )

    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.run",
        fake_run,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        _write_ready_graphrag_index()

        result = runner.invoke(
            main,
            [
                "graph",
                "ask",
                "What is dense passage retrieval used for?",
                "--method",
                "drift",
                "--save",
            ],
        )

        assert result.exit_code == 0
        saved_path = Path(
            "wiki/analysis/graphrag-what-is-dense-passage-retrieval-used-for.md"
        )
        assert saved_path.exists()
        text = saved_path.read_text(encoding="utf-8")
        assert "retriever: graphrag" in text
        assert "method: drift" in text
        assert "## Raw GraphRAG Output" in text
        assert (
            "Saved analysis page: "
            "wiki/analysis/graphrag-what-is-dense-passage-retrieval-used-for.md"
            in result.output
        )


def test_graph_ask_command_reports_missing_index_output() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        _write_graphrag_settings()
        Path("graph/graphrag/input").mkdir(parents=True, exist_ok=True)
        Path("graph/graphrag/input/sources.json").write_text(
            json.dumps([{"id": "src-1"}]),
            encoding="utf-8",
        )

        result = runner.invoke(
            main,
            ["graph", "ask", "What is RAG?", "--method", "basic"],
        )

        assert result.exit_code != 0
        assert "GraphRAG index output not found" in result.output


def test_top_level_ask_routes_auto_and_saves_graph_metadata(monkeypatch) -> None:
    calls = []

    def fake_run(command, *, cwd, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="REALM augments pretraining while RAG augments generation.\n",
            stderr="",
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.run",
        fake_run,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        _write_ready_graphrag_index()

        result = runner.invoke(
            main,
            [
                "ask",
                "--method",
                "auto",
                "--save",
                "--json",
                "How does REALM differ from RAG?",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["retriever"] == "graph"
        assert payload["method"] == "drift"
        assert payload["planner"] == "heuristic"
        assert payload["route_reason"] == "comparison keyword"
        assert payload["claim_support"] == "graph-grounded"
        assert calls[0][calls[0].index("--method") + 1] == "drift"
        saved_path = Path(payload["saved_path"])
        assert saved_path.exists()
        saved_text = saved_path.read_text(encoding="utf-8")
        assert "retriever: graph" in saved_text
        assert "planner: heuristic" in saved_text
        assert "claim_support: graph-grounded" in saved_text


def test_top_level_ask_explicit_method_bypasses_auto_router(monkeypatch) -> None:
    calls = []

    def fake_run(command, *, cwd, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout="Local answer.\n", stderr=""
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.run",
        fake_run,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        _write_ready_graphrag_index()

        result = runner.invoke(
            main,
            ["ask", "--method", "local", "What patterns appear across the corpus?"],
        )

        assert result.exit_code == 0
        assert "retriever: graph, method: local" in result.output
        assert calls[0][calls[0].index("--method") + 1] == "local"


def test_top_level_ask_show_evidence_and_saved_path(monkeypatch) -> None:
    def fake_run(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(
            command, 0, stdout="Graph answer with trace.\n", stderr=""
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.run",
        fake_run,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        _write_ready_graphrag_index()

        result = runner.invoke(
            main,
            [
                "ask",
                "--method",
                "basic",
                "--save-as",
                "graph-answer",
                "--show-evidence",
                "What is RAG?",
            ],
        )

        assert result.exit_code == 0
        assert "Source Trace" in result.output
        assert "GraphRAG input:" in result.output
        assert "Route reason: explicit method override" in result.output
        assert "Claim support: graph-grounded" in result.output
        assert "Saved analysis page:" in result.output
        assert list(Path("wiki/analysis").glob("*graph-answer*.md"))


def test_top_level_ask_requires_graph_credentials(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        _write_ready_graphrag_index()

        result = runner.invoke(main, ["ask", "What is RAG?"])

        assert result.exit_code != 0
        assert "OPENAI_API_KEY" in result.output


def test_top_level_ask_does_not_call_legacy_search(monkeypatch) -> None:
    def fail_legacy_search(*args, **kwargs):
        raise AssertionError("legacy search should not be used by kb ask")

    def fake_run(command, *, cwd, capture_output, text):
        return subprocess.CompletedProcess(
            command, 0, stdout="Graph answer.\n", stderr=""
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.services.search_service.SearchService.search",
        fail_legacy_search,
    )
    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.run",
        fake_run,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        _write_ready_graphrag_index()

        result = runner.invoke(main, ["ask", "--method", "basic", "What is RAG?"])

        assert result.exit_code == 0
        assert "Graph answer." in result.output


def test_graph_export_wiki_command_supports_json_output() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        _write_valid_graph_export_table()

        result = runner.invoke(main, ["graph", "export-wiki", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["exported_count"] == 2
        assert "wiki/graph/index.md" in payload["exported_paths"]
        assert payload["table_counts"]["entities"] == 1
        assert "relationships" in payload["missing_tables"]
        assert Path("wiki/graph/entities/retrieval-augmented-generation.md").exists()


def test_graph_export_wiki_command_supports_human_output() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        _write_valid_graph_export_table()

        result = runner.invoke(main, ["graph", "export-wiki"])

        assert result.exit_code == 0
        assert "Exported 2 GraphRAG wiki page(s) to wiki/graph" in result.output
        assert "Missing GraphRAG table(s):" in result.output
