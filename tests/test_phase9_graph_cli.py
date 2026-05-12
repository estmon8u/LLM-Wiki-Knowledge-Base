from __future__ import annotations

import json
from pathlib import Path
import subprocess
from unittest.mock import patch

from click.testing import CliRunner
import pandas as pd
import yaml

from src.cli import main
from src.services.compile_service import CompileResult
from src.services.concept_service import ConceptGenerationResult
from src.services.graphrag_command_service import GraphRAGCommandResult
from src.services.graphrag_input_sync_service import GraphRAGInputSyncResult
from src.services.graphrag_query_service import GraphRAGQueryAnswer
from src.services.graphrag_status_service import GraphRAGIndexRun
from src.services.graphrag_status_service import GraphRAGStatusService
from src.services.graphrag_sync_service import GraphRAGSyncDecision, GraphRAGSyncResult
from src.services.graphrag_wiki_export_service import GraphRAGWikiExportResult
from src.services.lint_service import _file_sha256, _input_manifest_hash
from src.services.project_service import build_project_paths
from src.services.update_service import GraphUpdateResult, IngestSummary, UpdateResult
from tests.test_cli import _CliFakeProvider, _set_provider_config


def _write_graph_tables(root: Path) -> None:
    output_dir = root / "graph" / "graphrag" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"id": "doc-1", "title": "RAG Paper"}]).to_parquet(
        output_dir / "documents.parquet"
    )
    pd.DataFrame([{"id": "tu-1", "text": "RAG text unit."}]).to_parquet(
        output_dir / "text_units.parquet"
    )
    pd.DataFrame(
        [{"id": "entity-1", "title": "RAG", "description": "Retrieval method."}]
    ).to_parquet(output_dir / "entities.parquet")
    pd.DataFrame(
        [{"id": "rel-1", "source": "RAG", "target": "REALM", "description": "rel"}]
    ).to_parquet(output_dir / "relationships.parquet")
    pd.DataFrame([{"id": "community-0", "community": 0}]).to_parquet(
        output_dir / "communities.parquet"
    )
    pd.DataFrame(
        [{"id": "report-0", "community": 0, "summary": "Retrieval systems."}]
    ).to_parquet(output_dir / "community_reports.parquet")


def _graph_answer(*, saved_path: str | None = None) -> GraphRAGQueryAnswer:
    return GraphRAGQueryAnswer(
        question="What changed?",
        answer="GraphRAG answered from the graph.",
        raw_output="GraphRAG answered from the graph.",
        method="drift",
        created_at="2026-05-12T00:00:00+00:00",
        index_run_id="run-1",
        input_manifest_hash="hash-1",
        command=("python", "-m", "graphrag", "query"),
        stdout="GraphRAG answered from the graph.",
        stderr="",
        saved_path=saved_path,
        retriever="graph",
        planner="heuristic",
        route_reason="comparison question",
        claim_support="graph-grounded",
        source_trace={
            "input_path": "graph/graphrag/input/sources.json",
            "output_dir": "graph/graphrag/output",
        },
    )


def _input_sync_result(root: Path) -> GraphRAGInputSyncResult:
    return GraphRAGInputSyncResult(
        source_count=1,
        output_path=root / "graph/graphrag/input/sources.json",
        settings_path=root / "graph/graphrag/settings.yaml",
        metadata_fields=("source_id",),
        settings_updated=False,
    )


def _decision(*, action: str = "index", method: str | None = "fast"):
    return GraphRAGSyncDecision(
        action=action,
        method=method,
        reason="test decision",
        output_state="missing",
        input_digest="input-digest",
        config_digest="config-digest",
        input_changed=True,
        config_changed=False,
        changed_source_count=1,
        cost_warning="GraphRAG cost warning.",
    )


def _sync_result(root: Path, *, action: str = "index", method: str | None = "fast"):
    return GraphRAGSyncResult(
        input_sync=_input_sync_result(root),
        decision=_decision(action=action, method=method),
    )


def _index_run() -> GraphRAGIndexRun:
    return GraphRAGIndexRun(
        run_id="run-1",
        created_at="2026-05-12T00:00:00+00:00",
        method="fast",
        dry_run=False,
        success=True,
        returncode=0,
        command=("python", "-m", "graphrag", "index"),
        stdout_tail="indexed",
        stderr_tail="",
        input_digest="input-digest",
    )


def test_graph_command_group_is_removed() -> None:
    result = CliRunner().invoke(main, ["graph"])

    assert result.exit_code != 0
    assert "No such command 'graph'" in result.output


def test_ask_prints_graph_answer_metadata_and_source_trace() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        with patch(
            "src.services.graph_ask_controller_service.GraphAskControllerService.ask",
            return_value=_graph_answer(saved_path="wiki/analysis/question.md"),
        ):
            result = runner.invoke(
                main,
                ["ask", "--show-evidence", "--save", "What", "changed?"],
            )

        assert result.exit_code == 0
        assert "retriever: graph" in result.output
        assert "Source Trace" in result.output
        assert "GraphRAG input: graph/graphrag/input/sources.json" in result.output
        assert "Route reason: comparison question" in result.output
        assert "GraphRAG answered from the graph." in result.output
        assert "Saved analysis page: wiki/analysis/question.md" in result.output


def test_ask_json_outputs_graph_answer_payload() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        with patch(
            "src.services.graph_ask_controller_service.GraphAskControllerService.ask",
            return_value=_graph_answer(),
        ):
            result = runner.invoke(main, ["ask", "--json", "What", "changed?"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["retriever"] == "graph"
        assert payload["method"] == "drift"
        assert payload["command"] == ["python", "-m", "graphrag", "query"]


def test_update_output_renders_resume_removed_concepts_and_graph_details() -> None:
    update_result = UpdateResult(
        ingest_summaries=[
            IngestSummary(Path("folder"), is_dir=True, created_count=2),
            IngestSummary(Path("note.md"), is_dir=False, created_count=0),
        ],
        compile_result=CompileResult(
            compiled_count=1,
            skipped_count=2,
            compiled_paths=["wiki/sources/note.md"],
            resumed_from_run_id="compile-run-1",
        ),
        concept_result=ConceptGenerationResult(
            concept_paths=["wiki/concepts/retrieval.md"],
            updated_source_paths=["wiki/sources/note.md"],
            removed_paths=["wiki/concepts/old.md"],
        ),
        search_refreshed=True,
        graph_result=GraphUpdateResult(
            initialized=True,
            preflight_result=_sync_result(Path.cwd()),
            sync_result=GraphRAGSyncResult(
                input_sync=_input_sync_result(Path.cwd()),
                decision=_decision(),
                index_run=_index_run(),
            ),
            export_result=GraphRAGWikiExportResult(
                exported_paths=["wiki/graph/index.md"],
                table_counts={"entities": 1},
            ),
        ),
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        with patch(
            "src.services.update_service.UpdateService.run", return_value=update_result
        ):
            result = runner.invoke(main, ["update"])

        assert result.exit_code == 0
        assert "resumed interrupted update run compile-run-1" in result.output
        assert "Removed 1 stale concept page(s)" in result.output
        assert "initialized graph workspace" in result.output
        assert "GraphRAG cost warning." in result.output
        assert "Graph index run: run-1 (fast)" in result.output
        assert "Graph wiki export: 1 page(s)" in result.output


def test_update_output_renders_non_index_graph_decision() -> None:
    update_result = UpdateResult(
        compile_result=CompileResult(
            compiled_count=0,
            skipped_count=0,
            compiled_paths=[],
        ),
        concept_result=ConceptGenerationResult(
            concept_paths=[],
            updated_source_paths=[],
            removed_paths=[],
        ),
        graph_result=GraphUpdateResult(
            skipped=True,
            skip_reason="test decision",
            preflight_result=_sync_result(Path.cwd(), action="skip", method=None),
        ),
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        with patch(
            "src.services.update_service.UpdateService.run",
            return_value=update_result,
        ):
            result = runner.invoke(main, ["update"])

        assert result.exit_code == 0
        assert "Graph index action: test decision" in result.output


def test_init_sets_up_graph_workspace_settings() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(main, ["init"])

        assert result.exit_code == 0
        settings_path = Path("graph/graphrag/settings.yaml")
        assert settings_path.exists()
        settings = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
        completion = settings["completion_models"]["default_completion_model"]
        embedding = settings["embedding_models"]["default_embedding_model"]
        assert completion["model_provider"] == "openai"
        assert completion["api_key"] == "${OPENAI_API_KEY}"
        assert embedding["api_key"] == "${OPENAI_API_KEY}"
        assert (
            settings["local_search"]["prompt"]
            == "prompts/local_search_system_prompt.txt"
        )
        assert Path("graph/graphrag/prompts/extract_graph.txt").exists()


def test_update_runs_graph_sync_index_and_export(monkeypatch) -> None:
    calls = []

    def fake_run(command, *, cwd, capture_output, text):
        calls.append(command)
        _write_graph_tables(Path(cwd))
        return subprocess.CompletedProcess(command, 0, stdout="indexed\n", stderr="")

    class FakePopen:
        """Simulate subprocess.Popen for the streaming path."""

        def __init__(
            self,
            command,
            *,
            cwd,
            stdout,
            stderr,
            text,
            encoding=None,
            errors=None,
            bufsize=None,
            env=None,
        ):
            calls.append(command)
            _write_graph_tables(Path(cwd))
            self.returncode = 0
            import io

            self.stdout = io.StringIO("indexed\n")
            self.stderr = io.StringIO("")

        def wait(self):
            pass

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.run",
        fake_run,
    )
    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.Popen",
        FakePopen,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nGraph update body.\n", encoding="utf-8"
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        _set_provider_config()

        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            result = runner.invoke(main, ["update", "sample.md"])

        assert result.exit_code == 0
        assert "GraphRAG Summary" in result.output
        assert "Graph index run:" in result.output
        assert "Graph wiki export:" in result.output
        assert Path("graph/graphrag/input/sources.json").exists()
        assert Path("wiki/graph/index.md").exists()
        assert calls and calls[0][1:4] == ("-m", "graphrag", "index")


def test_update_no_graph_flag_skips_graph(monkeypatch) -> None:
    def fail_run(*args, **kwargs):
        raise AssertionError("GraphRAG should not run with --no-graph")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.services.graphrag_command_service.subprocess.run",
        fail_run,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text("# Sample\n\nNo graph body.\n", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0
        _set_provider_config()

        with patch("src.services.build_provider", return_value=_CliFakeProvider()):
            result = runner.invoke(main, ["update", "--no-graph", "sample.md"])

        assert result.exit_code == 0
        assert "Graph skipped: --no-graph requested." in result.output


def test_status_json_includes_graph_status() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["status", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["graph_status"]["workspace_initialized"] is True
        assert payload["graph_status"]["workspace_dir"] == "graph/graphrag"


def test_status_human_output_includes_last_graph_index() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        GraphRAGStatusService(build_project_paths(Path.cwd())).record_index_run(
            method="fast",
            dry_run=False,
            result=GraphRAGCommandResult(
                command=("python", "-m", "graphrag", "index"),
                cwd=Path.cwd(),
                returncode=0,
                stdout="indexed",
                stderr="",
            ),
        )

        result = runner.invoke(main, ["status"])

        assert result.exit_code == 0
        assert "Last index:" in result.output
        assert "Index method: fast" in result.output


def test_export_includes_graph_wiki_export() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        _write_graph_tables(Path.cwd())

        result = runner.invoke(main, ["export"])

        assert result.exit_code == 0
        assert "Vault Export" in result.output
        assert "Graph Wiki Export" in result.output
        assert Path("wiki/graph/index.md").exists()


class _FakeWorkspace:
    def __init__(self, *, initialized: bool) -> None:
        self.initialized = initialized
        self.ensured = False

    def is_initialized(self) -> bool:
        return self.initialized

    def ensure_workspace(self) -> None:
        self.ensured = True
        self.initialized = True


class _FakeGraphSync:
    def __init__(self, root: Path, *results, error: Exception | None = None) -> None:
        self.workspace_dir = root / "graph/graphrag"
        self.results = list(results)
        self.error = error
        self.calls = []

    def sync(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.results.pop(0)


class _FakeGraphExport:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error

    def export_wiki(self):
        if self.error is not None:
            raise self.error
        return GraphRAGWikiExportResult(
            exported_paths=["wiki/graph/index.md"],
            table_counts={},
        )


def _update_service_for_graph(test_project, *, workspace, sync, export=None):
    from src.services.update_service import UpdateService

    return UpdateService(
        ingest_service=None,
        compile_service=None,
        concept_service=None,
        search_service=None,
        config=test_project.command_context.config,
        graphrag_workspace_service=workspace,
        graphrag_sync_service=sync,
        graphrag_wiki_export_service=export or _FakeGraphExport(),
    )


def test_update_service_skips_graph_when_services_or_config_are_missing(
    test_project,
) -> None:
    from src.services.update_service import UpdateOptions, UpdateService

    missing_services = UpdateService(
        ingest_service=None,
        compile_service=None,
        concept_service=None,
        search_service=None,
        config=test_project.command_context.config,
    )._run_graph_sync(UpdateOptions())
    missing_config = UpdateService(
        ingest_service=None,
        compile_service=None,
        concept_service=None,
        search_service=None,
        config={},
        graphrag_workspace_service=_FakeWorkspace(initialized=True),
        graphrag_sync_service=_FakeGraphSync(test_project.root),
        graphrag_wiki_export_service=_FakeGraphExport(),
    )._run_graph_sync(UpdateOptions())

    assert missing_services.skip_reason == "Graph services unavailable."
    assert missing_config.skip_reason == "Graph config not configured."


def test_update_service_initializes_and_skips_when_preflight_skips(
    test_project,
) -> None:
    from src.services.update_service import UpdateOptions

    workspace = _FakeWorkspace(initialized=False)
    service = _update_service_for_graph(
        test_project,
        workspace=workspace,
        sync=_FakeGraphSync(
            test_project.root,
            _sync_result(test_project.root, action="skip", method=None),
        ),
    )

    result = service._run_graph_sync(UpdateOptions())

    assert workspace.ensured is True
    assert result.initialized is True
    assert result.skipped is True
    assert result.skip_reason == "test decision"


def test_update_service_reports_preflight_failure(test_project) -> None:
    from src.services.update_service import UpdateOptions

    service = _update_service_for_graph(
        test_project,
        workspace=_FakeWorkspace(initialized=True),
        sync=_FakeGraphSync(test_project.root, error=RuntimeError("preflight boom")),
    )

    result = service._run_graph_sync(UpdateOptions())

    assert result.skipped is True
    assert "Graph preflight failed: preflight boom" == result.warning


def test_update_service_reports_missing_graph_credentials(
    test_project, monkeypatch
) -> None:
    from src.services.update_service import UpdateOptions

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = _update_service_for_graph(
        test_project,
        workspace=_FakeWorkspace(initialized=True),
        sync=_FakeGraphSync(test_project.root, _sync_result(test_project.root)),
    )

    result = service._run_graph_sync(UpdateOptions())

    assert result.skipped is True
    assert "OPENAI_API_KEY" in result.warning


def test_update_service_reports_invalid_graph_config(test_project) -> None:
    from src.services.update_service import UpdateOptions

    config = dict(test_project.command_context.config)
    config["graph"] = {
        "provider": "custom",
        "model": "custom-model",
        "embedding_provider": "custom",
        "embedding_model": "custom-embedding",
    }
    service = _update_service_for_graph(
        test_project,
        workspace=_FakeWorkspace(initialized=True),
        sync=_FakeGraphSync(test_project.root, _sync_result(test_project.root)),
    )
    service._config = config

    result = service._run_graph_sync(UpdateOptions())

    assert result.skipped is True
    assert "graph config is invalid" in result.warning
    assert "api_key_env" in result.warning


def test_update_service_reports_index_or_export_failure(
    test_project, monkeypatch
) -> None:
    from src.services.update_service import UpdateOptions

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    service = _update_service_for_graph(
        test_project,
        workspace=_FakeWorkspace(initialized=True),
        sync=_FakeGraphSync(
            test_project.root,
            _sync_result(test_project.root),
            _sync_result(test_project.root),
        ),
        export=_FakeGraphExport(error=RuntimeError("export boom")),
    )

    result = service._run_graph_sync(UpdateOptions())

    assert result.sync_result is not None
    assert result.warning == "Graph index/export failed: export boom"


def test_graph_hash_helpers_handle_missing_and_dict_inputs(tmp_path) -> None:
    missing = tmp_path / "missing.json"
    dict_payload = tmp_path / "dict.json"
    empty_payload = tmp_path / "empty.json"
    invalid_payload = tmp_path / "invalid.json"
    dict_payload.write_text(
        json.dumps({"manifest_hash": "top", "sources": [{"manifest_hash": "row"}]}),
        encoding="utf-8",
    )
    empty_payload.write_text(json.dumps({"sources": [{}]}), encoding="utf-8")
    invalid_payload.write_text("{bad json", encoding="utf-8")

    assert _file_sha256(missing) is None
    assert _input_manifest_hash(dict_payload) == "top"
    assert _input_manifest_hash(empty_payload) is None
    assert _input_manifest_hash(invalid_payload) is None
