"""Tests for test phase9 graph cli.

This module belongs to `tests.test_phase9_graph_cli` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from graphwiki_kb.cli import main
from graphwiki_kb.commands.find import _merge_results
from graphwiki_kb.models.wiki_models import SearchResult
from graphwiki_kb.services.compile_service import CompileResult
from graphwiki_kb.services.concept_service import ConceptGenerationResult
from graphwiki_kb.services.graphrag_command_service import GraphRAGCommandResult
from graphwiki_kb.services.graphrag_input_sync_service import GraphRAGInputSyncResult
from graphwiki_kb.services.graphrag_query_service import GraphRAGQueryAnswer
from graphwiki_kb.services.graphrag_status_service import (
    GraphRAGIndexRun,
    GraphRAGStatusService,
)
from graphwiki_kb.services.graphrag_sync_service import (
    GraphRAGSyncDecision,
    GraphRAGSyncResult,
)
from graphwiki_kb.services.graphrag_wiki_export_service import GraphRAGWikiExportResult
from graphwiki_kb.services.lint_service import _file_sha256, _input_manifest_hash
from graphwiki_kb.services.project_service import build_project_paths
from graphwiki_kb.services.update_service import (
    GraphUpdateResult,
    IngestSummary,
    UpdateResult,
)
from tests.test_cli import _CliFakeProvider, _set_provider_config


def _write_graph_tables(root: Path) -> None:
    """Handles write graph tables.

    Args:
        root: Root path used for discovery or relative path resolution.
    """
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
    vector_marker = output_dir / "lancedb" / "vector-store.marker"
    vector_marker.parent.mkdir(parents=True, exist_ok=True)
    vector_marker.write_text("ready", encoding="utf-8")


def _graph_answer(*, saved_path: str | None = None) -> GraphRAGQueryAnswer:
    """Handles graph answer.

    Args:
        saved_path: Saved path value used by the operation.

    Returns:
        GraphRAGQueryAnswer produced by the operation.
    """
    return GraphRAGQueryAnswer(
        question="What changed?",
        answer="GraphRAG answered from the graph.",
        raw_output="GraphRAG answered from the graph.",
        method="drift",
        created_at="2026-05-12T00:00:00+00:00",
        index_run_id="run-1",
        command=("python", "-m", "graphrag", "query"),
        stdout="GraphRAG answered from the graph.",
        stderr="",
        graph_input_hash="graph-input-hash-1",
        input_manifest_hash="manifest-hash-1",
        saved_path=saved_path,
        retriever="graph",
        planner="heuristic",
        route_reason="comparison question",
        claim_support="graph-index-answer",
        source_trace={
            "input_path": "graph/graphrag/input/sources.json",
            "output_dir": "graph/graphrag/output",
            "graph_input_hash": "graph-input-hash-1",
            "input_manifest_hash": "manifest-hash-1",
        },
    )


def _input_sync_result(root: Path) -> GraphRAGInputSyncResult:
    """Handles input sync result.

    Args:
        root: Root path used for discovery or relative path resolution.

    Returns:
        GraphRAGInputSyncResult produced by the operation.
    """
    return GraphRAGInputSyncResult(
        source_count=1,
        output_path=root / "graph/graphrag/input/sources.json",
        settings_path=root / "graph/graphrag/settings.yaml",
        metadata_fields=("source_id",),
        settings_updated=False,
    )


def _decision(
    *,
    action: str = "index",
    method: str | None = "fast",
    output_state: str = "missing",
):
    """Handles decision.

    Args:
        action: Action value used by the operation.
        method: Method value used by the operation.
    """
    return GraphRAGSyncDecision(
        action=action,
        method=method,
        reason="test decision",
        output_state=output_state,
        input_digest="input-digest",
        config_digest="config-digest",
        input_changed=True,
        config_changed=False,
        changed_source_count=1,
        cost_warning="GraphRAG cost warning.",
    )


def _sync_result(
    root: Path,
    *,
    action: str = "index",
    method: str | None = "fast",
    output_state: str = "missing",
):
    """Handles sync result.

    Args:
        root: Root path used for discovery or relative path resolution.
        action: Action value used by the operation.
        method: Method value used by the operation.
    """
    return GraphRAGSyncResult(
        input_sync=_input_sync_result(root),
        decision=_decision(action=action, method=method, output_state=output_state),
    )


def _index_run() -> GraphRAGIndexRun:
    """Handles index run.

    Returns:
        GraphRAGIndexRun produced by the operation.
    """
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
        active_output_dir="graph/graphrag/output/active",
    )


def test_graph_command_group_is_removed() -> None:
    """Verifies that graph command group is removed."""
    result = CliRunner().invoke(main, ["graph"])

    assert result.exit_code != 0
    assert "No such command 'graph'" in result.output


def test_ask_prints_graph_answer_metadata_and_source_trace() -> None:
    """Verifies that ask prints graph answer metadata and source trace."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        with patch(
            "graphwiki_kb.services.graph_ask_controller_service.GraphAskControllerService.ask",
            return_value=_graph_answer(saved_path="wiki/analysis/question.md"),
        ):
            result = runner.invoke(
                main,
                [
                    "ask",
                    "--engine",
                    "graphrag",
                    "--show-source-trace",
                    "--save",
                    "What",
                    "changed?",
                ],
            )

        assert result.exit_code == 0
        assert "retriever: graph" in result.output
        assert "Source Trace" in result.output
        assert "GraphRAG input: graph/graphrag/input/sources.json" in result.output
        assert "Route reason: comparison question" in result.output
        assert "Support level: graph-index-answer" in result.output
        assert "GraphRAG answered from the graph." in result.output
        assert "Saved analysis page: wiki/analysis/question.md" in result.output


def test_ask_json_outputs_graph_answer_payload() -> None:
    """Verifies that ask json outputs graph answer payload."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        with patch(
            "graphwiki_kb.services.graph_ask_controller_service.GraphAskControllerService.ask",
            return_value=_graph_answer(),
        ):
            result = runner.invoke(
                main,
                ["ask", "--engine", "graphrag", "--json", "What", "changed?"],
            )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["retriever"] == "graph"
        assert payload["method"] == "drift"
        assert payload["command"] == ["python", "-m", "graphrag", "query"]


def test_find_json_searches_direct_graph_artifacts() -> None:
    """Verifies top-level find searches GraphRAG parquet entities/relationships."""
    runner = CliRunner(mix_stderr=False)
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        _write_graph_tables(Path.cwd())

        result = runner.invoke(main, ["find", "--json", "REALM"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["retriever"] == "graph-and-wiki-index"
        assert payload["results"][0]["retriever"] == "graphrag-artifacts"
        assert payload["results"][0]["path"] == "graph://relationships/rel-1"
        assert payload["results"][0]["section"] == "GraphRAG Relationship"


def test_find_json_reports_unreadable_graph_artifacts() -> None:
    runner = CliRunner(mix_stderr=False)
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        output = Path("graph/graphrag/output")
        output.mkdir(parents=True, exist_ok=True)
        (output / "entities.parquet").write_text("not parquet", encoding="utf-8")

        result = runner.invoke(main, ["find", "--json", "RAG"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert any("entities artifacts" in item for item in payload["diagnostics"])


def test_find_merges_graph_and_wiki_before_global_ranking() -> None:
    """Verifies weak graph hits do not starve stronger wiki candidates."""
    graph_result = SearchResult(
        title="REALM",
        path="graph://entities/realm",
        score=1.0,
        snippet="Weak graph mention.",
        section="GraphRAG Entity",
    )
    wiki_result = SearchResult(
        title="REALM source page",
        path="wiki/sources/realm.md",
        score=35.0,
        snippet="Strong wiki match.",
        section="Overview",
    )

    results = _merge_results([graph_result], [wiki_result], limit=1)

    assert results[0].path == wiki_result.path
    assert results[0].score == pytest.approx(1 / 61)


def test_ask_streaming_option_is_removed() -> None:
    """Verifies the hidden unsupported streaming option is no longer wired."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["ask", "--streaming", "What", "changed?"])

        assert result.exit_code != 0
        assert "No such option: --streaming" in result.output


def test_ask_limit_is_rejected_for_graphrag_queries() -> None:
    """Verifies deprecated retrieval limits do not silently do nothing."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["ask", "--limit", "2", "What", "changed?"])

        assert result.exit_code != 0
        assert "--limit is not supported by kb ask" in result.output


def test_update_output_renders_resume_removed_concepts_and_graph_details() -> None:
    """Verifies that update output renders resume removed concepts and graph details."""
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
        search_warning="Search index refresh skipped because SQLite FTS5 is unavailable.",
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
            "graphwiki_kb.services.update_service.UpdateService.run",
            return_value=update_result,
        ):
            result = runner.invoke(main, ["update"])

        assert result.exit_code == 0
        assert "resumed interrupted update run compile-run-1" in result.output
        assert "Removed 1 stale concept page(s)" in result.output
        assert "Search Summary" in result.output
        assert "SQLite FTS5 is unavailable" in result.output
        assert "initialized graph workspace" in result.output
        assert "GraphRAG cost warning." in result.output
        assert "Graph index run: run-1 (fast)" in result.output
        assert "Graph output: graph/graphrag/output/active" in result.output
        assert "Graph wiki export: 1 page(s)" in result.output


def test_update_output_renders_non_index_graph_decision() -> None:
    """Verifies that update output renders non index graph decision."""
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
            "graphwiki_kb.services.update_service.UpdateService.run",
            return_value=update_result,
        ):
            result = runner.invoke(main, ["update"])

        assert result.exit_code == 0
        assert "Graph index action: test decision" in result.output


def test_update_passes_explicit_graph_method_to_preflight(test_project) -> None:
    """The CLI and update service expose GraphRAG's documented index methods."""
    from graphwiki_kb.services.update_service import UpdateOptions

    sync = _FakeGraphSync(
        test_project.root,
        _sync_result(test_project.root, action="skip", method=None),
    )
    service = _update_service_for_graph(
        test_project,
        workspace=_FakeWorkspace(initialized=True),
        sync=sync,
    )

    service._run_graph_sync(UpdateOptions(graph_method="standard"))

    assert sync.calls[0]["method"] == "standard"


def test_init_sets_up_graph_workspace_settings() -> None:
    """Verifies that init sets up graph workspace settings."""
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
        assert settings["vector_store"]["db_uri"] == "output/lancedb"
        assert Path("graph/graphrag/prompts/extract_graph.txt").exists()


def test_update_runs_graph_sync_index_and_export(monkeypatch) -> None:
    """Verifies that update runs graph sync index and export.

    Args:
        monkeypatch: Monkeypatch value used by the operation.
    """
    calls = []

    def fake_init(self, *, workspace_dir, model, embedding, force):
        """Simulate GraphRAG Python API workspace initialization."""
        calls.append(("init", workspace_dir, model, embedding, force))
        settings_path = self.paths.root / "graph" / "graphrag" / "settings.yaml"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text("input:\n  type: text\n", encoding="utf-8")
        return GraphRAGCommandResult(
            command=("graphrag.api", "initialize_project_at"),
            cwd=self.paths.root,
            returncode=0,
            stdout="initialized\n",
            stderr="",
        )

    def fake_index(self, *, workspace_dir, method, **kwargs):
        """Simulate GraphRAG Python API indexing and output artifacts."""
        calls.append(("index", workspace_dir, method))
        _write_graph_tables(self.paths.root)
        return GraphRAGCommandResult(
            command=("graphrag.api", "build_index", "--method", method),
            cwd=self.paths.root,
            returncode=0,
            stdout="indexed\n",
            stderr="",
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "graphwiki_kb.services.graphrag_command_service.GraphRAGApiBackend.init_workspace",
        fake_init,
    )
    monkeypatch.setattr(
        "graphwiki_kb.services.graphrag_command_service.GraphRAGApiBackend.index",
        fake_index,
    )
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text(
            "# Sample\n\nGraph update body.\n", encoding="utf-8"
        )
        assert runner.invoke(main, ["init"]).exit_code == 0
        _set_provider_config()

        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            result = runner.invoke(main, ["update", "sample.md"])

        assert result.exit_code == 0
        assert "GraphRAG Summary" in result.output
        assert "Graph index run:" in result.output
        assert "Graph wiki export:" in result.output
        assert Path("graph/graphrag/input/sources.json").exists()
        assert Path("wiki/graph/index.md").exists()
        assert any(call[0] == "index" for call in calls)


def test_update_no_graph_flag_skips_graph(monkeypatch) -> None:
    """Verifies that update no graph flag skips graph.

    Args:
        monkeypatch: Monkeypatch value used by the operation.
    """

    def fail_run(*args, **kwargs):
        """Fail run.

        Args:
            args: Parsed or forwarded command arguments.
            kwargs: Kwargs value used by the operation.
        """
        raise AssertionError("GraphRAG should not run with --no-graph")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("sample.md").write_text("# Sample\n\nNo graph body.\n", encoding="utf-8")
        assert runner.invoke(main, ["init"]).exit_code == 0
        monkeypatch.setattr(
            "graphwiki_kb.services.graphrag_command_service.GraphRAGApiBackend.index",
            fail_run,
        )
        _set_provider_config()

        with patch(
            "graphwiki_kb.services.build_provider", return_value=_CliFakeProvider()
        ):
            result = runner.invoke(main, ["update", "--no-graph", "sample.md"])

        assert result.exit_code == 0
        assert "Graph skipped: --no-graph requested." in result.output


def test_status_json_includes_graph_status() -> None:
    """Verifies that status json includes graph status."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["status", "--json"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["graph_status"]["workspace_initialized"] is True
        assert payload["graph_status"]["workspace_dir"] == "graph/graphrag"
        assert payload["graph_status"]["state"] == "missing"
        assert "documents" in payload["graph_status"]["missing_tables"]
        assert "graph" not in payload


def test_status_strict_fails_when_graph_is_not_ready() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0

        result = runner.invoke(main, ["status", "--strict", "--json"])

        assert result.exit_code == 1
        payload = json.loads(result.output)
        assert payload["strict_ok"] is False
        assert "GraphRAG input has no documents" in payload["strict_failures"]


def test_status_human_output_includes_last_graph_index() -> None:
    """Verifies that status human output includes last graph index."""
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
    """Verifies that export includes graph wiki export."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        assert runner.invoke(main, ["init"]).exit_code == 0
        _write_graph_tables(Path.cwd())

        result = runner.invoke(main, ["export"])

        assert result.exit_code == 0
        assert "Vault Export" in result.output
        assert "Graph Wiki Export" in result.output
        assert Path("wiki/graph/index.md").exists()
        assert Path("vault/obsidian/graph/index.md").exists()


class _FakeWorkspace:
    """Represents fake workspace behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    def __init__(self, *, initialized: bool) -> None:
        """Initializes the instance.

        Args:
            initialized: Initialized value used by the operation.
        """
        self.initialized = initialized
        self.ensured = False

    def is_initialized(self) -> bool:
        """Is initialized.

        Returns:
            bool produced by the operation.
        """
        return self.initialized

    def ensure_workspace(self) -> None:
        """Ensure workspace."""
        self.ensured = True
        self.initialized = True


class _FakeGraphSync:
    """Represents fake graph sync behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    def __init__(self, root: Path, *results, error: Exception | None = None) -> None:
        """Initializes the instance.

        Args:
            root: Root path used for discovery or relative path resolution.
            results: Results value used by the operation.
            error: Error value used by the operation.
        """
        self.workspace_dir = root / "graph/graphrag"
        self.results = list(results)
        self.error = error
        self.calls = []

    def sync(self, **kwargs):
        """Sync.

        Args:
            kwargs: Kwargs value used by the operation.
        """
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.results.pop(0)


class _FakeGraphExport:
    """Represents fake graph export behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    def __init__(self, *, error: Exception | None = None) -> None:
        """Initializes the instance.

        Args:
            error: Error value used by the operation.
        """
        self.error = error
        self.calls = 0

    def export_wiki(self):
        """Export wiki."""
        self.calls += 1
        if self.error is not None:
            raise self.error
        return GraphRAGWikiExportResult(
            exported_paths=["wiki/graph/index.md"],
            table_counts={},
        )


def _update_service_for_graph(test_project, *, workspace, sync, export=None):
    """Handles update service for graph.

    Args:
        test_project: Test project value used by the operation.
        workspace: Workspace value used by the operation.
        sync: Sync value used by the operation.
        export: Export value used by the operation.
    """
    from graphwiki_kb.services.update_service import UpdateService

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
    """Verifies that update service skips graph when services or config are missing.

    Args:
        test_project: Test project value used by the operation.
    """
    from graphwiki_kb.services.update_service import UpdateOptions, UpdateService

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
    """Verifies that update service initializes and skips when preflight skips.

    Args:
        test_project: Test project value used by the operation.
    """
    from graphwiki_kb.services.update_service import UpdateOptions

    workspace = _FakeWorkspace(initialized=False)
    sync = _FakeGraphSync(
        test_project.root,
        _sync_result(test_project.root, action="skip", method=None),
    )
    service = _update_service_for_graph(
        test_project,
        workspace=workspace,
        sync=sync,
    )

    result = service._run_graph_sync(UpdateOptions(allow_partial=True))

    assert workspace.ensured is True
    assert result.initialized is True
    assert result.skipped is True
    assert result.skip_reason == "test decision"
    assert sync.calls[0]["allow_missing_sources"] is True


def test_update_service_exports_graph_wiki_when_index_skips_with_complete_output(
    test_project,
) -> None:
    """Verifies that a current GraphRAG index still refreshes graph wiki pages."""
    from graphwiki_kb.services.update_service import UpdateOptions

    graph_export = _FakeGraphExport()
    sync = _FakeGraphSync(
        test_project.root,
        _sync_result(
            test_project.root,
            action="skip",
            method=None,
            output_state="complete",
        ),
        _sync_result(
            test_project.root,
            action="input-only",
            method=None,
            output_state="complete",
        ),
    )
    service = _update_service_for_graph(
        test_project,
        workspace=_FakeWorkspace(initialized=True),
        sync=sync,
        export=graph_export,
    )

    result = service._run_graph_sync(UpdateOptions())

    assert result.skipped is True
    assert result.skip_reason == "test decision"
    assert result.sync_result is not None
    assert sync.calls[1]["run_index"] is False
    assert sync.calls[1]["preview_only"] is False
    assert result.export_result is not None
    assert graph_export.calls == 1


def test_update_service_reports_preflight_failure(test_project) -> None:
    """Verifies that update service reports preflight failure.

    Args:
        test_project: Test project value used by the operation.
    """
    from graphwiki_kb.services.update_service import UpdateOptions

    service = _update_service_for_graph(
        test_project,
        workspace=_FakeWorkspace(initialized=True),
        sync=_FakeGraphSync(test_project.root, error=RuntimeError("preflight boom")),
    )

    result = service._run_graph_sync(UpdateOptions(allow_partial=True))

    assert result.skipped is True
    assert result.warning == "Graph preflight failed: preflight boom"


def test_update_service_reports_missing_graph_credentials(
    test_project, monkeypatch
) -> None:
    """Verifies that update service reports missing graph credentials.

    Args:
        test_project: Test project value used by the operation.
        monkeypatch: Monkeypatch value used by the operation.
    """
    from graphwiki_kb.services.update_service import UpdateOptions

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = _update_service_for_graph(
        test_project,
        workspace=_FakeWorkspace(initialized=True),
        sync=_FakeGraphSync(test_project.root, _sync_result(test_project.root)),
    )

    result = service._run_graph_sync(UpdateOptions())

    assert result.skipped is True
    assert "OPENAI_API_KEY" in result.warning


def test_update_service_graph_only_requires_graph_credentials(
    test_project, monkeypatch
) -> None:
    """Verifies that graph-only updates fail clearly when graph credentials are absent."""
    from graphwiki_kb.services.update_service import UpdateOptions

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = _update_service_for_graph(
        test_project,
        workspace=_FakeWorkspace(initialized=True),
        sync=_FakeGraphSync(test_project.root, _sync_result(test_project.root)),
    )

    try:
        service._run_graph_sync(UpdateOptions(graph_only=True))
    except ValueError as exc:
        assert "provider credentials are missing" in str(exc)
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("graph-only update should fail without graph credentials")


def test_update_service_reports_invalid_graph_config(test_project) -> None:
    """Verifies that update service reports invalid graph config.

    Args:
        test_project: Test project value used by the operation.
    """
    from graphwiki_kb.services.update_service import UpdateOptions

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

    result = service._run_graph_sync(UpdateOptions(allow_partial=True))

    assert result.skipped is True
    assert "graph config is invalid" in result.warning
    assert "api_key_env" in result.warning


def test_update_service_reports_index_or_export_failure(
    test_project, monkeypatch
) -> None:
    """Verifies that update service reports index or export failure.

    Args:
        test_project: Test project value used by the operation.
        monkeypatch: Monkeypatch value used by the operation.
    """
    from graphwiki_kb.services.update_service import UpdateOptions

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    sync = _FakeGraphSync(
        test_project.root,
        _sync_result(test_project.root),
        _sync_result(test_project.root),
    )
    service = _update_service_for_graph(
        test_project,
        workspace=_FakeWorkspace(initialized=True),
        sync=sync,
        export=_FakeGraphExport(error=RuntimeError("export boom")),
    )

    result = service._run_graph_sync(UpdateOptions(allow_partial=True))

    assert result.sync_result is not None
    assert result.warning == "Graph index/export failed: export boom"
    assert sync.calls[0]["allow_missing_sources"] is True
    assert sync.calls[1]["allow_missing_sources"] is True


def test_update_service_fails_graph_errors_unless_partial_allowed(
    test_project,
) -> None:
    """Verifies that graph failures are hard failures by default."""
    from graphwiki_kb.services.update_service import UpdateOptions

    service = _update_service_for_graph(
        test_project,
        workspace=_FakeWorkspace(initialized=True),
        sync=_FakeGraphSync(test_project.root, error=RuntimeError("preflight boom")),
    )

    try:
        service._run_graph_sync(UpdateOptions())
    except ValueError as exc:
        assert "Graph preflight failed: preflight boom" in str(exc)
    else:
        raise AssertionError("Graph preflight failure should be a hard failure")


def test_update_graph_only_skips_legacy_provider_preflight(
    test_project, monkeypatch
) -> None:
    """Verifies that graph-only update skips legacy compile/provider work."""
    from graphwiki_kb.services.update_service import UpdateOptions

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    sync = _FakeGraphSync(
        test_project.root,
        _sync_result(test_project.root),
        _sync_result(test_project.root),
    )
    service = _update_service_for_graph(
        test_project,
        workspace=_FakeWorkspace(initialized=True),
        sync=sync,
    )
    service._config["provider"] = {}

    result = service.run(UpdateOptions(graph_only=True))

    assert result.compile_result is None
    assert result.graph_result is not None
    assert result.graph_result.export_result is not None
    assert sync.calls[0]["allow_missing_sources"] is False
    assert sync.calls[1]["allow_missing_sources"] is False


def test_graph_hash_helpers_handle_missing_and_dict_inputs(tmp_path) -> None:
    """Verifies that graph hash helpers handle missing and dict inputs.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
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
