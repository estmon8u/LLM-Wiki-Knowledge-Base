"""Tests for test graphrag sync service.

This module belongs to `tests.test_graphrag_sync_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import subprocess
from contextlib import contextmanager

import pandas as pd
import pytest

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.services import graphrag_runtime
from graphwiki_kb.services.graphrag_command_service import (
    GraphRAGCommandError,
    GraphRAGCommandResult,
    GraphRAGCommandService,
)
from graphwiki_kb.services.graphrag_input_sync_service import GraphRAGInputSyncService
from graphwiki_kb.services.graphrag_status_service import GraphRAGStatusService
from graphwiki_kb.services.graphrag_sync_service import (
    GraphRAGSyncService,
    count_source_hash_changes,
    detect_sync_changes,
    graph_input_source_hashes,
    graph_runtime_digest,
)
from graphwiki_kb.services.graphrag_workspace_service import GraphRAGWorkspaceService


def _source_record(
    *,
    content_hash: str = "hash-1",
    text_name: str = "rag.md",
) -> RawSourceRecord:
    """Handles source record.

    Args:
        content_hash: Content hash value used by the operation.
        text_name: Text name value used by the operation.

    Returns:
        RawSourceRecord produced by the operation.
    """
    return RawSourceRecord(
        source_id="src-1",
        slug="rag",
        title="Retrieval-Augmented Generation",
        origin="C:/sources/rag.pdf",
        source_type="file",
        raw_path="raw/sources/rag.pdf",
        normalized_path=f"raw/normalized/{text_name}",
        content_hash=content_hash,
        origin_hash="raw-sha256",
        ingested_at="2026-05-11T00:00:00+00:00",
        metadata={"converter": "mistral-ocr"},
    )


def _write_settings(test_project) -> None:
    """Handles write settings.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file(
        "graph/graphrag/settings.yaml",
        "input:\n"
        "  type: text\n"
        "input_storage:\n"
        "  type: file\n"
        "  base_dir: input\n"
        "chunking:\n"
        "  type: tokens\n",
    )


def _write_source(test_project, *, content_hash: str = "hash-1") -> None:
    """Handles write source.

    Args:
        test_project: Test project value used by the operation.
        content_hash: Content hash value used by the operation.
    """
    test_project.write_file(
        "raw/normalized/rag.md",
        "# Retrieval-Augmented Generation\n\nRAG combines retrieval and generation.\n",
    )
    test_project.services["manifest"].save_source(
        _source_record(content_hash=content_hash)
    )


def _write_complete_output(test_project) -> None:
    """Handles write complete output.

    Args:
        test_project: Test project value used by the operation.
    """
    for table in (
        "documents",
        "text_units",
        "entities",
        "relationships",
        "communities",
        "community_reports",
    ):
        output_path = test_project.root / "graph/graphrag/output" / f"{table}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"id": f"{table}-1"}]).to_parquet(output_path)
    test_project.write_file(
        "graph/graphrag/output/lancedb/vector-store.marker",
        "ready",
    )


def _build_service(test_project, runner) -> GraphRAGSyncService:
    """Handles build service.

    Args:
        test_project: Test project value used by the operation.
        runner: Runner value used by the operation.

    Returns:
        GraphRAGSyncService produced by the operation.
    """
    command_service = GraphRAGCommandService(test_project.paths, runner=runner)
    workspace_service = GraphRAGWorkspaceService(
        test_project.paths,
        command_service,
        config=test_project.config,
    )
    input_sync_service = GraphRAGInputSyncService(
        test_project.paths,
        test_project.services["manifest"],
    )
    status_service = GraphRAGStatusService(test_project.paths)
    return GraphRAGSyncService(
        test_project.paths,
        workspace_service,
        input_sync_service,
        status_service,
        command_service,
    )


def test_sync_holds_workspace_lock_while_applying_workspace_state(
    test_project,
    monkeypatch,
) -> None:
    """Settings, input, and status reads should share the workspace lock."""
    _write_settings(test_project)
    _write_source(test_project)
    service = _build_service(test_project, runner=None)
    held = {"value": False}
    lock_events = []

    @contextmanager
    def fake_workspace_lock(path):
        lock_events.append(("enter", path))
        held["value"] = True
        try:
            yield
        finally:
            held["value"] = False
            lock_events.append(("exit", path))

    monkeypatch.setattr(
        "graphwiki_kb.services.graphrag_sync_service.workspace_lock",
        fake_workspace_lock,
    )
    original_sync_settings = service.workspace_service.sync_settings
    original_input_sync = service.input_sync_service.sync
    original_status = service.status_service.status

    def sync_settings_under_lock(*args, **kwargs):
        assert held["value"] is True
        return original_sync_settings(*args, **kwargs)

    def input_sync_under_lock(*args, **kwargs):
        assert held["value"] is True
        return original_input_sync(*args, **kwargs)

    def status_under_lock(*args, **kwargs):
        assert held["value"] is True
        return original_status(*args, **kwargs)

    monkeypatch.setattr(
        service.workspace_service, "sync_settings", sync_settings_under_lock
    )
    monkeypatch.setattr(service.input_sync_service, "sync", input_sync_under_lock)
    monkeypatch.setattr(service.status_service, "status", status_under_lock)

    result = service.sync(run_index=False)

    assert result.decision.action == "input-only"
    assert lock_events == [
        ("enter", test_project.paths.graph_dir / "graphrag"),
        ("exit", test_project.paths.graph_dir / "graphrag"),
    ]


def _record_successful_run(
    service: GraphRAGSyncService,
    *,
    input_digest: str,
    config_digest: str,
) -> None:
    """Handles record successful run.

    Args:
        service: Service value used by the operation.
        input_digest: Input digest value used by the operation.
        config_digest: Config digest value used by the operation.
    """
    input_path = service.status_service.input_path
    service.status_service.record_index_run(
        method="fast",
        dry_run=False,
        result=GraphRAGCommandResult(
            command=("python", "-m", "graphrag", "index"),
            cwd=service.paths.root,
            returncode=0,
            stdout="indexed",
            stderr="",
        ),
        input_digest=input_digest,
        config_digest=config_digest,
        input_source_count=1,
        source_hashes=graph_input_source_hashes(input_path),
        output_state="complete",
    )


def test_sync_skips_when_sources_config_and_complete_output_match(test_project) -> None:
    """Verifies that sync skips when sources config and complete output match.

    Args:
        test_project: Test project value used by the operation.
    """

    def fail_runner(command, **kwargs):
        """Fail runner.

        Args:
            command: Command value used by the operation.
            kwargs: Kwargs value used by the operation.
        """
        raise AssertionError("index should not run")

    _write_settings(test_project)
    _write_source(test_project)
    service = _build_service(test_project, fail_runner)
    baseline = service.sync(run_index=False)
    _write_complete_output(test_project)
    _record_successful_run(
        service,
        input_digest=baseline.decision.input_digest,
        config_digest=baseline.decision.config_digest,
    )

    result = service.sync()

    assert result.decision.action == "skip"
    assert result.decision.changed_source_count == 0
    assert result.command_result is None


def test_sync_preview_only_has_no_file_side_effects(test_project) -> None:
    """Regression: preflight planning must not rewrite settings or input files."""

    def fail_runner(command, **kwargs):
        """Fail runner."""
        raise AssertionError("index should not run")

    _write_settings(test_project)
    _write_source(test_project)
    settings_path = test_project.paths.graph_dir / "graphrag" / "settings.yaml"
    input_path = test_project.paths.graph_dir / "graphrag" / "input" / "sources.json"
    before_settings = settings_path.read_text(encoding="utf-8")
    service = _build_service(test_project, fail_runner)

    result = service.sync(preview_only=True, dry_run=True)

    assert result.decision.action == "index"
    assert result.decision.method == "fast"
    assert result.input_sync.source_count == 1
    assert result.input_sync.settings_updated is True
    assert settings_path.read_text(encoding="utf-8") == before_settings
    assert not input_path.exists()


def test_sync_does_not_skip_after_latest_failed_attempt(test_project) -> None:
    """Verifies latest failed index metadata forces a retry after prior success."""
    calls = []

    def runner(command, *, cwd, capture_output, text):
        """Runner.

        Args:
            command: Command value used by the operation.
            cwd: Cwd value used by the operation.
            capture_output: Capture output value used by the operation.
            text: Text content being processed.
        """
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="retried\n", stderr="")

    _write_settings(test_project)
    _write_source(test_project)
    service = _build_service(test_project, runner)
    baseline = service.sync(run_index=False)
    _write_complete_output(test_project)
    _record_successful_run(
        service,
        input_digest=baseline.decision.input_digest,
        config_digest=baseline.decision.config_digest,
    )
    service.status_service.record_index_run(
        method="fast-update",
        dry_run=False,
        result=GraphRAGCommandResult(
            command=("python", "-m", "graphrag", "index"),
            cwd=service.paths.root,
            returncode=2,
            stdout="",
            stderr="index failed",
        ),
        input_digest=baseline.decision.input_digest,
        config_digest=baseline.decision.config_digest,
        input_source_count=1,
        source_hashes=graph_input_source_hashes(service.status_service.input_path),
        output_state="complete",
    )

    result = service.sync()

    assert result.decision.action == "index"
    assert result.decision.method == "fast-update"
    assert result.decision.reason == "Previous GraphRAG index attempt failed."
    assert calls[0][calls[0].index("--method") + 1] == "fast-update"


def test_sync_uses_incremental_update_when_source_hash_changes(test_project) -> None:
    """Verifies that sync uses incremental update when source hash changes.

    Args:
        test_project: Test project value used by the operation.
    """
    calls = []

    def runner(command, *, cwd, capture_output, text):
        """Runner.

        Args:
            command: Command value used by the operation.
            cwd: Cwd value used by the operation.
            capture_output: Capture output value used by the operation.
            text: Text content being processed.
        """
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="updated\n", stderr="")

    _write_settings(test_project)
    _write_source(test_project)
    service = _build_service(test_project, runner)
    baseline = service.sync(run_index=False)
    _write_complete_output(test_project)
    _record_successful_run(
        service,
        input_digest=baseline.decision.input_digest,
        config_digest=baseline.decision.config_digest,
    )
    _write_source(test_project, content_hash="hash-2")

    result = service.sync()

    assert result.decision.action == "index"
    assert result.decision.method == "fast-update"
    assert result.decision.changed_source_count == 1
    assert calls[0][calls[0].index("--method") + 1] == "fast-update"


def test_sync_rebuilds_when_graph_runtime_config_changes(test_project) -> None:
    """Verifies that sync rebuilds when graph runtime config changes.

    Args:
        test_project: Test project value used by the operation.
    """
    calls = []

    def runner(command, *, cwd, capture_output, text):
        """Runner.

        Args:
            command: Command value used by the operation.
            cwd: Cwd value used by the operation.
            capture_output: Capture output value used by the operation.
            text: Text content being processed.
        """
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="rebuilt\n", stderr="")

    _write_settings(test_project)
    _write_source(test_project)
    service = _build_service(test_project, runner)
    baseline = service.sync(run_index=False)
    _write_complete_output(test_project)
    _record_successful_run(
        service,
        input_digest=baseline.decision.input_digest,
        config_digest=baseline.decision.config_digest,
    )
    test_project.config["graph"]["model"] = "gpt-5.5"

    result = service.sync()

    assert result.decision.action == "index"
    assert result.decision.method == "fast"
    assert result.decision.config_changed is True
    assert (
        result.decision.reason
        == "Graph runtime settings, prompts, GraphRAG version, or schema changed."
    )
    assert calls[0][calls[0].index("--method") + 1] == "fast"


def test_sync_rebuilds_partial_output_instead_of_incremental_update(
    test_project,
) -> None:
    """Verifies that sync rebuilds partial output instead of incremental update.

    Args:
        test_project: Test project value used by the operation.
    """
    calls = []

    def runner(command, *, cwd, capture_output, text):
        """Runner.

        Args:
            command: Command value used by the operation.
            cwd: Cwd value used by the operation.
            capture_output: Capture output value used by the operation.
            text: Text content being processed.
        """
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="rebuilt\n", stderr="")

    _write_settings(test_project)
    _write_source(test_project)
    test_project.write_file("graph/graphrag/output/entities.parquet", "")
    service = _build_service(test_project, runner)

    result = service.sync()

    assert result.decision.action == "index"
    assert result.decision.method == "fast"
    assert result.decision.output_state == "partial"
    assert result.decision.reason == "Graph index output is partial or incomplete."
    assert calls[0][calls[0].index("--method") + 1] == "fast"


def test_sync_skips_index_when_synced_input_has_no_documents(test_project) -> None:
    """Verifies that sync skips index when synced input has no documents.

    Args:
        test_project: Test project value used by the operation.
    """

    def fail_runner(command, **kwargs):
        """Fail runner.

        Args:
            command: Command value used by the operation.
            kwargs: Kwargs value used by the operation.
        """
        raise AssertionError("index should not run")

    _write_settings(test_project)
    service = _build_service(test_project, fail_runner)

    result = service.sync()

    assert result.decision.action == "skip"
    assert result.decision.method is None
    assert "no documents" in result.decision.reason


def test_sync_force_coerces_update_method_to_full_rebuild(test_project) -> None:
    """Verifies that sync force coerces update method to full rebuild.

    Args:
        test_project: Test project value used by the operation.
    """
    calls = []

    def runner(command, *, cwd, capture_output, text):
        """Runner.

        Args:
            command: Command value used by the operation.
            cwd: Cwd value used by the operation.
            capture_output: Capture output value used by the operation.
            text: Text content being processed.
        """
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="rebuilt\n", stderr="")

    _write_settings(test_project)
    _write_source(test_project)
    service = _build_service(test_project, runner)
    baseline = service.sync(run_index=False)
    _write_complete_output(test_project)
    _record_successful_run(
        service,
        input_digest=baseline.decision.input_digest,
        config_digest=baseline.decision.config_digest,
    )

    result = service.sync(method="fast-update", force=True)

    assert result.decision.action == "index"
    assert result.decision.method == "fast"
    assert result.decision.reason == "--force requested a full graph rebuild."
    assert calls[0][calls[0].index("--method") + 1] == "fast"


def test_sync_respects_explicit_full_method_override(test_project) -> None:
    """Verifies that sync respects explicit method override.

    Args:
        test_project: Test project value used by the operation.
    """
    calls = []

    def runner(command, *, cwd, capture_output, text):
        """Runner.

        Args:
            command: Command value used by the operation.
            cwd: Cwd value used by the operation.
            capture_output: Capture output value used by the operation.
            text: Text content being processed.
        """
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="updated\n", stderr="")

    _write_settings(test_project)
    _write_source(test_project)
    service = _build_service(test_project, runner)

    result = service.sync(method="standard")

    assert result.decision.action == "index"
    assert result.decision.method == "standard"
    assert result.decision.reason == (
        "Explicit GraphRAG index method requested: standard."
    )
    assert calls[0][calls[0].index("--method") + 1] == "standard"


def test_sync_coerces_explicit_update_method_when_output_missing(
    test_project,
) -> None:
    """Regression: update methods require complete existing graph output."""
    calls = []

    def runner(command, *, cwd, capture_output, text):
        """Runner."""
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="rebuilt\n", stderr="")

    _write_settings(test_project)
    _write_source(test_project)
    service = _build_service(test_project, runner)

    result = service.sync(method="standard-update")

    assert result.decision.action == "index"
    assert result.decision.method == "standard"
    assert "requires complete existing output" in result.decision.reason
    assert calls[0][calls[0].index("--method") + 1] == "standard"


def test_sync_rebuilds_complete_output_when_metadata_is_missing(test_project) -> None:
    """Verifies that sync rebuilds complete output when metadata is missing.

    Args:
        test_project: Test project value used by the operation.
    """
    calls = []

    def runner(command, *, cwd, capture_output, text):
        """Runner.

        Args:
            command: Command value used by the operation.
            cwd: Cwd value used by the operation.
            capture_output: Capture output value used by the operation.
            text: Text content being processed.
        """
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="rebuilt\n", stderr="")

    _write_settings(test_project)
    _write_source(test_project)
    _write_complete_output(test_project)
    service = _build_service(test_project, runner)

    result = service.sync()

    assert result.decision.action == "index"
    assert result.decision.method == "fast"
    assert result.decision.stale_metadata is True
    assert result.decision.reason == (
        "Graph index provenance metadata is missing; rebuilding once."
    )
    assert calls[0][calls[0].index("--method") + 1] == "fast"


def test_sync_records_failed_index_run_before_reraising(test_project) -> None:
    """Verifies that sync records failed index run before reraising.

    Args:
        test_project: Test project value used by the operation.
    """

    def runner(command, *, cwd, capture_output, text):
        """Runner.

        Args:
            command: Command value used by the operation.
            cwd: Cwd value used by the operation.
            capture_output: Capture output value used by the operation.
            text: Text content being processed.
        """
        return subprocess.CompletedProcess(
            command,
            2,
            stdout="",
            stderr="index failed\n",
        )

    _write_settings(test_project)
    _write_source(test_project)
    service = _build_service(test_project, runner)

    with pytest.raises(GraphRAGCommandError, match="index failed"):
        service.sync()

    runs = service.status_service._load_runs()
    assert runs[0]["success"] is False
    assert runs[0]["method"] == "fast"
    assert runs[0]["input_digest"]
    assert runs[0]["config_digest"]
    assert runs[0]["source_hashes"] == {"src-1": "hash-1"}
    assert runs[0]["output_state"] == "missing"


def test_sync_records_early_index_failure_without_result(test_project) -> None:
    _write_settings(test_project)
    _write_source(test_project)
    service = _build_service(
        test_project,
        lambda command, *, cwd, capture_output, text: subprocess.CompletedProcess(
            command, 0, stdout="", stderr=""
        ),
    )

    def fail_before_result(**kwargs):
        raise GraphRAGCommandError("runtime contract failed before command result")

    service.command_service.index = fail_before_result

    with pytest.raises(GraphRAGCommandError, match="runtime contract failed"):
        service.sync()

    runs = service.status_service._load_runs()
    assert runs[0]["success"] is False
    assert runs[0]["stderr_tail"] == "runtime contract failed before command result"
    assert runs[0]["method"] == "fast"


def test_graph_runtime_digest_includes_prompt_files(test_project) -> None:
    """Verifies that graph runtime digest includes prompt files.

    Args:
        test_project: Test project value used by the operation.
    """
    _write_settings(test_project)
    before = graph_runtime_digest(test_project.paths.graph_dir / "graphrag")
    test_project.write_file("graph/graphrag/prompts/entity_extraction.txt", "Prompt A")

    after = graph_runtime_digest(test_project.paths.graph_dir / "graphrag")

    assert after != before


def test_graph_runtime_digest_includes_graphrag_version(
    test_project,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies GraphRAG package upgrades invalidate runtime digest."""
    _write_settings(test_project)
    monkeypatch.setattr(
        graphrag_runtime,
        "installed_graphrag_version",
        lambda: "3.0.9",
    )
    before = graph_runtime_digest(test_project.paths.graph_dir / "graphrag")

    monkeypatch.setattr(
        graphrag_runtime,
        "installed_graphrag_version",
        lambda: "3.0.10",
    )
    after = graph_runtime_digest(test_project.paths.graph_dir / "graphrag")

    assert after != before


def test_graph_input_source_hashes_supports_object_payload_and_ignores_bad_records(
    test_project,
) -> None:
    """Verifies that graph input source hashes supports object payload and ignores bad records.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file(
        "graph/graphrag/input/sources.json",
        (
            '{"sources": ['
            '{"id": "src-1", "source_hash": "hash-1"},'
            '"not-a-record",'
            '{"id": "src-2"}'
            "]}"
        ),
    )

    assert graph_input_source_hashes(
        test_project.paths.graph_dir / "graphrag" / "input" / "sources.json"
    ) == {"src-1": "hash-1"}


def test_count_source_hash_changes_handles_missing_previous_snapshot() -> None:
    """Verifies that count source hash changes handles missing previous snapshot."""
    assert count_source_hash_changes(None, {"src-1": "hash-1"}) is None


def test_detect_sync_changes_separates_input_config_and_metadata(
    test_project,
) -> None:
    """Verifies sync planning inputs are testable without running the planner."""
    _write_settings(test_project)
    _write_source(test_project)
    _write_complete_output(test_project)
    status = GraphRAGStatusService(test_project.paths).status()

    change_state = detect_sync_changes(
        status=status,
        input_digest="current-input",
        config_digest="current-config",
        current_source_hashes={"src-1": "hash-2"},
        last_successful_run={
            "input_digest": "previous-input",
            "config_digest": "current-config",
            "source_hashes": {"src-1": "hash-1"},
        },
    )

    assert change_state.output_state == "complete"
    assert change_state.input_changed is True
    assert change_state.config_changed is False
    assert change_state.changed_source_count == 1
    assert change_state.stale_metadata is False
