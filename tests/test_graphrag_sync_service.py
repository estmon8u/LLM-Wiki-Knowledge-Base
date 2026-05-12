from __future__ import annotations

import subprocess

import pytest

from src.models.source_models import RawSourceRecord
from src.services.graphrag_command_service import (
    GraphRAGCommandError,
    GraphRAGCommandResult,
    GraphRAGCommandService,
)
from src.services.graphrag_input_sync_service import GraphRAGInputSyncService
from src.services.graphrag_status_service import GraphRAGStatusService
from src.services.graphrag_sync_service import (
    GraphRAGSyncService,
    count_source_hash_changes,
    graph_input_source_hashes,
    graph_runtime_digest,
)
from src.services.graphrag_workspace_service import GraphRAGWorkspaceService


def _source_record(
    *,
    content_hash: str = "hash-1",
    text_name: str = "rag.md",
) -> RawSourceRecord:
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
    test_project.write_file(
        "raw/normalized/rag.md",
        "# Retrieval-Augmented Generation\n\nRAG combines retrieval and generation.\n",
    )
    test_project.services["manifest"].save_source(
        _source_record(content_hash=content_hash)
    )


def _write_complete_output(test_project) -> None:
    for table in (
        "documents",
        "text_units",
        "entities",
        "relationships",
        "communities",
        "community_reports",
    ):
        test_project.write_file(f"graph/graphrag/output/{table}.parquet", "")


def _build_service(test_project, runner) -> GraphRAGSyncService:
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


def _record_successful_run(
    service: GraphRAGSyncService,
    *,
    input_digest: str,
    config_digest: str,
) -> None:
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
    def fail_runner(command, **kwargs):
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


def test_sync_uses_incremental_update_when_source_hash_changes(test_project) -> None:
    calls = []

    def runner(command, *, cwd, capture_output, text):
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
    calls = []

    def runner(command, *, cwd, capture_output, text):
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
    assert result.decision.reason == "Graph runtime settings or prompts changed."
    assert calls[0][calls[0].index("--method") + 1] == "fast"


def test_sync_rebuilds_partial_output_instead_of_incremental_update(
    test_project,
) -> None:
    calls = []

    def runner(command, *, cwd, capture_output, text):
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
    def fail_runner(command, **kwargs):
        raise AssertionError("index should not run")

    _write_settings(test_project)
    service = _build_service(test_project, fail_runner)

    result = service.sync()

    assert result.decision.action == "skip"
    assert result.decision.method is None
    assert "no documents" in result.decision.reason


def test_sync_force_coerces_update_method_to_full_rebuild(test_project) -> None:
    calls = []

    def runner(command, *, cwd, capture_output, text):
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


def test_sync_respects_explicit_method_override(test_project) -> None:
    calls = []

    def runner(command, *, cwd, capture_output, text):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="updated\n", stderr="")

    _write_settings(test_project)
    _write_source(test_project)
    service = _build_service(test_project, runner)

    result = service.sync(method="standard-update")

    assert result.decision.action == "index"
    assert result.decision.method == "standard-update"
    assert result.decision.reason == (
        "Explicit GraphRAG index method requested: standard-update."
    )
    assert calls[0][calls[0].index("--method") + 1] == "standard-update"


def test_sync_rebuilds_complete_output_when_metadata_is_missing(test_project) -> None:
    calls = []

    def runner(command, *, cwd, capture_output, text):
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
    def runner(command, *, cwd, capture_output, text):
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


def test_graph_runtime_digest_includes_prompt_files(test_project) -> None:
    _write_settings(test_project)
    before = graph_runtime_digest(test_project.paths.graph_dir / "graphrag")
    test_project.write_file("graph/graphrag/prompts/entity_extraction.txt", "Prompt A")

    after = graph_runtime_digest(test_project.paths.graph_dir / "graphrag")

    assert after != before


def test_graph_input_source_hashes_supports_object_payload_and_ignores_bad_records(
    test_project,
) -> None:
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
    assert count_source_hash_changes(None, {"src-1": "hash-1"}) is None
