"""Tests for test graphrag status service.

This module belongs to `tests.test_graphrag_status_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.services.graphrag_command_service import GraphRAGCommandResult
from src.services.graphrag_status_service import GraphRAGStatus
from src.services.graphrag_status_service import GraphRAGStatusService


def test_status_reports_workspace_input_outputs_and_last_run(test_project) -> None:
    """Verifies that status reports workspace input outputs and last run.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    test_project.write_file(
        "graph/graphrag/input/sources.json",
        json.dumps([{"id": "a"}, {"id": "b"}]),
    )
    for table in (
        "documents",
        "text_units",
        "entities",
        "relationships",
        "communities",
        "community_reports",
    ):
        test_project.write_file(f"graph/graphrag/output/{table}.parquet", "")
    service = GraphRAGStatusService(test_project.paths)
    run = service.record_index_run(
        method="fast",
        dry_run=False,
        result=GraphRAGCommandResult(
            command=("python", "-m", "graphrag", "index"),
            cwd=test_project.paths.root,
            returncode=0,
            stdout="indexed",
            stderr="",
        ),
    )

    status = service.status()

    assert status.workspace_initialized is True
    assert status.input_exists is True
    assert status.input_document_count == 2
    assert status.output_present is True
    assert status.documents_present is True
    assert status.text_units_present is True
    assert status.entities_present is True
    assert status.relationships_present is True
    assert status.communities_present is True
    assert status.community_reports_present is True
    assert (
        status.active_output_dir == test_project.paths.graph_dir / "graphrag" / "output"
    )
    assert status.last_index_run_id == run.run_id
    assert status.last_index_method == "fast"
    assert status.last_index_success is True
    assert status.next_action.startswith("Run `kb ask")
    payload = status.to_dict(test_project.paths.root)
    assert payload["workspace_dir"] == "graph/graphrag"
    assert payload["input_path"] == "graph/graphrag/input/sources.json"
    assert payload["active_output_dir"] == "graph/graphrag/output"


def test_status_counts_realistic_nested_graphrag_parquet_tables(
    test_project,
) -> None:
    """Verifies that status counts realistic nested graphrag parquet tables.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    test_project.write_file(
        "graph/graphrag/input/sources.json",
        json.dumps([{"id": "doc-1", "text": "GraphRAG source text."}]),
    )
    output_dir = test_project.paths.graph_dir / "graphrag" / "output" / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_rows = {
        "create_final_documents.parquet": [{"id": "doc-1", "title": "RAG Paper"}],
        "create_final_text_units.parquet": [{"id": "tu-1", "text": "A chunk."}],
        "create_final_entities.parquet": [{"id": "ent-1", "title": "RAG"}],
        "create_final_relationships.parquet": [
            {"id": "rel-1", "source": "RAG", "target": "REALM"}
        ],
        "create_final_communities.parquet": [{"id": "0", "community": 0}],
        "create_final_community_reports.parquet": [
            {"id": "report-0", "community": 0, "summary": "Community report."}
        ],
    }
    for filename, rows in table_rows.items():
        pd.DataFrame(rows).to_parquet(output_dir / filename)
    test_project.write_file("wiki/graph/index.md", "# GraphRAG Index\n")

    status = GraphRAGStatusService(test_project.paths).status()

    assert status.output_present is True
    assert status.documents_present is True
    assert status.text_units_present is True
    assert status.entities_present is True
    assert status.relationships_present is True
    assert status.communities_present is True
    assert status.community_reports_present is True
    assert status.document_count == 1
    assert status.text_unit_count == 1
    assert status.entity_count == 1
    assert status.relationship_count == 1
    assert status.community_count == 1
    assert status.community_report_count == 1
    assert status.input_updated_at is not None
    assert status.output_updated_at is not None
    assert status.wiki_export_present is True
    assert status.wiki_export_updated_at is not None
    payload = status.to_dict(test_project.paths.root)
    assert payload["output_dir"] == "graph/graphrag/output"
    assert payload["active_output_dir"] == "graph/graphrag/output/artifacts"
    assert payload["entity_count"] == 1


def test_status_prefers_complete_output_over_newer_partial_output(
    test_project,
) -> None:
    """Verifies active output resolution ignores newer incomplete output folders."""
    import os
    import time

    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    test_project.write_file(
        "graph/graphrag/input/sources.json",
        json.dumps([{"id": "a"}]),
    )
    complete_dir = test_project.paths.graph_dir / "graphrag" / "output" / "complete"
    partial_dir = test_project.paths.graph_dir / "graphrag" / "output" / "partial"
    for table in (
        "documents",
        "text_units",
        "entities",
        "relationships",
        "communities",
        "community_reports",
    ):
        test_project.write_file(
            f"graph/graphrag/output/complete/{table}.parquet",
            "",
        )
    test_project.write_file("graph/graphrag/output/partial/entities.parquet", "")
    older = time.time() - 120
    newer = time.time()
    for path in complete_dir.glob("*.parquet"):
        os.utime(path, (older, older))
    for path in partial_dir.glob("*.parquet"):
        os.utime(path, (newer, newer))

    service = GraphRAGStatusService(test_project.paths)
    status = service.status()

    assert status.output_complete is True
    assert status.active_output_dir == complete_dir
    assert service.table_path("entities") == complete_dir / "entities.parquet"
    payload = status.to_dict(test_project.paths.root)
    assert payload["active_output_dir"] == "graph/graphrag/output/complete"


def test_status_prefers_recorded_successful_output_dir(test_project) -> None:
    """Regression: a later complete folder should not override the recorded run."""
    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    test_project.write_file(
        "graph/graphrag/input/sources.json",
        json.dumps([{"id": "a"}]),
    )
    recorded_dir = test_project.paths.graph_dir / "graphrag" / "output" / "recorded"
    later_dir = test_project.paths.graph_dir / "graphrag" / "output" / "later"
    for directory in (recorded_dir, later_dir):
        for table in (
            "documents",
            "text_units",
            "entities",
            "relationships",
            "communities",
            "community_reports",
        ):
            path = directory / f"{table}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
    service = GraphRAGStatusService(test_project.paths)
    service.record_index_run(
        method="fast",
        dry_run=False,
        result=GraphRAGCommandResult(
            command=("python", "-m", "graphrag", "index"),
            cwd=test_project.paths.root,
            returncode=0,
            stdout="indexed",
            stderr="",
        ),
    )
    runs = service._load_runs()
    runs[-1]["active_output_dir"] = "graph/graphrag/output/recorded"
    service.runs_file.write_text(json.dumps(runs), encoding="utf-8")

    assert service.status().active_output_dir == recorded_dir


def test_status_reports_next_actions_for_missing_workspace(test_project) -> None:
    """Verifies that status reports next actions for missing workspace.

    Args:
        test_project: Test project value used by the operation.
    """
    service = GraphRAGStatusService(test_project.paths)

    status = service.status()

    assert status.workspace_initialized is False
    assert status.input_document_count == 0
    assert status.output_present is False
    assert status.next_action == "Run `kb init`."


def test_status_counts_documents_from_dict_payload(test_project) -> None:
    """Verifies that status counts documents from dict payload.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    test_project.write_file(
        "graph/graphrag/input/sources.json",
        json.dumps({"sources": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}),
    )
    service = GraphRAGStatusService(test_project.paths)

    status = service.status()

    assert status.input_document_count == 3
    assert status.next_action == ("Run `kb update` to sync and build the graph index.")


def test_status_after_successful_dry_run_points_to_full_index(test_project) -> None:
    """Verifies that status after successful dry run points to full index.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    test_project.write_file(
        "graph/graphrag/input/sources.json",
        json.dumps([{"id": "a"}]),
    )
    service = GraphRAGStatusService(test_project.paths)
    service.record_index_run(
        method="fast",
        dry_run=True,
        result=GraphRAGCommandResult(
            command=("python", "-m", "graphrag", "index", "--dry-run"),
            cwd=test_project.paths.root,
            returncode=0,
            stdout="",
            stderr="",
        ),
    )

    status = service.status()

    assert status.output_present is False
    assert status.next_action == ("Run `kb update` to build the graph index.")


def test_status_after_failed_index_points_to_error_recovery(test_project) -> None:
    """Verifies that status after failed index points to error recovery.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    test_project.write_file(
        "graph/graphrag/input/sources.json",
        json.dumps([{"id": "a"}]),
    )
    service = GraphRAGStatusService(test_project.paths)
    service.record_index_run(
        method="fast",
        dry_run=False,
        result=GraphRAGCommandResult(
            command=("python", "-m", "graphrag", "index"),
            cwd=test_project.paths.root,
            returncode=2,
            stdout="",
            stderr="failed",
        ),
    )

    status = service.status()

    assert status.last_index_success is False
    assert status.next_action == (
        "Fix the last graph index error, then rerun `kb update`."
    )


def test_status_handles_invalid_input_and_run_metadata(test_project) -> None:
    """Verifies that status handles invalid input and run metadata.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    test_project.write_file("graph/graphrag/input/sources.json", "{not json")
    test_project.write_file("graph/runs/graph_index_runs.json", "{not json")
    service = GraphRAGStatusService(test_project.paths)

    status = service.status()

    assert status.input_exists is True
    assert status.input_document_count == 0
    assert status.last_index_run_id is None
    assert status.next_action == "Add and compile sources, then run `kb update`."


def test_status_ignores_non_list_run_metadata(test_project) -> None:
    """Verifies that status ignores non list run metadata.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file("graph/runs/graph_index_runs.json", json.dumps({"run": 1}))
    service = GraphRAGStatusService(test_project.paths)

    assert service._load_runs() == []


def test_status_to_dict_handles_paths_outside_project(test_project, tmp_path) -> None:
    """Verifies that status to dict handles paths outside project.

    Args:
        test_project: Test project value used by the operation.
        tmp_path: Tmp path value used by the operation.
    """
    status = GraphRAGStatus(
        workspace_dir=Path("D:/outside/workspace"),
        settings_path=Path("D:/outside/workspace/settings.yaml"),
        input_path=Path("D:/outside/workspace/input/sources.json"),
        output_dir=Path("D:/outside/workspace/output"),
        workspace_initialized=False,
        input_exists=False,
        input_document_count=0,
        output_present=False,
        documents_present=False,
        text_units_present=False,
        entities_present=False,
        relationships_present=False,
        communities_present=False,
        community_reports_present=False,
        last_index_run_id=None,
        last_index_run_at=None,
        last_index_method=None,
        last_index_success=None,
        next_action="Run `kb init`.",
    )

    payload = status.to_dict(tmp_path)

    assert payload["workspace_dir"].endswith("outside/workspace")
