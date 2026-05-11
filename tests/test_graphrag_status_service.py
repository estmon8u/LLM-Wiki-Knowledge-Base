from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.services.graphrag_command_service import GraphRAGCommandResult
from src.services.graphrag_status_service import GraphRAGStatus
from src.services.graphrag_status_service import GraphRAGStatusService


def test_status_reports_workspace_input_outputs_and_last_run(test_project) -> None:
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
    assert status.last_index_run_id == run.run_id
    assert status.last_index_method == "fast"
    assert status.last_index_success is True
    assert status.next_action.startswith("Run `kb graph ask")
    payload = status.to_dict(test_project.paths.root)
    assert payload["workspace_dir"] == "graph/graphrag"
    assert payload["input_path"] == "graph/graphrag/input/sources.json"


def test_status_counts_realistic_nested_graphrag_parquet_tables(
    test_project,
) -> None:
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
    assert payload["entity_count"] == 1


def test_status_reports_next_actions_for_missing_workspace(test_project) -> None:
    service = GraphRAGStatusService(test_project.paths)

    status = service.status()

    assert status.workspace_initialized is False
    assert status.input_document_count == 0
    assert status.output_present is False
    assert status.next_action == "Run `kb graph init`."


def test_status_counts_documents_from_dict_payload(test_project) -> None:
    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    test_project.write_file(
        "graph/graphrag/input/sources.json",
        json.dumps({"sources": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}),
    )
    service = GraphRAGStatusService(test_project.paths)

    status = service.status()

    assert status.input_document_count == 3
    assert status.next_action == (
        "Run `kb graph index --method fast --dry-run` before a full index."
    )


def test_status_handles_invalid_input_and_run_metadata(test_project) -> None:
    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    test_project.write_file("graph/graphrag/input/sources.json", "{not json")
    test_project.write_file("graph/runs/graph_index_runs.json", "{not json")
    service = GraphRAGStatusService(test_project.paths)

    status = service.status()

    assert status.input_exists is True
    assert status.input_document_count == 0
    assert status.last_index_run_id is None
    assert status.next_action == "Add and compile sources, then run `kb graph sync`."


def test_status_ignores_non_list_run_metadata(test_project) -> None:
    test_project.write_file("graph/runs/graph_index_runs.json", json.dumps({"run": 1}))
    service = GraphRAGStatusService(test_project.paths)

    assert service._load_runs() == []


def test_status_to_dict_handles_paths_outside_project(test_project, tmp_path) -> None:
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
        next_action="Run `kb graph init`.",
    )

    payload = status.to_dict(tmp_path)

    assert payload["workspace_dir"].endswith("outside/workspace")
