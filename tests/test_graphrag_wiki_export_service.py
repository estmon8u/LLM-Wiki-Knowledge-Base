from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from src.services.graphrag_command_service import GraphRAGCommandResult
from src.services.graphrag_status_service import GraphRAGStatusService
from src.services.graphrag_wiki_export_service import (
    GraphRAGWikiExportError,
    GraphRAGWikiExportService,
    _clean_value,
    _field_list,
    _findings_markdown,
    _first_number,
    _first_text,
    _relationship_table,
    _unique_slug,
)


def _write_graph_tables(test_project) -> None:
    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    test_project.write_file(
        "graph/graphrag/input/sources.json",
        json.dumps([{"id": "src-1", "text": "RAG text"}]),
    )
    output_dir = test_project.paths.graph_dir / "graphrag" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "id": "entity-1",
                "human_readable_id": "1",
                "title": "Retrieval-Augmented Generation",
                "type": "concept",
                "description": "A generation pattern grounded in retrieved text.",
                "degree": 8,
                "text_unit_ids": ["tu-1"],
            }
        ]
    ).to_parquet(output_dir / "entities.parquet")
    pd.DataFrame(
        [
            {
                "id": "rel-1",
                "human_readable_id": "1",
                "source": "Retrieval-Augmented Generation",
                "target": "Dense Passage Retrieval",
                "description": "uses dense retrieval",
                "combined_degree": 3,
            }
        ]
    ).to_parquet(output_dir / "relationships.parquet")
    pd.DataFrame(
        [
            {
                "id": "community-row-0",
                "community": 0,
                "title": "Retrieval methods",
                "level": 1,
                "entity_ids": ["entity-1"],
                "text_unit_ids": ["tu-1"],
            }
        ]
    ).to_parquet(output_dir / "communities.parquet")
    pd.DataFrame(
        [
            {
                "id": "report-0",
                "community": 0,
                "title": "Retrieval methods",
                "level": 1,
                "summary": "This community covers retrieval-grounded methods.",
                "full_content": "Full community report.",
            }
        ]
    ).to_parquet(output_dir / "community_reports.parquet")
    pd.DataFrame([{"id": "tu-1", "text": "Dense retrieval supports RAG."}]).to_parquet(
        output_dir / "text_units.parquet"
    )
    pd.DataFrame(
        [{"id": "doc-1", "title": "RAG Paper", "text": "Paper text."}]
    ).to_parquet(output_dir / "documents.parquet")
    GraphRAGStatusService(test_project.paths).record_index_run(
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


def _build_service(test_project) -> GraphRAGWikiExportService:
    return GraphRAGWikiExportService(
        test_project.paths,
        GraphRAGStatusService(test_project.paths),
        test_project.services["search"],
        refresh_index=test_project.services["compile"].refresh_index,
    )


def test_export_wiki_writes_graph_pages_and_preserves_legacy_concepts(
    test_project,
) -> None:
    _write_graph_tables(test_project)
    legacy_concept = test_project.write_file(
        "wiki/concepts/legacy-concept.md",
        "---\ntype: concept\n---\n\n# Legacy Concept\n",
    )
    stale_graph_page = test_project.write_file(
        "wiki/graph/entities/stale.md",
        "# Stale\n",
    )
    kept_non_markdown = test_project.write_file(
        "wiki/graph/entities/keep.txt",
        "keep\n",
    )
    service = _build_service(test_project)

    result = service.export_wiki()

    assert stale_graph_page.exists() is False
    assert kept_non_markdown.exists() is True
    assert legacy_concept.exists() is True
    assert result.table_counts["entities"] == 1
    assert result.table_counts["community_reports"] == 1
    assert result.missing_tables == []
    assert "wiki/graph/index.md" in result.exported_paths
    entity_path = (
        test_project.paths.wiki_dir
        / "graph"
        / "entities"
        / "retrieval-augmented-generation.md"
    )
    assert entity_path.exists()
    entity_text = entity_path.read_text(encoding="utf-8")
    entity_frontmatter = yaml.safe_load(entity_text.split("---", 2)[1])
    assert entity_frontmatter["type"] == "graph_entity"
    assert entity_frontmatter["entity_title"] == "Retrieval-Augmented Generation"
    assert "uses dense retrieval" in entity_text
    community_pages = list(
        (test_project.paths.wiki_dir / "graph" / "communities").glob("*.md")
    )
    assert len(community_pages) == 1
    assert "This community covers retrieval-grounded methods" in community_pages[
        0
    ].read_text(encoding="utf-8")
    assert (
        test_project.paths.wiki_dir / "graph" / "text-units" / "text-unit-tu-1.md"
    ).exists()
    assert (
        test_project.paths.wiki_dir / "graph" / "documents" / "rag-paper.md"
    ).exists()


def test_export_wiki_requires_graph_output(test_project) -> None:
    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    service = _build_service(test_project)

    with pytest.raises(GraphRAGWikiExportError, match="index output not found"):
        service.export_wiki()


def test_export_wiki_requires_initialized_workspace(test_project) -> None:
    output_dir = test_project.paths.graph_dir / "graphrag" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"id": "entity-1", "title": "RAG"}]).to_parquet(
        output_dir / "entities.parquet"
    )
    service = _build_service(test_project)

    with pytest.raises(GraphRAGWikiExportError, match="workspace is not initialized"):
        service.export_wiki()


def test_export_wiki_reports_missing_tables_and_finds_nested_outputs(
    test_project,
) -> None:
    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    output_dir = test_project.paths.graph_dir / "graphrag" / "output" / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "id": "entity-1",
                "human_readable_id": "1",
                "title": "Retrieval",
                "description": "A graph entity.",
            }
        ]
    ).to_parquet(output_dir / "create_final_entities.parquet")
    service = _build_service(test_project)

    result = service.export_wiki()

    assert result.table_counts["entities"] == 1
    assert "documents" in result.missing_tables
    assert "relationships" in result.missing_tables
    assert (
        test_project.paths.wiki_dir / "graph" / "entities" / "retrieval.md"
    ).exists()


def test_graph_wiki_export_helpers_handle_sparse_values() -> None:
    class ArrayLike:
        def tolist(self):
            return [1, float("nan"), {"value": float("nan")}]

    assert _clean_value(ArrayLike()) == [1, None, {"value": None}]
    assert _clean_value({"items": ("a", None)}) == {"items": ["a", None]}

    assert (
        _first_text(
            {"none": None, "empty": " ", "number": 7}, "none", "empty", "number"
        )
        == "7"
    )
    assert _first_text({"items": ["RAG"]}, "items", default="fallback") == "fallback"

    assert _first_number({"bad": "not-a-number", "value": "3.5"}, "bad", "value") == 3.5
    assert _first_number({"value": "2.0"}, "value") == 2

    metadata = _field_list(
        {"skip": "hidden", "empty": [], "payload": {"b": 2, "a": 1}},
        exclude={"skip"},
    )
    assert '- `payload`: {"a": 1, "b": 2}' in metadata
    assert _field_list({"none": None, "empty": ""}) == "No additional metadata."

    assert _relationship_table([]) == "No relationships listed."
    assert "A -> B" in _relationship_table([{"source": "A", "target": "B"}])

    findings = _findings_markdown(
        {
            "findings": [
                {"summary": "First finding."},
                {"explanation": "Second finding."},
                "Third finding.",
            ]
        }
    )
    assert "- First finding." in findings
    assert "- Second finding." in findings
    assert "- Third finding." in findings
    assert (
        _findings_markdown({"findings": []}) == "No key findings exported by GraphRAG."
    )

    used = {"entity", "entity-2"}
    assert _unique_slug("", used, prefix="entity") == "entity-3"
