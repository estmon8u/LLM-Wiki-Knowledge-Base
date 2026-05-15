"""Tests for test graphrag wiki export service.

This module belongs to `tests.test_graphrag_wiki_export_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml

from graphwiki_kb.services.graphrag_command_service import GraphRAGCommandResult
from graphwiki_kb.services.graphrag_status_service import GraphRAGStatusService
from graphwiki_kb.services.graphrag_wiki_export_service import (
    GraphRAGWikiExportError,
    GraphRAGWikiExportService,
    MAX_ENTITY_RELATIONSHIP_ROWS,
    MAX_EXPORTED_RELATIONSHIP_PAGES,
    _clean_value,
    _field_list,
    _fenced_text,
    _findings_markdown,
    _first_number,
    _first_text,
    _relationship_table,
    _relationships_by_entity,
    _relationships_for_entity,
    _top_relationships,
    _unique_slug,
)


def _write_vector_store(test_project) -> None:
    test_project.write_file(
        "graph/graphrag/output/lancedb/vector-store.marker",
        "ready",
    )


def _write_graph_tables(test_project) -> None:
    """Handles write graph tables.

    Args:
        test_project: Test project value used by the operation.
    """
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
    _write_vector_store(test_project)
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
    """Handles build service.

    Args:
        test_project: Test project value used by the operation.

    Returns:
        GraphRAGWikiExportService produced by the operation.
    """
    return GraphRAGWikiExportService(
        test_project.paths,
        GraphRAGStatusService(test_project.paths),
        test_project.services["search"],
        refresh_index=test_project.services["compile"].refresh_index,
    )


def test_export_wiki_writes_graph_pages_and_preserves_legacy_concepts(
    test_project,
) -> None:
    """Verifies that export wiki writes graph pages and preserves legacy concepts.

    Args:
        test_project: Test project value used by the operation.
    """
    _write_graph_tables(test_project)
    legacy_concept = test_project.write_file(
        "wiki/concepts/legacy-concept.md",
        "---\ntype: concept\n---\n\n# Legacy Concept\n",
    )
    stale_graph_page = test_project.write_file(
        "wiki/graph/entities/stale.md",
        "---\ntype: graph_entity\ngenerated: true\n---\n\n# Stale\n",
    )
    manual_graph_note = test_project.write_file(
        "wiki/graph/entities/manual.md",
        "---\ntype: graph_note\n---\n\n# Manual\n",
    )
    kept_non_markdown = test_project.write_file(
        "wiki/graph/entities/keep.txt",
        "keep\n",
    )
    service = _build_service(test_project)

    result = service.export_wiki()

    assert stale_graph_page.exists() is False
    assert manual_graph_note.exists() is True
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
    assert entity_frontmatter["generated"] is True
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
    """Verifies that export wiki requires graph output.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    service = _build_service(test_project)

    with pytest.raises(GraphRAGWikiExportError, match="index output not found"):
        service.export_wiki()


def test_export_wiki_requires_initialized_workspace(test_project) -> None:
    """Verifies that export wiki requires initialized workspace.

    Args:
        test_project: Test project value used by the operation.
    """
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
    """Verifies that export wiki rejects incomplete nested outputs.

    Args:
        test_project: Test project value used by the operation.
    """
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

    status = GraphRAGStatusService(test_project.paths).status()

    assert status.output_present is True
    assert status.output_complete is False
    assert status.entity_count == 1
    assert "documents" in status.missing_tables
    assert "relationships" in status.missing_tables
    with pytest.raises(GraphRAGWikiExportError, match="output is incomplete"):
        service.export_wiki()


def test_export_wiki_handles_realistic_create_final_parquet_shapes(
    test_project,
) -> None:
    """Verifies that export wiki handles realistic create final parquet shapes.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    test_project.write_file(
        "graph/graphrag/input/sources.json",
        json.dumps([{"id": "doc-1", "text": "REALM retrieves evidence."}]),
    )
    output_dir = test_project.paths.graph_dir / "graphrag" / "output" / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "id": "doc-1",
                "human_readable_id": "1",
                "title": "REALM Paper",
                "raw_content": "REALM augments language model pre-training.",
            }
        ]
    ).to_parquet(output_dir / "create_final_documents.parquet")
    pd.DataFrame(
        [
            {
                "id": "tu-1",
                "human_readable_id": "1",
                "chunk": "REALM retrieves passages before prediction.",
                "document_ids": ["doc-1"],
                "entity_ids": ["entity-1"],
            }
        ]
    ).to_parquet(output_dir / "create_final_text_units.parquet")
    pd.DataFrame(
        [
            {
                "id": "entity-1",
                "human_readable_id": "1",
                "title": "REALM",
                "type": "method",
                "description": "REALM retrieves evidence for language modeling.",
                "text_unit_ids": ["tu-1"],
                "degree": 2,
            }
        ]
    ).to_parquet(output_dir / "create_final_entities.parquet")
    pd.DataFrame(
        [
            {
                "id": "rel-1",
                "human_readable_id": "1",
                "source": "REALM",
                "target": "Retrieval-Augmented Generation",
                "description": "uses retrieval to ground model behavior",
                "weight": 1.5,
                "text_unit_ids": ["tu-1"],
            }
        ]
    ).to_parquet(output_dir / "create_final_relationships.parquet")
    pd.DataFrame(
        [
            {
                "id": "community-0",
                "community": 0,
                "level": 1,
                "title": "Retrieval Methods",
                "entity_ids": ["entity-1"],
                "text_unit_ids": ["tu-1"],
            }
        ]
    ).to_parquet(output_dir / "create_final_communities.parquet")
    pd.DataFrame(
        [
            {
                "id": "report-0",
                "community": 0,
                "level": 1,
                "title": "Retrieval Methods",
                "summary": "Retrieval systems ground generation with evidence.",
                "full_content": "Detailed community report.",
                "findings": [
                    {
                        "summary": "REALM retrieves passages before prediction.",
                        "explanation": "The report links retrieval to model behavior.",
                    }
                ],
            }
        ]
    ).to_parquet(output_dir / "create_final_community_reports.parquet")
    _write_vector_store(test_project)
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
    service = _build_service(test_project)

    result = service.export_wiki()

    assert result.missing_tables == []
    assert result.exported_count == 6
    assert result.table_counts == {
        "documents": 1,
        "text_units": 1,
        "entities": 1,
        "relationships": 1,
        "communities": 1,
        "community_reports": 1,
    }
    assert "REALM augments language model pre-training." in (
        test_project.paths.wiki_dir / "graph" / "documents" / "realm-paper.md"
    ).read_text(encoding="utf-8")
    assert "REALM retrieves passages before prediction." in (
        test_project.paths.wiki_dir / "graph" / "text-units" / "text-unit-tu-1.md"
    ).read_text(encoding="utf-8")
    assert "uses retrieval to ground model behavior" in (
        test_project.paths.wiki_dir
        / "graph"
        / "relationships"
        / "realm-retrieval-augmented-generation.md"
    ).read_text(encoding="utf-8")
    community_page = (
        test_project.paths.wiki_dir
        / "graph"
        / "communities"
        / "community-0-retrieval-methods.md"
    )
    assert (
        "Retrieval systems ground generation with evidence."
        in community_page.read_text(encoding="utf-8")
    )
    assert "REALM retrieves passages before prediction." in community_page.read_text(
        encoding="utf-8"
    )


def test_export_wiki_caps_relationship_pages_and_entity_tables(test_project) -> None:
    """Verifies that export wiki caps relationship pages and entity tables.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    test_project.write_file(
        "graph/graphrag/input/sources.json",
        json.dumps([{"id": "doc-1", "text": "Hub has many relationships."}]),
    )
    output_dir = test_project.paths.graph_dir / "graphrag" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    relationship_count = MAX_EXPORTED_RELATIONSHIP_PAGES + 3
    pd.DataFrame(
        [
            {
                "id": "entity-1",
                "title": "Hub",
                "description": "Central graph entity.",
            }
        ]
    ).to_parquet(output_dir / "entities.parquet")
    pd.DataFrame(
        [
            {
                "id": f"rel-{index}",
                "source": "Hub",
                "target": f"Target {index}",
                "description": f"relationship {index}",
                "weight": index,
            }
            for index in range(relationship_count)
        ]
    ).to_parquet(output_dir / "relationships.parquet")
    pd.DataFrame(
        [{"id": "doc-1", "title": "Hub Document", "text": "Hub text."}]
    ).to_parquet(output_dir / "documents.parquet")
    pd.DataFrame([{"id": "tu-1", "text": "Hub text unit."}]).to_parquet(
        output_dir / "text_units.parquet"
    )
    pd.DataFrame(
        [{"id": "community-0", "community": 0, "title": "Hub Community"}]
    ).to_parquet(output_dir / "communities.parquet")
    pd.DataFrame(
        [
            {
                "id": "report-0",
                "community": 0,
                "title": "Hub Community",
                "summary": "Hub community summary.",
            }
        ]
    ).to_parquet(output_dir / "community_reports.parquet")
    _write_vector_store(test_project)
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
    service = _build_service(test_project)

    result = service.export_wiki()

    relationship_pages = list(
        (test_project.paths.wiki_dir / "graph" / "relationships").glob("*.md")
    )
    assert len(relationship_pages) == MAX_EXPORTED_RELATIONSHIP_PAGES
    assert result.table_counts["relationships"] == relationship_count
    index_text = (test_project.paths.wiki_dir / "graph" / "index.md").read_text(
        encoding="utf-8"
    )
    assert (
        f"`relationships/`: {MAX_EXPORTED_RELATIONSHIP_PAGES} of "
        f"{relationship_count} page(s)"
    ) in index_text
    exported_relationship_text = "\n".join(
        path.read_text(encoding="utf-8") for path in relationship_pages
    )
    assert f"relationship {relationship_count - 1}" in exported_relationship_text
    assert "relationship 0" not in exported_relationship_text
    entity_text = (
        test_project.paths.wiki_dir / "graph" / "entities" / "hub.md"
    ).read_text(encoding="utf-8")
    connected_rows = [
        line for line in entity_text.splitlines() if line.startswith("| Hub -> Target")
    ]
    assert len(connected_rows) == MAX_ENTITY_RELATIONSHIP_ROWS
    assert f"relationship {relationship_count - 1}" in entity_text
    assert "relationship 0" not in entity_text


def test_export_wiki_fences_raw_document_and_text_unit_markdown(test_project) -> None:
    """Verifies that export wiki fences raw document and text unit markdown.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file("graph/graphrag/settings.yaml", "input:\n  type: json\n")
    test_project.write_file(
        "graph/graphrag/input/sources.json",
        json.dumps([{"id": "doc-1", "text": "Raw markdown"}]),
    )
    output_dir = test_project.paths.graph_dir / "graphrag" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_text = "# Raw Heading\nSee [tbl-0.md](tbl-0.md)."
    pd.DataFrame(
        [
            {
                "id": "doc-1",
                "title": "Raw Markdown Document",
                "raw_content": raw_text,
                "raw_data": {"text": raw_text},
            }
        ]
    ).to_parquet(output_dir / "documents.parquet")
    pd.DataFrame(
        [{"id": "tu-1", "chunk": raw_text, "raw_data": {"text": raw_text}}]
    ).to_parquet(output_dir / "text_units.parquet")
    pd.DataFrame(
        [
            {
                "id": "entity-1",
                "title": "Raw Heading",
                "description": "Raw markdown entity.",
            }
        ]
    ).to_parquet(output_dir / "entities.parquet")
    pd.DataFrame([], columns=["id", "source", "target", "description"]).to_parquet(
        output_dir / "relationships.parquet"
    )
    pd.DataFrame(
        [{"id": "community-0", "community": 0, "title": "Raw Markdown"}]
    ).to_parquet(output_dir / "communities.parquet")
    pd.DataFrame(
        [
            {
                "id": "report-0",
                "community": 0,
                "title": "Raw Markdown",
                "summary": "Raw markdown community.",
            }
        ]
    ).to_parquet(output_dir / "community_reports.parquet")
    _write_vector_store(test_project)
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
    service = _build_service(test_project)

    service.export_wiki()

    document_text = (
        test_project.paths.wiki_dir / "graph" / "documents" / "raw-markdown-document.md"
    ).read_text(encoding="utf-8")
    text_unit_text = (
        test_project.paths.wiki_dir / "graph" / "text-units" / "text-unit-tu-1.md"
    ).read_text(encoding="utf-8")
    assert f"```text\n{raw_text}\n```" in document_text
    assert f"```text\n{raw_text}\n```" in text_unit_text
    assert "`raw_data`" not in document_text
    assert "`raw_data`" not in text_unit_text


def test_graph_wiki_export_helpers_handle_sparse_values() -> None:
    """Verifies that graph wiki export helpers handle sparse values."""

    class ArrayLike:
        """Represents array like behavior and data.

        Attributes:
            See annotated class attributes for stored values.
        """

        def tolist(self):
            """Tolist."""
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
    assert _fenced_text("before\n```text\ninside") == (
        "````text\nbefore\n```text\ninside\n````"
    )
    assert _top_relationships(
        [
            {"source": "A", "target": "low", "weight": 1},
            {"source": "A", "target": "high", "weight": 9},
            {"source": "A", "target": "mid", "weight": 3},
        ],
        2,
    ) == [
        {"source": "A", "target": "high", "weight": 9},
        {"source": "A", "target": "mid", "weight": 3},
    ]
    relationships_by_entity = _relationships_by_entity(
        [
            {"source": "Hub", "target": "A", "weight": 1},
            {"source": "Hub", "target": "B", "weight": 3},
        ]
    )
    assert _relationships_for_entity(relationships_by_entity, "Hub", limit=1) == [
        {"source": "Hub", "target": "B", "weight": 3}
    ]

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
