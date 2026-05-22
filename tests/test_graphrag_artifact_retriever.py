"""Tests for the GraphRAG parquet-based retrieval used by the evaluator."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.graphrag_artifact_retriever import (
    GraphRAGArtifactRetriever,
    _record_result,
    _score,
)


def _write_parquet(path: Path, columns: dict[str, list]) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table(columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_record_result_text_unit_scores_body_match() -> None:
    record = {
        "id": "tu-1",
        "human_readable_id": 1,
        "text": ("REALM jointly trains a retriever and a masked language model."),
        "document_ids": ["doc-realm"],
    }
    result = _record_result("text_units", record, ["realm", "jointly"])
    assert result is not None
    assert result.kind == "text_units"
    assert "REALM jointly trains" in result.snippet
    assert result.source_ids == ("doc-realm",)
    assert result.score > 0


def test_record_result_returns_none_on_no_match() -> None:
    record = {"id": "tu-1", "text": "Completely unrelated body text."}
    result = _record_result("text_units", record, ["graphrag"])
    assert result is None


def test_score_is_zero_when_no_terms_match() -> None:
    assert (
        _score(
            ["graphrag"],
            title="entity X",
            searchable="entity x is a thing",
            kind="entities",
        )
        == 0.0
    )


def test_score_prefers_text_units_over_relationships_with_equal_hits() -> None:
    tu = _score(
        ["realm"], title="text_unit 5", searchable="realm body", kind="text_units"
    )
    rel = _score(
        ["realm"], title="A -> B", searchable="realm body", kind="relationships"
    )
    assert tu > rel


def _make_status_service(output_dir: Path) -> object:
    class _Status:
        def __init__(self, root: Path) -> None:
            self.root = root

        def table_path(self, name: str) -> Path | None:
            candidate = self.root / f"{name}.parquet"
            return candidate if candidate.exists() else None

    return _Status(output_dir)


def test_retriever_reads_text_units_and_community_reports(tmp_path: Path) -> None:
    output_dir = tmp_path / "graphrag_out"
    _write_parquet(
        output_dir / "text_units.parquet",
        {
            "id": ["tu-1", "tu-2"],
            "human_readable_id": [1, 2],
            "text": [
                "REALM trains a retriever and a masked language model jointly.",
                "Unrelated paragraph about distillation.",
            ],
            "document_ids": [["doc-realm"], ["doc-other"]],
        },
    )
    _write_parquet(
        output_dir / "community_reports.parquet",
        {
            "id": ["c-1"],
            "human_readable_id": [1],
            "title": ["Retrieval-Augmented Methods"],
            "summary": ["A community summary covering REALM, RAG, and DPR."],
            "full_content": [""],
            "community": [1],
            "rank": [1.0],
        },
    )
    retriever = GraphRAGArtifactRetriever(
        _make_status_service(output_dir), mode="text_units"
    )
    results = retriever.search("How does REALM differ from RAG?", limit=5)
    assert results
    titles = [r.title for r in results]
    assert any("text_unit" in t for t in titles)
    assert any("Retrieval-Augmented" in t for t in titles)


def test_artifact_mode_only_reads_entities_and_relationships(tmp_path: Path) -> None:
    output_dir = tmp_path / "graphrag_out"
    _write_parquet(
        output_dir / "text_units.parquet",
        {
            "id": ["tu-1"],
            "human_readable_id": [1],
            "text": ["REALM trains jointly with the retriever."],
            "document_ids": [["doc-realm"]],
        },
    )
    _write_parquet(
        output_dir / "entities.parquet",
        {
            "id": ["e-1"],
            "human_readable_id": [1],
            "title": ["REALM"],
            "name": ["REALM"],
            "description": ["Retrieval-Augmented Language Model"],
            "type": ["method"],
            "community": [None],
        },
    )
    retriever = GraphRAGArtifactRetriever(
        _make_status_service(output_dir), mode="artifact"
    )
    results = retriever.search("REALM", limit=5)
    assert results
    for r in results:
        # Should never include text_units in artifact mode.
        assert r.kind != "text_units"


def test_invalid_mode_raises() -> None:
    class _StubStatus:
        def table_path(self, name):
            return None

    with pytest.raises(ValueError):
        GraphRAGArtifactRetriever(_StubStatus(), mode="invalid")  # type: ignore[arg-type]


def test_search_handles_missing_output(tmp_path: Path) -> None:
    retriever = GraphRAGArtifactRetriever(
        _make_status_service(tmp_path / "no-such"), mode="text_units"
    )
    assert retriever.search("anything", limit=5) == []
