from __future__ import annotations

import json

import pytest
import yaml

from src.models.source_models import RawSourceRecord
from src.services.graphrag_defaults import (
    DEFAULT_GRAPHRAG_EMBEDDING_MODEL,
    DEFAULT_GRAPHRAG_MODEL,
)
from src.services.graphrag_input_sync_service import (
    GRAPH_INPUT_METADATA_FIELDS,
    GraphRAGInputSyncError,
    GraphRAGInputSyncService,
)


def _write_graphrag_settings(test_project) -> None:
    test_project.write_file(
        "graph/graphrag/settings.yaml",
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
    )


def _source_record(
    *,
    source_id: str = "src-1",
    normalized_path: str | None = "raw/normalized/rag.md",
) -> RawSourceRecord:
    return RawSourceRecord(
        source_id=source_id,
        slug="rag",
        title="Retrieval-Augmented Generation",
        origin="C:/sources/rag.pdf",
        source_type="file",
        raw_path="raw/sources/rag.pdf",
        normalized_path=normalized_path,
        content_hash="normalized-sha256",
        origin_hash="raw-sha256",
        ingested_at="2026-05-11T00:00:00+00:00",
        metadata={
            "converter": "mistral-ocr",
            "normalization_route": "mistral-document",
            "source_extension": ".pdf",
        },
    )


def test_sync_writes_json_records_and_preserves_provenance(test_project) -> None:
    _write_graphrag_settings(test_project)
    test_project.write_file(
        "raw/normalized/rag.md",
        "# Retrieval-Augmented Generation\n\nRAG combines retrieval and generation.\n",
    )
    test_project.services["manifest"].save_source(_source_record())

    service = GraphRAGInputSyncService(
        test_project.paths,
        test_project.services["manifest"],
    )
    result = service.sync()

    assert result.source_count == 1
    assert result.output_path == test_project.root / "graph/graphrag/input/sources.json"
    records = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert records == [
        {
            "compiled_at": None,
            "compiled_from_hash": None,
            "converter": "mistral-ocr",
            "id": "src-1",
            "ingested_at": "2026-05-11T00:00:00+00:00",
            "metadata": {
                "converter": "mistral-ocr",
                "normalization_route": "mistral-document",
                "source_extension": ".pdf",
            },
            "normalization_route": "mistral-document",
            "normalized_path": "raw/normalized/rag.md",
            "origin": "C:/sources/rag.pdf",
            "origin_hash": "raw-sha256",
            "raw_path": "raw/sources/rag.pdf",
            "slug": "rag",
            "source_hash": "normalized-sha256",
            "source_id": "src-1",
            "source_type": "file",
            "text": "# Retrieval-Augmented Generation\n\n"
            "RAG combines retrieval and generation.\n",
            "title": "Retrieval-Augmented Generation",
        }
    ]


def test_sync_configures_json_input_and_metadata_prepending(test_project) -> None:
    _write_graphrag_settings(test_project)

    service = GraphRAGInputSyncService(
        test_project.paths,
        test_project.services["manifest"],
    )
    result = service.sync()

    settings = yaml.safe_load(result.settings_path.read_text(encoding="utf-8"))
    assert settings["input"] == {
        "type": "json",
        "encoding": "utf-8",
        "file_pattern": ".*\\.json$",
        "id_column": "id",
        "title_column": "title",
        "text_column": "text",
    }
    assert settings["input_storage"]["base_dir"] == "input"
    assert settings["chunking"]["prepend_metadata"] == list(GRAPH_INPUT_METADATA_FIELDS)
    assert json.loads(result.output_path.read_text(encoding="utf-8")) == []
    assert service.configure_settings() is False


def test_sync_reports_missing_normalized_artifact(test_project) -> None:
    _write_graphrag_settings(test_project)
    test_project.services["manifest"].save_source(_source_record())

    service = GraphRAGInputSyncService(
        test_project.paths,
        test_project.services["manifest"],
    )

    with pytest.raises(GraphRAGInputSyncError, match="Normalized artifact missing"):
        service.sync()


def test_sync_reports_missing_normalized_path(test_project) -> None:
    _write_graphrag_settings(test_project)
    test_project.services["manifest"].save_source(_source_record(normalized_path=None))

    service = GraphRAGInputSyncService(
        test_project.paths,
        test_project.services["manifest"],
    )

    with pytest.raises(GraphRAGInputSyncError, match="no normalized artifact path"):
        service.sync()


def test_sync_rejects_duplicate_source_ids(test_project) -> None:
    _write_graphrag_settings(test_project)
    source = _source_record().to_dict()
    payload = {
        "version": 1,
        "created_at": "2026-05-11T00:00:00+00:00",
        "updated_at": "2026-05-11T00:00:00+00:00",
        "sources": [source, source],
    }
    test_project.paths.raw_manifest_file.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    service = GraphRAGInputSyncService(
        test_project.paths,
        test_project.services["manifest"],
    )

    with pytest.raises(GraphRAGInputSyncError, match="Duplicate source_id"):
        service.sync()


def test_sync_reports_missing_graphrag_settings(test_project) -> None:
    service = GraphRAGInputSyncService(
        test_project.paths,
        test_project.services["manifest"],
    )

    with pytest.raises(
        GraphRAGInputSyncError, match="GraphRAG settings not found"
    ) as exc_info:
        service.sync()

    assert f"--model {DEFAULT_GRAPHRAG_MODEL}" in str(exc_info.value)
    assert f"--embedding {DEFAULT_GRAPHRAG_EMBEDDING_MODEL}" in str(exc_info.value)
    assert not service.input_file.exists()


def test_sync_reports_invalid_graphrag_settings(test_project) -> None:
    test_project.write_file("graph/graphrag/settings.yaml", "- not\n- a mapping\n")
    service = GraphRAGInputSyncService(
        test_project.paths,
        test_project.services["manifest"],
    )

    with pytest.raises(GraphRAGInputSyncError, match="must contain a YAML mapping"):
        service.sync()

    assert not service.input_file.exists()
