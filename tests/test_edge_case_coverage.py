"""Focused edge-case coverage for defensive GraphRAG branch code."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import pytest

from src.services.config_service import DEFAULT_CONFIG, resolve_graph_config
from src.services.config_service import (
    _optional_str,
    _provider_catalog_api_key_env,
    concept_generation_enabled,
    concept_provider_backed_enabled,
)
from src.services.graphrag_status_service import (
    GraphRAGStatus,
    GraphRAGStatusService,
    iso_timestamp_after,
    _timestamp_iso,
)
from src.services.graphrag_wiki_export_service import _is_generated_graph_page
from src.services.normalization_service import (
    Xhtml2pdfRenderer,
    _extract_title,
    _is_probable_title,
)
from src.services.project_service import build_project_paths
from src.services.query_router_service import (
    QueryRouterService,
    _read_term_columns,
    _term_in_question,
)
from src.services.stopwords import _load_stopwords
from src.storage.compile_run_store import CompileRunStore, MAX_COMPILE_RUN_HISTORY


class _FakePdfStatus:
    def __init__(self, err: int = 0) -> None:
        self.err = err


def test_xhtml2pdf_renderer_success_and_failure_paths(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "source.html"
    source_path.write_text("<html><body>GraphRAG</body></html>", encoding="utf-8")
    calls: list[str] = []

    def create_pdf_success(html_text, *, dest, encoding):
        calls.append(f"{html_text}|{encoding}")
        dest.write(b"%PDF")
        return _FakePdfStatus()

    fake_pisa = types.SimpleNamespace(CreatePDF=create_pdf_success)
    monkeypatch.setitem(
        sys.modules,
        "xhtml2pdf",
        types.SimpleNamespace(pisa=fake_pisa),
    )

    assert Xhtml2pdfRenderer.available() is True
    assert Xhtml2pdfRenderer().render_file(source_path) == b"%PDF"
    assert calls == ["<html><body>GraphRAG</body></html>|utf-8"]

    def create_pdf_exception(html_text, *, dest, encoding):
        raise RuntimeError("boom")

    fake_pisa.CreatePDF = create_pdf_exception
    with pytest.raises(ValueError, match="could not render"):
        Xhtml2pdfRenderer().render_file(source_path)

    def create_pdf_status_error(html_text, *, dest, encoding):
        dest.write(b"%PDF")
        return _FakePdfStatus(err=1)

    fake_pisa.CreatePDF = create_pdf_status_error
    with pytest.raises(ValueError, match="reported errors"):
        Xhtml2pdfRenderer().render_file(source_path)

    def create_pdf_empty(html_text, *, dest, encoding):
        return _FakePdfStatus()

    fake_pisa.CreatePDF = create_pdf_empty
    with pytest.raises(ValueError, match="produced no PDF bytes"):
        Xhtml2pdfRenderer().render_file(source_path)


def test_generated_graph_page_detection_handles_bad_frontmatter(tmp_path) -> None:
    assert _is_generated_graph_page(tmp_path / "missing.md") is False

    plain = tmp_path / "plain.md"
    plain.write_text("# Manual note\n", encoding="utf-8")
    assert _is_generated_graph_page(plain) is False

    unterminated = tmp_path / "unterminated.md"
    unterminated.write_text("---\ngenerated: true\n", encoding="utf-8")
    assert _is_generated_graph_page(unterminated) is False

    invalid_yaml = tmp_path / "invalid.md"
    invalid_yaml.write_text("---\nvalue: [broken\n---\n", encoding="utf-8")
    assert _is_generated_graph_page(invalid_yaml) is False

    manual = tmp_path / "manual.md"
    manual.write_text("---\ngenerated: false\n---\n\n# Manual\n", encoding="utf-8")
    assert _is_generated_graph_page(manual) is False

    generated = tmp_path / "generated.md"
    generated.write_text("---\ngenerated: true\n---\n\n# Generated\n", encoding="utf-8")
    assert _is_generated_graph_page(generated) is True


def test_compile_run_store_recovers_corrupt_and_non_mapping_state(tmp_path) -> None:
    state_file = tmp_path / "compile-runs.json"
    state_file.write_text("{", encoding="utf-8")
    store = CompileRunStore(state_file)

    assert store.load_history() == []
    assert not state_file.exists()
    assert list(tmp_path.glob("compile-runs.json.*.corrupt"))

    state_file.write_text("[]", encoding="utf-8")
    assert store.load_history() == []

    payload = {
        "history": [
            {
                "run_id": str(index),
                "status": "completed",
                "started_at": "2026-05-11T00:00:00+00:00",
            }
            for index in range(MAX_COMPILE_RUN_HISTORY + 3)
        ]
    }
    store._write_payload(payload)
    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    assert persisted["version"] == 1
    assert len(persisted["history"]) == MAX_COMPILE_RUN_HISTORY
    assert persisted["history"][0]["run_id"] == "3"


def test_resolve_graph_config_reports_invalid_graph_shapes() -> None:
    with pytest.raises(ValueError, match="graph' must contain"):
        resolve_graph_config({"graph": []})

    graph = {**DEFAULT_CONFIG["graph"], "unknown": "value"}
    with pytest.raises(ValueError, match="unknown keys"):
        resolve_graph_config({"graph": graph})


def test_config_helper_edge_cases() -> None:
    assert concept_generation_enabled({"concepts": []}) is False
    assert concept_provider_backed_enabled({"concepts": []}) is False
    assert _optional_str(123) is None
    assert _provider_catalog_api_key_env({"providers": []}, "openai") is None
    assert (
        _provider_catalog_api_key_env({"providers": {"openai": []}}, "openai") is None
    )

    with pytest.raises(ValueError, match="not configured"):
        resolve_graph_config(
            {
                "graph": {
                    **DEFAULT_CONFIG["graph"],
                    "provider": "custom",
                    "embedding_provider": "custom",
                    "api_key_env": None,
                    "embedding_api_key_env": None,
                },
                "providers": {},
            }
        )


def test_query_router_helper_edge_cases(tmp_path) -> None:
    router = QueryRouterService()

    assert router.route("Where is the graph config?").method == "basic"
    assert router.route("Explain Dense Passage Retrieval.").method == "local"
    assert router._mentions_known_graph_term(" rag ") is False

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    table_path = output_dir / "create_final_entities.parquet"
    table_path.write_text("not parquet", encoding="utf-8")
    assert list(_read_term_columns(table_path)) == []
    assert _term_in_question("rag", "what is rag?") is True

    import pandas as pd

    pd.DataFrame([{"title": "Neural Graph Retrieval"}]).to_parquet(table_path)

    class _StatusService:
        def status(self):
            return types.SimpleNamespace(
                output_updated_at="2026-05-11T00:00:00+00:00",
                last_index_run_id=None,
            )

        def table_path(self, table_name: str):
            return table_path if table_name == "entities" else None

    graph_router = QueryRouterService(_StatusService())
    assert graph_router.route("What is Neural Graph Retrieval?").method == "local"
    assert graph_router.route("Explain Neural Graph Retrieval.").method == "local"


def test_stopword_loader_rejects_missing_and_empty_files(tmp_path) -> None:
    missing = tmp_path / "missing.txt"
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing or unreadable"):
        _load_stopwords(missing)
    with pytest.raises(RuntimeError, match="empty"):
        _load_stopwords(empty)


def test_stopword_loader_normalizes_terms(tmp_path) -> None:
    path = tmp_path / "stopwords.txt"
    path.write_text("The\n and \n\n", encoding="utf-8")

    assert _load_stopwords(path) == frozenset({"the", "and"})


def test_graphrag_status_edge_cases(tmp_path) -> None:
    paths = build_project_paths(tmp_path)
    service = GraphRAGStatusService(paths)

    assert service.table_path("unknown") is None
    assert service._input_document_count() == 0
    service.input_path.parent.mkdir(parents=True)
    service.input_path.write_text("{not-json", encoding="utf-8")
    assert service._input_document_count() == 0
    service.input_path.write_text('{"documents": [{}, {}]}', encoding="utf-8")
    assert service._input_document_count() == 2
    service.input_path.write_text('{"sources": [{}]}', encoding="utf-8")
    assert service._input_document_count() == 1
    service.input_path.write_text('{"other": []}', encoding="utf-8")
    assert service._input_document_count() == 0

    assert service._load_runs() == []
    service.runs_file.parent.mkdir(parents=True)
    service.runs_file.write_text("{not-json", encoding="utf-8")
    assert service._load_runs() == []
    service.runs_file.write_text('{"not": "a-list"}', encoding="utf-8")
    assert service._load_runs() == []
    service.runs_file.write_text('[{"run": 1}, "bad"]', encoding="utf-8")
    assert service._load_runs() == [{"run": 1}]

    assert (
        GraphRAGStatusService._next_action(
            workspace_initialized=True,
            input_exists=True,
            input_document_count=1,
            output_present=False,
            output_complete=False,
            last_run={"success": False},
        )
        == "Fix the last graph index error, then rerun `kb update`."
    )
    assert (
        GraphRAGStatusService._next_action(
            workspace_initialized=True,
            input_exists=True,
            input_document_count=1,
            output_present=False,
            output_complete=False,
            last_run={"dry_run": True},
        )
        == "Run `kb update` to build the graph index."
    )
    assert (
        GraphRAGStatusService._next_action(
            workspace_initialized=True,
            input_exists=True,
            input_document_count=1,
            output_present=True,
            output_complete=False,
            last_run=None,
        )
        == "Run `kb update` to rebuild incomplete graph index output."
    )
    assert (
        GraphRAGStatusService._next_action(
            workspace_initialized=True,
            input_exists=True,
            input_document_count=1,
            output_present=False,
            output_complete=False,
            last_run=None,
        )
        == "Run `kb update` to sync and build the graph index."
    )

    base = {
        "workspace_dir": paths.graph_dir / "graphrag",
        "settings_path": paths.graph_dir / "graphrag" / "settings.yaml",
        "input_path": service.input_path,
        "output_dir": service.output_dir,
        "workspace_initialized": True,
        "input_exists": True,
        "input_document_count": 1,
        "output_present": True,
        "documents_present": True,
        "text_units_present": True,
        "entities_present": True,
        "relationships_present": True,
        "communities_present": True,
        "community_reports_present": True,
        "last_index_run_id": None,
        "last_index_run_at": None,
        "last_index_method": None,
        "last_index_success": None,
        "next_action": "",
    }
    assert GraphRAGStatus(**{**base, "last_index_success": False}).state == "failed"
    assert GraphRAGStatus(**{**base, "relationships_present": False}).state == "partial"
    assert iso_timestamp_after(
        "2026-05-11T00:00:02Z",
        "2026-05-11T00:00:01+00:00",
    )
    assert not iso_timestamp_after("not-a-date", "2026-05-11T00:00:01+00:00")
    assert (
        GraphRAGStatus(
            **{
                **base,
                "input_updated_at": "2026-05-11T00:00:02+00:00",
                "output_updated_at": "2026-05-11T00:00:01+00:00",
            }
        ).state
        == "stale"
    )
    service.output_dir.mkdir(parents=True)
    assert service._active_output_dir() is None
    table = service.output_dir / "create_final_entities.parquet"
    table.write_text("not parquet", encoding="utf-8")
    assert service._table_path(service.output_dir, "entities") == table
    assert _timestamp_iso(None) is None


def test_normalization_title_rejection_edges() -> None:
    source_path = Path("x.md")
    assert _extract_title("---\nDelimiter Skipped", source_path) == "Delimiter Skipped"
    assert _extract_title("Good Title\n\n# Later Heading", source_path) == "Good Title"
    assert (
        _extract_title("# Abstract\n\nPost Heading Title", source_path)
        == "Post Heading Title"
    )
    assert _is_probable_title("author@example.com") is False
    assert _is_probable_title(" ".join(f"word{i}" for i in range(21))) is False
    assert _is_probable_title(" ".join(f"word{i}" for i in range(11)) + ".") is False
