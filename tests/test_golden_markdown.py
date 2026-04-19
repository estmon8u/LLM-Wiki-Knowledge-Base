from __future__ import annotations

from pathlib import Path

import src.services.compile_service as compile_mod
import src.services.concept_service as concept_mod
import src.services.query_service as query_mod
from src.models.source_models import RawSourceRecord
from src.models.wiki_models import SearchResult
from src.services.concept_service import _ConceptDraft, _SourcePage
from src.services.query_service import QueryAnswer


_FIXED_TIMESTAMP = "2026-04-20T12:00:00+00:00"
_GOLDEN_DIR = Path(__file__).parent / "golden_markdown"


def _golden_text(name: str) -> str:
    return (_GOLDEN_DIR / name).read_text(encoding="utf-8")


def test_source_page_markdown_matches_golden(test_project) -> None:
    source = RawSourceRecord(
        source_id="src-123",
        slug="example",
        title="Example Source",
        origin="notes/example.md",
        source_type="file",
        raw_path="raw/sources/example.md",
        normalized_path="raw/normalized/example.md",
        content_hash="hash-example",
        ingested_at="2026-04-19T08:00:00+00:00",
    )

    rendered = test_project.services["compile"]._render_source_page(
        source,
        "# Example Source\n\nThis is the first paragraph.\n\nThis is the second paragraph.\n",
        _FIXED_TIMESTAMP,
    )

    assert rendered == _golden_text("source_page.md")


def test_index_markdown_matches_golden(monkeypatch, test_project) -> None:
    sources = [
        RawSourceRecord(
            source_id="b",
            slug="beta",
            title="Beta Page",
            origin="notes/b.md",
            source_type="file",
            raw_path="raw/sources/b.md",
            normalized_path="raw/normalized/b.md",
            content_hash="hb",
            ingested_at="2026-04-19T08:05:00+00:00",
            compiled_at="2026-04-20T11:00:00+00:00",
        ),
        RawSourceRecord(
            source_id="a",
            slug="alpha",
            title="Alpha Page",
            origin="notes/a.md",
            source_type="file",
            raw_path="raw/sources/a.md",
            normalized_path="raw/normalized/a.md",
            content_hash="ha",
            ingested_at="2026-04-19T08:00:00+00:00",
            compiled_at="2026-04-20T10:00:00+00:00",
        ),
    ]
    test_project.write_file(
        "wiki/concepts/analysis-note.md",
        "---\ntitle: Analysis Note\ntype: analysis\n---\n\n# Analysis Note\n",
    )
    monkeypatch.setattr(compile_mod, "utc_now_iso", lambda: _FIXED_TIMESTAMP)

    test_project.services["compile"]._write_index(sources)

    rendered = test_project.paths.wiki_index_markdown.read_text(encoding="utf-8")
    assert rendered == _golden_text("index.md")


def test_concept_page_markdown_matches_golden(monkeypatch, test_project) -> None:
    draft = _ConceptDraft(
        title="Dense Retrieval and Question Answering",
        slug="dense-retrieval-question-answering",
        summary=(
            "This concept groups dense retrieval sources that support question "
            "answering workflows."
        ),
        topic_terms=["dense-retrieval", "question-answering"],
        source_pages=[
            _SourcePage(
                file_path=test_project.root / "wiki/sources/alpha.md",
                relative_path="wiki/sources/alpha.md",
                slug="alpha",
                title="Alpha Page",
                summary="Alpha summary.",
                terms={"dense", "retrieval"},
            ),
            _SourcePage(
                file_path=test_project.root / "wiki/sources/beta.md",
                relative_path="wiki/sources/beta.md",
                slug="beta",
                title="Beta Page",
                summary="Beta summary.",
                terms={"question", "answering"},
            ),
        ],
    )
    monkeypatch.setattr(concept_mod, "utc_now_iso", lambda: _FIXED_TIMESTAMP)

    relative_path = test_project.services["concepts"]._write_concept_page(draft, set())

    assert relative_path == "wiki/concepts/dense-retrieval-question-answering.md"
    rendered = (test_project.root / relative_path).read_text(encoding="utf-8")
    assert rendered == _golden_text("concept_page.md")


def test_analysis_page_markdown_matches_golden(monkeypatch, test_project) -> None:
    answer = QueryAnswer(
        answer="Traceability is preserved through compiled source pages.",
        citations=[
            SearchResult(
                title="Alpha Page",
                path="wiki/sources/alpha.md",
                score=5,
                snippet="traceability",
                section="Summary",
                chunk_index=0,
            ),
            SearchResult(
                title="Beta Page",
                path="wiki/sources/beta.md",
                score=4,
                snippet="citations",
                section="Evidence",
                chunk_index=2,
            ),
        ],
        mode="provider:stub",
    )
    monkeypatch.setattr(query_mod, "utc_now_iso", lambda: _FIXED_TIMESTAMP)

    relative_path = test_project.services["query"].save_answer(
        "How does traceability work?", answer
    )

    assert relative_path == "wiki/concepts/how-does-traceability-work.md"
    rendered = (test_project.root / relative_path).read_text(encoding="utf-8")
    assert rendered == _golden_text("analysis_page.md")
