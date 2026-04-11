from __future__ import annotations

from pathlib import Path

import pytest

from src.models.source_models import RawSourceRecord
from src.services.ingest_service import IngestService
from src.services.normalization_service import NormalizationService
from src.services.normalization_service import _extract_title
from src.services.project_service import utc_now_iso


class FakePdfConverter:
    def __init__(self, markdown: str) -> None:
        self.markdown = markdown

    def convert_local(self, source_path: Path) -> str:
        return self.markdown


def test_raw_source_record_round_trip_serialization() -> None:
    record = RawSourceRecord(
        source_id="source-1",
        slug="sample",
        title="Sample",
        origin="origin.md",
        source_type="file",
        raw_path="raw/sources/sample.md",
        normalized_path="raw/normalized/sample.md",
        content_hash="hash",
        ingested_at=utc_now_iso(),
        compiled_at=utc_now_iso(),
        compiled_from_hash="hash",
        metadata={"a": 1},
    )

    restored = RawSourceRecord.from_dict(record.to_dict())

    assert restored == record


def test_manifest_service_ensures_empty_manifest(test_project) -> None:
    manifest_service = test_project.services["manifest"]

    assert manifest_service.ensure_manifest() is False
    assert manifest_service.list_sources() == []
    assert manifest_service.find_by_hash("missing") is None


def test_manifest_service_reads_even_when_manifest_file_is_missing(
    uninitialized_project,
) -> None:
    manifest_service = uninitialized_project.services["manifest"]

    assert manifest_service.list_sources() == []
    assert uninitialized_project.paths.raw_manifest_file.exists()


def test_manifest_service_save_source_updates_existing_record(test_project) -> None:
    manifest_service = test_project.services["manifest"]
    record = RawSourceRecord(
        source_id="source-1",
        slug="sample",
        title="Sample",
        origin="sample.md",
        source_type="file",
        raw_path="raw/sources/sample.md",
        content_hash="hash-1",
        ingested_at=utc_now_iso(),
    )

    manifest_service.save_source(record)
    record.title = "Updated Sample"
    record.compiled_from_hash = "hash-1"
    manifest_service.save_source(record)

    sources = manifest_service.list_sources()
    assert len(sources) == 1
    assert sources[0].title == "Updated Sample"
    assert manifest_service.find_by_hash("hash-1").source_id == "source-1"


@pytest.mark.parametrize(
    "contents,expected",
    [
        ("# Header\n\nBody\n", "Header"),
        ("First line\n\nSecond\n", "First line"),
        ("\n\n", "Fallback Name"),
    ],
)
def test_extract_title_selects_best_available_title(
    contents: str, expected: str
) -> None:
    source_path = Path("fallback-name.md")

    assert _extract_title(contents, source_path) == expected


def test_ingest_service_copies_source_and_updates_manifest(test_project) -> None:
    source_path = test_project.write_file(
        "notes/example.md",
        "# Example Document\n\nUseful text about citations.\n",
    )
    ingest_service = test_project.services["ingest"]

    result = ingest_service.ingest_path(source_path)

    assert result.created is True
    assert result.source is not None
    assert result.source.slug == "example-document"
    assert result.source.normalized_path == "raw/normalized/example-document.md"
    assert result.source.metadata["ingest_mode"] == "direct-canonical-text"
    assert result.source.metadata["normalization_route"] == "markdown-passthrough"
    assert result.source.metadata["canonical_text_format"] == ".md"
    assert (test_project.root / result.source.raw_path).exists()
    assert (test_project.root / result.source.normalized_path).exists()
    assert (
        test_project.services["manifest"].list_sources()[0].title == "Example Document"
    )


def test_ingest_service_converts_html_and_stores_normalized_markdown(
    test_project,
) -> None:
    source_path = test_project.write_file(
        "notes/example.html",
        "<html><body><h1>HTML Research Note</h1><p>Useful converted text.</p></body></html>",
    )
    ingest_service = test_project.services["ingest"]

    result = ingest_service.ingest_path(source_path)

    assert result.created is True
    assert result.source is not None
    assert result.source.slug == "html-research-note"
    assert result.source.raw_path == "raw/sources/html-research-note.html"
    assert result.source.normalized_path == "raw/normalized/html-research-note.md"
    assert result.source.metadata["ingest_mode"] == "markitdown-convert"
    assert result.source.metadata["converter"] == "markitdown"
    assert result.source.metadata["normalization_route"] == "markitdown-born-digital"
    normalized_text = (test_project.root / result.source.normalized_path).read_text(
        encoding="utf-8"
    )
    assert "HTML Research Note" in normalized_text
    assert "Useful converted text." in normalized_text


def test_ingest_service_routes_pdf_sources_through_docling(test_project) -> None:
    source_path = test_project.write_file("notes/example.pdf", "not-a-real-pdf")
    normalization_service = NormalizationService(
        pdf_converter=FakePdfConverter(
            "# PDF Research Note\n\nUseful extracted text.\n"
        )
    )
    ingest_service = IngestService(
        test_project.paths,
        test_project.services["manifest"],
        normalization_service,
    )

    result = ingest_service.ingest_path(source_path)

    assert result.created is True
    assert result.source is not None
    assert result.source.slug == "pdf-research-note"
    assert result.source.raw_path == "raw/sources/pdf-research-note.pdf"
    assert result.source.normalized_path == "raw/normalized/pdf-research-note.md"
    assert result.source.metadata["ingest_mode"] == "docling-pdf-convert"
    assert result.source.metadata["converter"] == "docling"
    assert result.source.metadata["normalization_route"] == "docling-pdf"
    normalized_text = (test_project.root / result.source.normalized_path).read_text(
        encoding="utf-8"
    )
    assert "Useful extracted text." in normalized_text


def test_ingest_service_rejects_missing_and_unsupported_sources(test_project) -> None:
    ingest_service = test_project.services["ingest"]
    unsupported = test_project.write_file(
        "notes/data.bin",
        "binary-ish",
    )

    with pytest.raises(FileNotFoundError):
        ingest_service.ingest_path(test_project.root / "missing.md")

    with pytest.raises(
        ValueError,
        match="Supported ingest inputs are canonical text",
    ):
        ingest_service.ingest_path(unsupported)


def test_ingest_service_detects_duplicate_content_hash(test_project) -> None:
    source_path = test_project.write_file(
        "notes/duplicate.md",
        "# Duplicate\n\nSame body.\n",
    )
    ingest_service = test_project.services["ingest"]

    first = ingest_service.ingest_path(source_path)
    second = ingest_service.ingest_path(source_path)

    assert first.created is True
    assert second.created is False
    assert second.duplicate_of is not None
    assert second.duplicate_of.source_id == first.source.source_id
    assert len(test_project.services["manifest"].list_sources()) == 1


def test_ingest_service_uses_unique_slug_for_duplicate_titles(test_project) -> None:
    first = test_project.write_file("a.md", "# Same Title\n\nOne\n")
    second = test_project.write_file("b.md", "# Same Title\n\nTwo\n")
    ingest_service = test_project.services["ingest"]

    first_result = ingest_service.ingest_path(first)
    second_result = ingest_service.ingest_path(second)

    assert first_result.source.slug == "same-title"
    assert second_result.source.slug == "same-title-2"
