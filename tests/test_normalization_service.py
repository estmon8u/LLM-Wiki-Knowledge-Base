from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from markitdown import MarkItDownException

import src.services.normalization_service as normalization_service_module
from src.services.normalization_service import (
    NormalizationService,
    PdfDocumentConverter,
)


class FakePdfConverter:
    def __init__(self, markdown: str) -> None:
        self.markdown = markdown
        self.paths: list[Path] = []

    def convert_local(self, source_path: Path) -> str:
        self.paths.append(source_path)
        return self.markdown


class FakeDoclingInnerConverter:
    def __init__(self, markdown: str) -> None:
        self.markdown = markdown
        self.paths: list[Path] = []

    def convert(self, source_path: Path) -> SimpleNamespace:
        self.paths.append(source_path)
        return SimpleNamespace(
            document=SimpleNamespace(export_to_markdown=lambda: self.markdown)
        )


def test_normalization_service_preserves_markdown_inputs(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.md"
    source_path.write_text(
        "# Existing Note\n\nKeep this as markdown.\n", encoding="utf-8"
    )

    result = NormalizationService().normalize_path(source_path)

    assert result.title == "Existing Note"
    assert result.normalized_suffix == ".md"
    assert result.metadata["ingest_mode"] == "direct-canonical-text"
    assert result.metadata["normalization_route"] == "markdown-passthrough"
    assert result.metadata["canonical_text_format"] == ".md"
    assert "Keep this as markdown." in result.normalized_text


def test_normalization_service_preserves_plain_text_inputs(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.txt"
    source_path.write_text("First line\nSecond line\n", encoding="utf-8")

    result = NormalizationService().normalize_path(source_path)

    assert result.title == "First line"
    assert result.normalized_suffix == ".txt"
    assert result.metadata["ingest_mode"] == "direct-canonical-text"
    assert result.metadata["normalization_route"] == "plain-text-passthrough"
    assert result.metadata["canonical_text_format"] == ".txt"
    assert result.normalized_text == "First line\nSecond line\n"


def test_normalization_service_converts_html_to_markdown(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.html"
    source_path.write_text(
        "<html><body><h1>HTML Research Note</h1><p>Traceability matters.</p></body></html>",
        encoding="utf-8",
    )

    result = NormalizationService().normalize_path(source_path)

    assert result.title == "HTML Research Note"
    assert result.normalized_suffix == ".md"
    assert result.metadata["ingest_mode"] == "markitdown-convert"
    assert result.metadata["converter"] == "markitdown"
    assert result.metadata["normalization_route"] == "markitdown-born-digital"
    assert "HTML Research Note" in result.normalized_text
    assert "Traceability matters." in result.normalized_text


def test_normalization_service_routes_pdf_to_docling(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.pdf"
    source_path.write_text("not-a-real-pdf", encoding="utf-8")
    pdf_converter = FakePdfConverter("# PDF Note\n\nTable-aware text.\n")

    result = NormalizationService(pdf_converter=pdf_converter).normalize_path(
        source_path
    )

    assert pdf_converter.paths == [source_path]
    assert result.title == "PDF Note"
    assert result.normalized_suffix == ".md"
    assert result.metadata["ingest_mode"] == "docling-pdf-convert"
    assert result.metadata["converter"] == "docling"
    assert result.metadata["normalization_route"] == "docling-pdf"
    assert "Table-aware text." in result.normalized_text


def test_pdf_document_converter_uses_provided_converter(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.pdf"
    source_path.write_text("not-a-real-pdf", encoding="utf-8")
    inner_converter = FakeDoclingInnerConverter("# Wrapped PDF\n\nExported text.\n")

    result = PdfDocumentConverter(inner_converter).convert_local(source_path)

    assert inner_converter.paths == [source_path]
    assert result == "# Wrapped PDF\n\nExported text.\n"


def test_pdf_document_converter_lazy_loads_docling_converter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "sample.pdf"
    source_path.write_text("not-a-real-pdf", encoding="utf-8")

    class FakeLazyDoclingConverter:
        paths: list[Path] = []

        def convert(self, source_path: Path) -> SimpleNamespace:
            self.paths.append(source_path)
            return SimpleNamespace(
                document=SimpleNamespace(
                    export_to_markdown=lambda: "# Lazy PDF\n\nLoaded on demand.\n"
                )
            )

    import docling.document_converter as docling_document_converter

    monkeypatch.setattr(
        docling_document_converter,
        "DocumentConverter",
        FakeLazyDoclingConverter,
    )

    result = PdfDocumentConverter().convert_local(source_path)

    assert result == "# Lazy PDF\n\nLoaded on demand.\n"
    assert FakeLazyDoclingConverter.paths == [source_path]


def test_pdf_document_converter_wraps_docling_errors(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.pdf"
    source_path.write_text("not-a-real-pdf", encoding="utf-8")

    class BrokenDoclingConverter:
        def convert(self, source_path: Path) -> SimpleNamespace:
            raise RuntimeError("broken pdf")

    with pytest.raises(
        ValueError,
        match="Docling could not convert sample.pdf: broken pdf",
    ):
        PdfDocumentConverter(BrokenDoclingConverter()).convert_local(source_path)


def test_normalization_service_lazy_loads_pdf_converter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "sample.pdf"
    source_path.write_text("not-a-real-pdf", encoding="utf-8")

    class FakeServicePdfConverter:
        instances = 0

        def __init__(self) -> None:
            type(self).instances += 1

        def convert_local(self, source_path: Path) -> str:
            return "# Lazy Service PDF\n\nCreated inside normalization service.\n"

    monkeypatch.setattr(
        normalization_service_module,
        "PdfDocumentConverter",
        FakeServicePdfConverter,
    )

    result = NormalizationService().normalize_path(source_path)

    assert FakeServicePdfConverter.instances == 1
    assert result.title == "Lazy Service PDF"
    assert result.metadata["converter"] == "docling"


def test_normalization_service_wraps_markitdown_exception(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.html"
    source_path.write_text("<p>ignored</p>", encoding="utf-8")

    class BrokenMarkItDownConverter:
        def convert_local(self, source_path: Path) -> SimpleNamespace:
            raise MarkItDownException("broken html")

    with pytest.raises(
        ValueError,
        match="MarkItDown could not convert sample.html: broken html",
    ):
        NormalizationService(converter=BrokenMarkItDownConverter()).normalize_path(
            source_path
        )


def test_normalization_service_wraps_unexpected_markitdown_exception(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "sample.html"
    source_path.write_text("<p>ignored</p>", encoding="utf-8")

    class BrokenMarkItDownConverter:
        def convert_local(self, source_path: Path) -> SimpleNamespace:
            raise RuntimeError("unexpected html failure")

    with pytest.raises(
        ValueError,
        match="MarkItDown could not convert sample.html: unexpected html failure",
    ):
        NormalizationService(converter=BrokenMarkItDownConverter()).normalize_path(
            source_path
        )


def test_normalization_service_rejects_unsupported_suffix(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.bin"
    source_path.write_text("bits", encoding="utf-8")

    with pytest.raises(ValueError, match="Supported ingest inputs are canonical text"):
        NormalizationService().normalize_path(source_path)
