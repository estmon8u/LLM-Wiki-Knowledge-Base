"""Tests for test normalization service.

This module belongs to `tests.test_normalization_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from markitdown import MarkItDownException

import graphwiki_kb.services.normalization_service as normalization_service_module
from graphwiki_kb.services.normalization_service import (
    HTML_FALLBACK_ROUTE,
    HTML_RENDERED_OCR_ROUTE,
    HTML_XHTML2PDF_OCR_ROUTE,
    MARKITDOWN_ROUTE,
    MISTRAL_DOCUMENT_ROUTE,
    MISTRAL_IMAGE_ROUTE,
    PDF_FALLBACK_ROUTE,
    DOCX_PPTX_FALLBACK_ROUTE,
    DoclingPdfConverter,
    MistralOcrConverter,
    NormalizationService,
    WkhtmltopdfRenderer,
    _ConvertedText,
    _extract_title,
    _plain_text,
    _validate_conversion_output,
    resolve_wkhtmltopdf_binary,
)


class FakeDoclingFallbackConverter:
    """Represents fake docling fallback converter behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    def __init__(self, markdown: str) -> None:
        """Initializes the instance.

        Args:
            markdown: Markdown content being processed.
        """
        self.markdown = markdown
        self.paths: list[Path] = []

    def convert_local(self, source_path: Path) -> str:
        """Convert local.

        Args:
            source_path: Source path being processed.

        Returns:
            str produced by the operation.
        """
        self.paths.append(source_path)
        return self.markdown


class FakeDoclingInnerConverter:
    """Represents fake docling inner converter behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    def __init__(self, markdown: str) -> None:
        """Initializes the instance.

        Args:
            markdown: Markdown content being processed.
        """
        self.markdown = markdown
        self.paths: list[Path] = []

    def convert(self, source_path: Path) -> SimpleNamespace:
        """Convert.

        Args:
            source_path: Source path being processed.

        Returns:
            SimpleNamespace produced by the operation.
        """
        self.paths.append(source_path)
        return SimpleNamespace(
            document=SimpleNamespace(export_to_markdown=lambda: self.markdown)
        )


class FakeMistralConverter:
    """Represents fake mistral converter behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    def __init__(
        self,
        *,
        document_text: str = "",
        image_text: str = "",
        document_error: Exception | None = None,
        image_error: Exception | None = None,
        page_count: int = 1,
    ) -> None:
        """Initializes the instance.

        Args:
            document_text: Document text value used by the operation.
            image_text: Image text value used by the operation.
            document_error: Document error value used by the operation.
            image_error: Image error value used by the operation.
            page_count: Page count value used by the operation.
        """
        self.document_text = document_text
        self.image_text = image_text
        self.document_error = document_error
        self.image_error = image_error
        self.page_count = page_count
        self.document_calls: list[dict[str, object]] = []
        self.image_calls: list[dict[str, object]] = []

    def convert_document_bytes(
        self,
        document_bytes: bytes,
        *,
        mime_type: str,
        document_name: str = "",
    ) -> _ConvertedText:
        """Convert document bytes.

        Args:
            document_bytes: Document bytes value used by the operation.
            mime_type: Mime type value used by the operation.
            document_name: Document name value used by the operation.

        Returns:
            _ConvertedText produced by the operation.
        """
        self.document_calls.append(
            {
                "bytes": document_bytes,
                "mime_type": mime_type,
                "document_name": document_name,
            }
        )
        if self.document_error is not None:
            raise self.document_error
        return _ConvertedText(
            normalized_text=self.document_text,
            page_count=self.page_count,
        )

    def convert_image_bytes(
        self,
        image_bytes: bytes,
        *,
        mime_type: str,
    ) -> _ConvertedText:
        """Convert image bytes.

        Args:
            image_bytes: Image bytes value used by the operation.
            mime_type: Mime type value used by the operation.

        Returns:
            _ConvertedText produced by the operation.
        """
        self.image_calls.append({"bytes": image_bytes, "mime_type": mime_type})
        if self.image_error is not None:
            raise self.image_error
        return _ConvertedText(
            normalized_text=self.image_text,
            page_count=self.page_count,
        )


class FakeMarkItDownConverter:
    """Represents fake mark it down converter behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    def __init__(self, markdown: str, *, title: str | None = None) -> None:
        """Initializes the instance.

        Args:
            markdown: Markdown content being processed.
            title: Title value used by the operation.
        """
        self.markdown = markdown
        self.title = title
        self.paths: list[Path] = []

    def convert_local(self, source_path: Path) -> SimpleNamespace:
        """Convert local.

        Args:
            source_path: Source path being processed.

        Returns:
            SimpleNamespace produced by the operation.
        """
        self.paths.append(source_path)
        return SimpleNamespace(markdown=self.markdown, title=self.title)


class FakeHtmlRenderer:
    """Represents fake html renderer behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    def __init__(
        self,
        *,
        pdf_bytes: bytes = b"%PDF-1.4\nfake\n",
        binary: str = "C:/wkhtmltopdf.exe",
    ) -> None:
        """Initializes the instance.

        Args:
            pdf_bytes: Pdf bytes value used by the operation.
            binary: Binary value used by the operation.
        """
        self.pdf_bytes = pdf_bytes
        self.binary = binary
        self.paths: list[Path] = []

    def resolve_binary(self) -> str:
        """Resolve binary.

        Returns:
            str produced by the operation.
        """
        return self.binary

    def render_file(self, source_path: Path) -> bytes:
        """Render file.

        Args:
            source_path: Source path being processed.

        Returns:
            bytes produced by the operation.
        """
        self.paths.append(source_path)
        return self.pdf_bytes


class FakeOcrClient:
    """Represents fake ocr client behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    def __init__(self) -> None:
        """Initializes the instance."""
        self.calls: list[dict[str, object]] = []
        self.ocr = self

    def process(self, **kwargs) -> SimpleNamespace:
        """Process.

        Args:
            kwargs: Kwargs value used by the operation.

        Returns:
            SimpleNamespace produced by the operation.
        """
        self.calls.append(kwargs)
        return SimpleNamespace(
            pages=[
                SimpleNamespace(index=1, markdown="Second page."),
                SimpleNamespace(index=0, markdown="# OCR Title\n\nFirst page."),
            ]
        )


def test_normalization_service_preserves_markdown_inputs(tmp_path: Path) -> None:
    """Verifies that normalization service preserves markdown inputs.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
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
    """Verifies that normalization service preserves plain text inputs.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "sample.txt"
    source_path.write_text("First line\nSecond line\n", encoding="utf-8")

    result = NormalizationService().normalize_path(source_path)

    assert result.title == "First line"
    assert result.normalized_suffix == ".txt"
    assert result.metadata["ingest_mode"] == "direct-canonical-text"
    assert result.metadata["normalization_route"] == "plain-text-passthrough"
    assert result.metadata["canonical_text_format"] == ".txt"
    assert result.normalized_text == "First line\nSecond line\n"


def test_normalization_service_routes_html_through_renderer_then_mistral(
    tmp_path: Path,
) -> None:
    """Verifies that normalization service routes html through renderer then mistral.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "sample.html"
    source_path.write_text(
        "<html><body><h1>HTML Research Note</h1></body></html>",
        encoding="utf-8",
    )
    mistral = FakeMistralConverter(
        document_text=(
            "# HTML Research Note\n\nTraceability matters for ingest quality.\n"
        )
    )
    renderer = FakeHtmlRenderer()

    result = NormalizationService(
        mistral_ocr_converter=mistral,
        html_renderer=renderer,
    ).normalize_path(source_path)

    assert renderer.paths == [source_path]
    assert mistral.document_calls[0]["mime_type"] == "application/pdf"
    assert result.title == "HTML Research Note"
    assert result.metadata["converter"] == "mistral-ocr"
    assert result.metadata["normalization_route"] == HTML_RENDERED_OCR_ROUTE
    assert result.metadata["wkhtmltopdf_path"] == "C:/wkhtmltopdf.exe"


def test_normalization_service_routes_pdf_to_mistral_by_default(tmp_path: Path) -> None:
    """Verifies that normalization service routes pdf to mistral by default.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"%PDF-1.4\nfake\n")
    mistral = FakeMistralConverter(
        document_text=(
            "# Dense Passage Retrieval\n\n"
            "This paper improves open-domain question answering with dense retrieval.\n"
        )
    )

    result = NormalizationService(mistral_ocr_converter=mistral).normalize_path(
        source_path
    )

    assert mistral.document_calls[0]["mime_type"] == "application/pdf"
    assert result.title == "Dense Passage Retrieval"
    assert result.metadata["converter"] == "mistral-ocr"
    assert result.metadata["normalization_route"] == MISTRAL_DOCUMENT_ROUTE


def test_normalization_service_falls_back_to_docling_for_pdf(tmp_path: Path) -> None:
    """Verifies that normalization service falls back to docling for pdf.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"%PDF-1.4\nfake\n")
    mistral = FakeMistralConverter(document_error=ValueError("ocr unavailable"))
    pdf_converter = FakeDoclingFallbackConverter(
        "# Dense Passage Retrieval\n\nFallback output with enough words to validate.\n"
    )

    result = NormalizationService(
        mistral_ocr_converter=mistral,
        pdf_converter=pdf_converter,
    ).normalize_path(source_path)

    assert pdf_converter.paths == [source_path]
    assert result.title == "Dense Passage Retrieval"
    assert result.metadata["fallback_used"] is True
    assert result.metadata["fallback_converter"] == "docling"
    assert result.metadata["normalization_route"] == PDF_FALLBACK_ROUTE


def test_normalization_service_falls_back_when_mistral_key_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that normalization service falls back when mistral key missing.

    Args:
        tmp_path: Tmp path value used by the operation.
        monkeypatch: Monkeypatch value used by the operation.
    """
    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"%PDF-1.4\nfake\n")
    pdf_converter = FakeDoclingFallbackConverter(
        "# Dense Passage Retrieval\n\nFallback output with enough words to validate.\n"
    )
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    result = NormalizationService(pdf_converter=pdf_converter).normalize_path(
        source_path
    )

    assert pdf_converter.paths == [source_path]
    assert result.metadata["fallback_used"] is True
    assert result.metadata["fallback_converter"] == "docling"
    assert "MISTRAL_API_KEY" in result.metadata["primary_error"]


def test_normalization_service_falls_back_to_markitdown_for_docx(
    tmp_path: Path,
) -> None:
    """Verifies that normalization service falls back to markitdown for docx.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "sample.docx"
    source_path.write_bytes(b"docx")
    mistral = FakeMistralConverter(document_error=ValueError("ocr unavailable"))
    markitdown = FakeMarkItDownConverter(
        "# DOCX Research Note\n\nA structured office document fallback path.\n",
        title="DOCX Research Note",
    )

    result = NormalizationService(
        mistral_ocr_converter=mistral,
        converter=markitdown,
    ).normalize_path(source_path)

    assert markitdown.paths == [source_path]
    assert result.title == "DOCX Research Note"
    assert result.metadata["fallback_used"] is True
    assert result.metadata["fallback_converter"] == "markitdown"
    assert result.metadata["normalization_route"] == DOCX_PPTX_FALLBACK_ROUTE


def test_normalization_service_routes_images_to_mistral(tmp_path: Path) -> None:
    """Verifies that normalization service routes images to mistral.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "figure.png"
    source_path.write_bytes(b"\x89PNG\r\n")
    mistral = FakeMistralConverter(
        image_text="# Screenshot Summary\n\nA diagram with readable labels and arrows.\n"
    )

    result = NormalizationService(mistral_ocr_converter=mistral).normalize_path(
        source_path
    )

    assert mistral.image_calls[0]["mime_type"] == "image/png"
    assert result.title == "Screenshot Summary"
    assert result.metadata["normalization_route"] == MISTRAL_IMAGE_ROUTE


def test_pdf_document_converter_uses_provided_converter(tmp_path: Path) -> None:
    """Verifies that pdf document converter uses provided converter.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"%PDF-1.4\nfake\n")
    inner_converter = FakeDoclingInnerConverter("# Wrapped PDF\n\nExported text.\n")

    result = DoclingPdfConverter(inner_converter).convert_local(source_path)

    assert inner_converter.paths == [source_path]
    assert result.normalized_text == "# Wrapped PDF\n\nExported text.\n"


def test_pdf_document_converter_uses_ascii_temp_copy_for_unicode_paths(
    tmp_path: Path,
) -> None:
    """Verifies that pdf document converter uses ascii temp copy for unicode paths.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "karpathy’s-note.pdf"
    source_path.write_bytes(b"%PDF-1.4\nfake\n")

    class UnicodeSensitiveDoclingConverter:
        """Represents unicode sensitive docling converter behavior and data.

        Attributes:
            See annotated class attributes for stored values.
        """

        def __init__(self) -> None:
            """Initializes the instance."""
            self.paths: list[Path] = []

        def convert(self, source_path: Path) -> SimpleNamespace:
            """Convert.

            Args:
                source_path: Source path being processed.

            Returns:
                SimpleNamespace produced by the operation.
            """
            self.paths.append(source_path)
            if not str(source_path).isascii():
                raise RuntimeError("unicode path not supported")
            return SimpleNamespace(
                document=SimpleNamespace(
                    export_to_markdown=lambda: "# Unicode Safe PDF\n\nConverted text.\n"
                )
            )

    inner_converter = UnicodeSensitiveDoclingConverter()

    result = DoclingPdfConverter(inner_converter).convert_local(source_path)

    assert result.normalized_text == "# Unicode Safe PDF\n\nConverted text.\n"
    assert len(inner_converter.paths) == 1
    assert inner_converter.paths[0] != source_path
    assert inner_converter.paths[0].suffix == ".pdf"
    assert str(inner_converter.paths[0]).isascii()


def test_pdf_document_converter_lazy_loads_docling_converter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that pdf document converter lazy loads docling converter.

    Args:
        tmp_path: Tmp path value used by the operation.
        monkeypatch: Monkeypatch value used by the operation.
    """
    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"%PDF-1.4\nfake\n")

    class FakeLazyDoclingConverter:
        """Represents fake lazy docling converter behavior and data.

        Attributes:
            See annotated class attributes for stored values.
        """

        paths: list[Path] = []

        def convert(self, source_path: Path) -> SimpleNamespace:
            """Convert.

            Args:
                source_path: Source path being processed.

            Returns:
                SimpleNamespace produced by the operation.
            """
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

    result = DoclingPdfConverter().convert_local(source_path)

    assert result.normalized_text == "# Lazy PDF\n\nLoaded on demand.\n"
    assert FakeLazyDoclingConverter.paths == [source_path]


def test_pdf_document_converter_wraps_docling_errors(tmp_path: Path) -> None:
    """Verifies that pdf document converter wraps docling errors.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"%PDF-1.4\nfake\n")

    class BrokenDoclingConverter:
        """Represents broken docling converter behavior and data.

        Attributes:
            See annotated class attributes for stored values.
        """

        def convert(self, source_path: Path) -> SimpleNamespace:
            """Convert.

            Args:
                source_path: Source path being processed.

            Returns:
                SimpleNamespace produced by the operation.
            """
            raise RuntimeError("broken pdf")

    with pytest.raises(
        ValueError,
        match="Docling could not convert sample.pdf: broken pdf",
    ):
        DoclingPdfConverter(BrokenDoclingConverter()).convert_local(source_path)


def test_pdf_document_converter_rejects_partial_success(tmp_path: Path) -> None:
    """Verifies that pdf document converter rejects partial success.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"%PDF-1.4\nfake\n")

    class PartialDoclingConverter:
        """Represents partial docling converter behavior and data.

        Attributes:
            See annotated class attributes for stored values.
        """

        def convert(self, source_path: Path) -> SimpleNamespace:
            """Convert.

            Args:
                source_path: Source path being processed.

            Returns:
                SimpleNamespace produced by the operation.
            """
            return SimpleNamespace(
                status=SimpleNamespace(value="partial_success"),
                errors=["page 2 failed"],
                document=SimpleNamespace(export_to_markdown=lambda: "# Partial\n"),
            )

    with pytest.raises(
        ValueError,
        match="Docling could not convert sample.pdf cleanly: status=partial_success; errors=page 2 failed",
    ):
        DoclingPdfConverter(PartialDoclingConverter()).convert_local(source_path)


def test_normalization_service_lazy_loads_pdf_fallback_converter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that normalization service lazy loads pdf fallback converter.

    Args:
        tmp_path: Tmp path value used by the operation.
        monkeypatch: Monkeypatch value used by the operation.
    """
    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"%PDF-1.4\nfake\n")

    class FakeServicePdfConverter:
        """Represents fake service pdf converter behavior and data.

        Attributes:
            See annotated class attributes for stored values.
        """

        instances = 0

        def __init__(self) -> None:
            """Initializes the instance."""
            type(self).instances += 1

        def convert_local(self, source_path: Path) -> str:
            """Convert local.

            Args:
                source_path: Source path being processed.

            Returns:
                str produced by the operation.
            """
            return (
                "# Lazy Service PDF\n\nCreated inside normalization service fallback.\n"
            )

    monkeypatch.setattr(
        normalization_service_module,
        "DoclingPdfConverter",
        FakeServicePdfConverter,
    )

    result = NormalizationService(
        mistral_ocr_converter=FakeMistralConverter(
            document_error=ValueError("ocr unavailable")
        )
    ).normalize_path(source_path)

    assert FakeServicePdfConverter.instances == 1
    assert result.title == "Lazy Service PDF"
    assert result.metadata["fallback_converter"] == "docling"


def test_normalization_service_wraps_markitdown_exception(tmp_path: Path) -> None:
    """Verifies that normalization service wraps markitdown exception.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "sample.csv"
    source_path.write_text("ignored", encoding="utf-8")

    class BrokenMarkItDownConverter:
        """Represents broken mark it down converter behavior and data.

        Attributes:
            See annotated class attributes for stored values.
        """

        def convert_local(self, source_path: Path) -> SimpleNamespace:
            """Convert local.

            Args:
                source_path: Source path being processed.

            Returns:
                SimpleNamespace produced by the operation.
            """
            raise MarkItDownException("broken csv")

    with pytest.raises(
        ValueError,
        match="MarkItDown could not convert sample.csv: broken csv",
    ):
        NormalizationService(converter=BrokenMarkItDownConverter()).normalize_path(
            source_path
        )


def test_normalization_service_wraps_unexpected_markitdown_exception(
    tmp_path: Path,
) -> None:
    """Verifies that normalization service wraps unexpected markitdown exception.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "sample.csv"
    source_path.write_text("ignored", encoding="utf-8")

    class BrokenMarkItDownConverter:
        """Represents broken mark it down converter behavior and data.

        Attributes:
            See annotated class attributes for stored values.
        """

        def convert_local(self, source_path: Path) -> SimpleNamespace:
            """Convert local.

            Args:
                source_path: Source path being processed.

            Returns:
                SimpleNamespace produced by the operation.
            """
            raise RuntimeError("unexpected csv failure")

    with pytest.raises(
        ValueError,
        match="MarkItDown could not convert sample.csv: unexpected csv failure",
    ):
        NormalizationService(converter=BrokenMarkItDownConverter()).normalize_path(
            source_path
        )


def test_normalization_service_uses_markitdown_for_csv_inputs(tmp_path: Path) -> None:
    """Verifies that normalization service uses markitdown for csv inputs.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "sample.csv"
    source_path.write_text("name,value\nalpha,1\n", encoding="utf-8")
    markitdown = FakeMarkItDownConverter(
        "# CSV Note\n\nStructured tabular content converted to markdown.\n",
        title="CSV Note",
    )

    result = NormalizationService(converter=markitdown).normalize_path(source_path)

    assert result.title == "CSV Note"
    assert result.metadata["converter"] == "markitdown"
    assert result.metadata["normalization_route"] == MARKITDOWN_ROUTE


def test_normalization_service_html_falls_back_to_markitdown(tmp_path: Path) -> None:
    """Verifies that normalization service html falls back to markitdown.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "sample.html"
    source_path.write_text("<p>ignored</p>", encoding="utf-8")
    mistral = FakeMistralConverter(document_error=ValueError("ocr unavailable"))
    markitdown = FakeMarkItDownConverter(
        "# HTML Note\n\nFallback markdown text with enough words.\n",
        title="HTML Note",
    )

    result = NormalizationService(
        mistral_ocr_converter=mistral,
        converter=markitdown,
        html_renderer=FakeHtmlRenderer(),
    ).normalize_path(source_path)

    assert result.title == "HTML Note"
    assert result.metadata["fallback_used"] is True
    assert result.metadata["normalization_route"] == HTML_FALLBACK_ROUTE


def test_normalization_service_html_without_markitdown_fallback_raises(
    tmp_path: Path,
) -> None:
    """Verifies that normalization service html without markitdown fallback raises.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "sample.html"
    source_path.write_text("<p>ignored</p>", encoding="utf-8")

    with pytest.raises(ValueError, match="HTML conversion failed for sample.html"):
        NormalizationService(
            config={"conversion": {"fallbacks": {"html": "none"}}},
            mistral_ocr_converter=FakeMistralConverter(
                document_error=ValueError("ocr unavailable")
            ),
            html_renderer=FakeHtmlRenderer(),
        ).normalize_path(source_path)


def test_normalization_service_pdf_without_supported_fallback_raises(
    tmp_path: Path,
) -> None:
    """Verifies that normalization service pdf without supported fallback raises.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"%PDF-1.4\nfake\n")

    with pytest.raises(ValueError, match="Mistral OCR could not convert sample.pdf"):
        NormalizationService(
            config={"conversion": {"fallbacks": {"pdf": "none"}}},
            mistral_ocr_converter=FakeMistralConverter(
                document_error=ValueError("ocr unavailable")
            ),
        ).normalize_path(source_path)


def test_normalization_service_fallback_failure_surfaces_both_errors(
    tmp_path: Path,
) -> None:
    """Verifies that normalization service fallback failure surfaces both errors.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "sample.html"
    source_path.write_text("<p>ignored</p>", encoding="utf-8")
    markitdown = FakeMarkItDownConverter("tiny", title="tiny")

    with pytest.raises(
        ValueError, match="All conversion routes failed for sample.html"
    ):
        NormalizationService(
            mistral_ocr_converter=FakeMistralConverter(
                document_error=ValueError("ocr unavailable")
            ),
            converter=markitdown,
            html_renderer=FakeHtmlRenderer(),
        ).normalize_path(source_path)


def test_normalization_service_rejects_unsupported_suffix(tmp_path: Path) -> None:
    """Verifies that normalization service rejects unsupported suffix.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "sample.bin"
    source_path.write_text("bits", encoding="utf-8")

    with pytest.raises(ValueError, match="Supported ingest inputs are canonical text"):
        NormalizationService().normalize_path(source_path)


def test_validate_conversion_output_rejects_truncated_multi_page_text() -> None:
    """Multi-page output that is below page_count * 60 chars triggers truncation."""
    # 10 pages but only ~180 chars of plain text — well below 10 * 60 = 600.
    # The implausibly-short check fires first with the same gate, so match either.
    truncated = "This is a real sentence with enough words. " * 4

    with pytest.raises(ValueError, match="implausibly short|appears truncated"):
        _validate_conversion_output(
            truncated,
            page_count=10,
            source_name="sample.pdf",
        )


def test_validate_conversion_output_allows_terminal_markdown_link_line() -> None:
    """Verifies that validate conversion output allows terminal markdown link line."""
    contents = (
        "This paragraph has enough words and ends with punctuation.\n\n" * 60
    ) + "[tbl-14.md](tbl-14.md)\n"

    _validate_conversion_output(
        contents,
        page_count=5,
        source_name="sample.pdf",
    )


def test_validate_conversion_output_rejects_empty_output() -> None:
    """Verifies that validate conversion output rejects empty output."""
    with pytest.raises(ValueError, match="conversion produced empty output"):
        _validate_conversion_output("", page_count=1, source_name="empty.pdf")


def test_validate_conversion_output_rejects_implausibly_short_multi_page_output() -> (
    None
):
    """Verifies that validate conversion output rejects implausibly short multi page output."""
    with pytest.raises(ValueError, match="implausibly short"):
        _validate_conversion_output(
            "Enough words to clear the tiny text gate but not enough for five pages.",
            page_count=5,
            source_name="short.pdf",
        )


def test_extract_title_prefers_heading_and_ignores_abstract() -> None:
    """Verifies that extract title prefers heading and ignores abstract."""
    contents = (
        "Figure 1: Overview\n\n"
        "# Abstract\n\n"
        "This abstract should not become the title.\n\n"
        "# Retrieval Augmented Generation\n\n"
        "Body text.\n"
    )

    assert (
        _extract_title(contents, Path("retrieval-augmented-generation.pdf"))
        == "Retrieval Augmented Generation"
    )


def test_extract_title_skips_frontmatter_and_affiliation_lines() -> None:
    """Verifies that extract title skips frontmatter and affiliation lines."""
    contents = (
        "---\n"
        "title: ignored\n"
        "---\n"
        "Department of Computer Science\n\n"
        "# Actual Title\n\n"
        "Body text.\n"
    )

    assert _extract_title(contents, Path("actual-title.md")) == "Actual Title"


def test_extract_title_empty_file_falls_back_to_filename() -> None:
    """Verifies that extract title empty file falls back to filename."""
    assert _extract_title("", Path("my-research.md")) == "My Research"


def test_extract_title_rejects_numbered_generic_heading() -> None:
    """Verifies that extract title rejects numbered generic heading."""
    contents = (
        "REALM: Retrieval-Augmented Language Model Pre-Training\n\n"
        "# Abstract\n\n"
        "Abstract text.\n\n"
        "# 1. Introduction\n\n"
        "Body text.\n"
    )

    assert (
        _extract_title(contents, Path("realm.pdf"))
        == "REALM: Retrieval-Augmented Language Model Pre-Training"
    )


def test_extract_title_long_first_line_truncates() -> None:
    """Verifies that extract title long first line truncates."""
    long_line = "A" * 200
    title = _extract_title(long_line, Path("source.md"))

    assert len(title) == 160
    assert title == "A" * 160


def test_extract_title_preserves_long_valid_title() -> None:
    """Verifies that extract title preserves long valid title."""
    title = (
        "Leveraging Passage Retrieval with Generative Models for Open Domain "
        "Question Answering"
    )

    assert _extract_title(title, Path("fid.pdf")) == title


def test_resolve_wkhtmltopdf_binary_uses_configured_path(tmp_path: Path) -> None:
    """Verifies that resolve wkhtmltopdf binary uses configured path.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    binary = tmp_path / "wkhtmltopdf.exe"
    binary.write_text("binary", encoding="utf-8")
    config = {
        "conversion": {
            "html": {
                "renderer": "wkhtmltopdf",
                "wkhtmltopdf_path": str(binary),
            }
        }
    }

    assert resolve_wkhtmltopdf_binary(config) == str(binary)


def test_resolve_wkhtmltopdf_binary_returns_none_for_invalid_html_config() -> None:
    """Verifies that resolve wkhtmltopdf binary returns none for invalid html config."""
    config = {"conversion": {"html": "oops"}}

    assert resolve_wkhtmltopdf_binary(config) is None


# ---------------------------------------------------------------------------
# markdown-it-py based _plain_text
# ---------------------------------------------------------------------------


def test_plain_text_extracts_text_from_markdown() -> None:
    """Verifies that plain text extracts text from markdown."""
    md = "# Heading\n\nSome **bold** text with [a link](http://example.com).\n"
    result = _plain_text(md)
    assert "Heading" in result
    assert "bold" in result
    assert "a link" in result
    assert "**" not in result
    assert "http" not in result


def test_plain_text_skips_fenced_code() -> None:
    """Verifies that plain text skips fenced code."""
    md = "Hello.\n\n```python\ncode = True\n```\n\nWorld.\n"
    result = _plain_text(md)
    assert "Hello." in result
    assert "World." in result
    assert "code = True" not in result


def test_plain_text_skips_images() -> None:
    """Verifies that plain text skips images."""
    md = "Text before.\n\n![alt text](image.png)\n\nText after.\n"
    result = _plain_text(md)
    assert "Text before." in result
    assert "Text after." in result


def test_plain_text_handles_empty_input() -> None:
    """Verifies that plain text handles empty input."""
    assert _plain_text("") == ""
    assert _plain_text("   ") == ""


def test_mistral_ocr_converter_uses_client_for_document_and_image() -> None:
    """Verifies that mistral ocr converter uses client for document and image."""
    client = FakeOcrClient()
    converter = MistralOcrConverter(client=client)

    document = converter.convert_document_bytes(
        b"document",
        mime_type="application/pdf",
        document_name="doc.pdf",
    )
    image = converter.convert_image_bytes(b"image", mime_type="image/png")

    assert document.normalized_text.startswith("# OCR Title")
    assert document.page_count == 2
    assert image.normalized_text.startswith("# OCR Title")
    assert client.calls[0]["document"]["type"] == "document_url"
    assert client.calls[1]["document"]["type"] == "image_url"


def test_mistral_ocr_converter_rejects_empty_payloads() -> None:
    """Verifies that mistral ocr converter rejects empty payloads."""
    converter = MistralOcrConverter(client=FakeOcrClient())

    with pytest.raises(ValueError, match="empty document"):
        converter.convert_document_bytes(b"", mime_type="application/pdf")
    with pytest.raises(ValueError, match="empty image"):
        converter.convert_image_bytes(b"", mime_type="image/png")


def test_mistral_ocr_converter_requires_api_key_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that mistral ocr converter requires api key env.

    Args:
        monkeypatch: Monkeypatch value used by the operation.
    """
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    converter = MistralOcrConverter()

    with pytest.raises(ValueError, match="Mistral OCR requires environment variable"):
        converter.convert_document_bytes(b"document", mime_type="application/pdf")


def test_mistral_ocr_converter_uses_public_sdk_client_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the converter should import the public Mistral SDK client path."""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    converter = MistralOcrConverter()

    client = converter._client_instance()

    assert client.__class__.__module__.startswith("mistralai.client")


def test_wkhtmltopdf_renderer_validates_binary_paths(tmp_path: Path) -> None:
    """Verifies that wkhtmltopdf renderer validates binary paths.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    missing_renderer = WkhtmltopdfRenderer(str(tmp_path / "missing.exe"))
    with pytest.raises(ValueError, match="configured path"):
        missing_renderer.resolve_binary()

    renderer = WkhtmltopdfRenderer()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            normalization_service_module.shutil, "which", lambda name: None
        )
        with pytest.raises(ValueError, match="wkhtmltopdf is required"):
            renderer.resolve_binary()


def test_wkhtmltopdf_renderer_wraps_render_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that wkhtmltopdf renderer wraps render failures.

    Args:
        tmp_path: Tmp path value used by the operation.
        monkeypatch: Monkeypatch value used by the operation.
    """
    source_path = tmp_path / "sample.html"
    source_path.write_text("<p>hello</p>", encoding="utf-8")
    binary = tmp_path / "wkhtmltopdf.exe"
    binary.write_text("binary", encoding="utf-8")
    renderer = WkhtmltopdfRenderer(str(binary))

    monkeypatch.setattr(
        normalization_service_module.pdfkit,
        "configuration",
        lambda wkhtmltopdf: object(),
    )
    monkeypatch.setattr(
        normalization_service_module.pdfkit,
        "from_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("render failed")),
    )

    with pytest.raises(ValueError, match="wkhtmltopdf could not render sample.html"):
        renderer.render_file(source_path)


def test_wkhtmltopdf_renderer_accepts_string_pdf_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that wkhtmltopdf renderer accepts string pdf output.

    Args:
        tmp_path: Tmp path value used by the operation.
        monkeypatch: Monkeypatch value used by the operation.
    """
    source_path = tmp_path / "sample.html"
    source_path.write_text("<p>hello</p>", encoding="utf-8")
    binary = tmp_path / "wkhtmltopdf.exe"
    binary.write_text("binary", encoding="utf-8")
    renderer = WkhtmltopdfRenderer(str(binary))

    monkeypatch.setattr(
        normalization_service_module.pdfkit,
        "configuration",
        lambda wkhtmltopdf: object(),
    )
    monkeypatch.setattr(
        normalization_service_module.pdfkit,
        "from_file",
        lambda *args, **kwargs: "%PDF-1.4\nfake\n",
    )

    pdf_bytes = renderer.render_file(source_path)

    assert pdf_bytes.startswith(b"%PDF-1.4")


def test_wkhtmltopdf_renderer_disables_local_file_access_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies wkhtmltopdf local-file access is trusted-source opt-in."""
    source_path = tmp_path / "sample.html"
    source_path.write_text("<p>hello</p>", encoding="utf-8")
    binary = tmp_path / "wkhtmltopdf.exe"
    binary.write_text("binary", encoding="utf-8")
    captured_options: dict[str, object] = {}

    monkeypatch.setattr(
        normalization_service_module.pdfkit,
        "configuration",
        lambda wkhtmltopdf: object(),
    )

    def fake_from_file(*args, **kwargs):
        captured_options.update(kwargs["options"])
        return b"%PDF-1.4\nfake\n"

    monkeypatch.setattr(
        normalization_service_module.pdfkit,
        "from_file",
        fake_from_file,
    )

    WkhtmltopdfRenderer(str(binary)).render_file(source_path)
    assert "enable-local-file-access" not in captured_options

    captured_options.clear()
    WkhtmltopdfRenderer(
        str(binary),
        allow_local_file_access=True,
    ).render_file(source_path)
    assert "enable-local-file-access" in captured_options


def test_wkhtmltopdf_renderer_rejects_empty_pdf_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that wkhtmltopdf renderer rejects empty pdf output.

    Args:
        tmp_path: Tmp path value used by the operation.
        monkeypatch: Monkeypatch value used by the operation.
    """
    source_path = tmp_path / "sample.html"
    source_path.write_text("<p>hello</p>", encoding="utf-8")
    binary = tmp_path / "wkhtmltopdf.exe"
    binary.write_text("binary", encoding="utf-8")
    renderer = WkhtmltopdfRenderer(str(binary))

    monkeypatch.setattr(
        normalization_service_module.pdfkit,
        "configuration",
        lambda wkhtmltopdf: object(),
    )
    monkeypatch.setattr(
        normalization_service_module.pdfkit,
        "from_file",
        lambda *args, **kwargs: b"",
    )

    with pytest.raises(ValueError, match="produced no PDF bytes"):
        renderer.render_file(source_path)


def test_normalization_service_adds_trailing_newline(tmp_path: Path) -> None:
    """Verifies that normalization service adds trailing newline.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "no-newline.md"
    source_path.write_text("# Title\n\nNo trailing newline", encoding="utf-8")

    result = NormalizationService().normalize_path(source_path)

    assert result.normalized_text.endswith("\n")
    assert not result.normalized_text.endswith("\n\n")


def test_normalization_service_newline_only_file(tmp_path: Path) -> None:
    """Verifies that normalization service newline only file.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    source_path = tmp_path / "newline-only.md"
    source_path.write_text("\n", encoding="utf-8")

    result = NormalizationService().normalize_path(source_path)

    assert result.normalized_text == "\n"


def test_normalization_service_html_falls_back_when_wkhtmltopdf_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HTML input with broken renderer should fall back to MarkItDown."""
    source_path = tmp_path / "report.html"
    source_path.write_text("<h1>Report</h1><p>Body text.</p>", encoding="utf-8")

    class BrokenRenderer:
        """Represents broken renderer behavior and data.

        Attributes:
            See annotated class attributes for stored values.
        """

        def resolve_binary(self) -> str:
            """Resolve binary.

            Returns:
                str produced by the operation.
            """
            return "/nonexistent/wkhtmltopdf"

        def render_file(self, source_path: Path) -> bytes:
            """Render file.

            Args:
                source_path: Source path being processed.

            Returns:
                bytes produced by the operation.
            """
            raise ValueError("wkhtmltopdf not found")

    # Ensure xhtml2pdf fallback is also skipped so we reach MarkItDown fallback.
    monkeypatch.setattr(
        "graphwiki_kb.services.normalization_service.Xhtml2pdfRenderer.available",
        staticmethod(lambda: False),
    )

    markitdown = FakeMarkItDownConverter(
        "# Report\n\nBody text from MarkItDown is good.\n",
        title="Report",
    )
    service = NormalizationService(
        converter=markitdown,
        html_renderer=BrokenRenderer(),
    )
    result = service.normalize_path(source_path)

    assert result.metadata["fallback_used"] is True
    assert result.metadata["fallback_converter"] == "markitdown"
    assert result.title == "Report"


def test_mistral_ocr_converter_retries_transient_errors() -> None:
    """Verify the OCR process call uses the retry decorator for transient errors."""
    call_count = 0

    class FlakeyOcrClient:
        """Represents flakey ocr client behavior and data.

        Attributes:
            See annotated class attributes for stored values.
        """

        def __init__(self) -> None:
            """Initializes the instance."""
            self.ocr = self

        def process(self, **kwargs):
            """Process.

            Args:
                kwargs: Kwargs value used by the operation.
            """
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return SimpleNamespace(
                pages=[SimpleNamespace(index=0, markdown="# Recovered\n\nBody.")]
            )

    converter = MistralOcrConverter(client=FlakeyOcrClient())
    result = converter.convert_document_bytes(
        b"%PDF-fake", mime_type="application/pdf", document_name="test.pdf"
    )
    assert call_count == 3
    assert "Recovered" in result.normalized_text


class FakeFailingHtmlRenderer:
    """Renderer whose render_file always fails (simulates missing wkhtmltopdf)."""

    def resolve_binary(self) -> str:
        """Resolve binary.

        Returns:
            str produced by the operation.
        """
        return "C:/missing-wkhtmltopdf.exe"

    def render_file(self, source_path: Path) -> bytes:
        """Render file.

        Args:
            source_path: Source path being processed.

        Returns:
            bytes produced by the operation.
        """
        raise ValueError(f"wkhtmltopdf could not render {source_path.name}")


def test_html_falls_back_to_xhtml2pdf_when_wkhtmltopdf_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that html falls back to xhtml2pdf when wkhtmltopdf fails.

    Args:
        tmp_path: Tmp path value used by the operation.
        monkeypatch: Monkeypatch value used by the operation.
    """
    source_path = tmp_path / "sample.html"
    source_path.write_text(
        "<html><body><h1>Via xhtml2pdf</h1></body></html>", encoding="utf-8"
    )
    mistral = FakeMistralConverter(
        document_text="# Via xhtml2pdf\n\nRendered through xhtml2pdf fallback.\n"
    )

    # Patch Xhtml2pdfRenderer.available to return True and render_file to succeed
    monkeypatch.setattr(
        "graphwiki_kb.services.normalization_service.Xhtml2pdfRenderer.available",
        staticmethod(lambda: True),
    )
    monkeypatch.setattr(
        "graphwiki_kb.services.normalization_service.Xhtml2pdfRenderer.render_file",
        lambda self, path: b"%PDF-1.4\nxhtml2pdf-fake\n",
    )

    result = NormalizationService(
        mistral_ocr_converter=mistral,
        html_renderer=FakeFailingHtmlRenderer(),
    ).normalize_path(source_path)

    assert result.title == "Via xhtml2pdf"
    assert result.metadata["html_renderer"] == "xhtml2pdf"
    assert result.metadata["normalization_route"] == HTML_XHTML2PDF_OCR_ROUTE
    assert "wkhtmltopdf_path" not in result.metadata
