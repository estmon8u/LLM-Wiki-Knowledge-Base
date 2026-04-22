from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from markitdown import MarkItDownException

import src.services.normalization_service as normalization_service_module
from src.services.normalization_service import (
    HTML_FALLBACK_ROUTE,
    HTML_RENDERED_OCR_ROUTE,
    MARKITDOWN_ROUTE,
    MISTRAL_DOCUMENT_ROUTE,
    MISTRAL_IMAGE_ROUTE,
    PDF_FALLBACK_ROUTE,
    DOCX_PPTX_FALLBACK_ROUTE,
    DoclingPdfConverter,
    MistralOcrConverter,
    NormalizationService,
    PdfDocumentConverter,
    WkhtmltopdfRenderer,
    _ConvertedText,
    _extract_title,
    _validate_conversion_output,
    resolve_wkhtmltopdf_binary,
)


class FakeDoclingFallbackConverter:
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


class FakeMistralConverter:
    def __init__(
        self,
        *,
        document_text: str = "",
        image_text: str = "",
        document_error: Exception | None = None,
        image_error: Exception | None = None,
        page_count: int = 1,
    ) -> None:
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
        self.image_calls.append({"bytes": image_bytes, "mime_type": mime_type})
        if self.image_error is not None:
            raise self.image_error
        return _ConvertedText(
            normalized_text=self.image_text,
            page_count=self.page_count,
        )


class FakeMarkItDownConverter:
    def __init__(self, markdown: str, *, title: str | None = None) -> None:
        self.markdown = markdown
        self.title = title
        self.paths: list[Path] = []

    def convert_local(self, source_path: Path) -> SimpleNamespace:
        self.paths.append(source_path)
        return SimpleNamespace(markdown=self.markdown, title=self.title)


class FakeHtmlRenderer:
    def __init__(
        self,
        *,
        pdf_bytes: bytes = b"%PDF-1.4\nfake\n",
        binary: str = "C:/wkhtmltopdf.exe",
    ) -> None:
        self.pdf_bytes = pdf_bytes
        self.binary = binary
        self.paths: list[Path] = []

    def resolve_binary(self) -> str:
        return self.binary

    def render_file(self, source_path: Path) -> bytes:
        self.paths.append(source_path)
        return self.pdf_bytes


class FakeOcrClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.ocr = self

    def process(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            pages=[
                SimpleNamespace(index=1, markdown="Second page."),
                SimpleNamespace(index=0, markdown="# OCR Title\n\nFirst page."),
            ]
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


def test_normalization_service_routes_html_through_renderer_then_mistral(
    tmp_path: Path,
) -> None:
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
    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"%PDF-1.4\nfake\n")
    inner_converter = FakeDoclingInnerConverter("# Wrapped PDF\n\nExported text.\n")

    result = PdfDocumentConverter(inner_converter).convert_local(source_path)

    assert inner_converter.paths == [source_path]
    assert result.normalized_text == "# Wrapped PDF\n\nExported text.\n"


def test_pdf_document_converter_uses_ascii_temp_copy_for_unicode_paths(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "karpathy’s-note.pdf"
    source_path.write_bytes(b"%PDF-1.4\nfake\n")

    class UnicodeSensitiveDoclingConverter:
        def __init__(self) -> None:
            self.paths: list[Path] = []

        def convert(self, source_path: Path) -> SimpleNamespace:
            self.paths.append(source_path)
            if not str(source_path).isascii():
                raise RuntimeError("unicode path not supported")
            return SimpleNamespace(
                document=SimpleNamespace(
                    export_to_markdown=lambda: "# Unicode Safe PDF\n\nConverted text.\n"
                )
            )

    inner_converter = UnicodeSensitiveDoclingConverter()

    result = PdfDocumentConverter(inner_converter).convert_local(source_path)

    assert result.normalized_text == "# Unicode Safe PDF\n\nConverted text.\n"
    assert len(inner_converter.paths) == 1
    assert inner_converter.paths[0] != source_path
    assert inner_converter.paths[0].suffix == ".pdf"
    assert str(inner_converter.paths[0]).isascii()


def test_pdf_document_converter_lazy_loads_docling_converter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"%PDF-1.4\nfake\n")

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

    assert result.normalized_text == "# Lazy PDF\n\nLoaded on demand.\n"
    assert FakeLazyDoclingConverter.paths == [source_path]


def test_pdf_document_converter_wraps_docling_errors(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"%PDF-1.4\nfake\n")

    class BrokenDoclingConverter:
        def convert(self, source_path: Path) -> SimpleNamespace:
            raise RuntimeError("broken pdf")

    with pytest.raises(
        ValueError,
        match="Docling could not convert sample.pdf: broken pdf",
    ):
        PdfDocumentConverter(BrokenDoclingConverter()).convert_local(source_path)


def test_pdf_document_converter_rejects_partial_success(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"%PDF-1.4\nfake\n")

    class PartialDoclingConverter:
        def convert(self, source_path: Path) -> SimpleNamespace:
            return SimpleNamespace(
                status=SimpleNamespace(value="partial_success"),
                errors=["page 2 failed"],
                document=SimpleNamespace(export_to_markdown=lambda: "# Partial\n"),
            )

    with pytest.raises(
        ValueError,
        match="Docling could not convert sample.pdf cleanly: status=partial_success; errors=page 2 failed",
    ):
        PdfDocumentConverter(PartialDoclingConverter()).convert_local(source_path)


def test_normalization_service_lazy_loads_pdf_fallback_converter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "sample.pdf"
    source_path.write_bytes(b"%PDF-1.4\nfake\n")

    class FakeServicePdfConverter:
        instances = 0

        def __init__(self) -> None:
            type(self).instances += 1

        def convert_local(self, source_path: Path) -> str:
            return (
                "# Lazy Service PDF\n\nCreated inside normalization service fallback.\n"
            )

    monkeypatch.setattr(
        normalization_service_module,
        "PdfDocumentConverter",
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
    source_path = tmp_path / "sample.csv"
    source_path.write_text("ignored", encoding="utf-8")

    class BrokenMarkItDownConverter:
        def convert_local(self, source_path: Path) -> SimpleNamespace:
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
    source_path = tmp_path / "sample.csv"
    source_path.write_text("ignored", encoding="utf-8")

    class BrokenMarkItDownConverter:
        def convert_local(self, source_path: Path) -> SimpleNamespace:
            raise RuntimeError("unexpected csv failure")

    with pytest.raises(
        ValueError,
        match="MarkItDown could not convert sample.csv: unexpected csv failure",
    ):
        NormalizationService(converter=BrokenMarkItDownConverter()).normalize_path(
            source_path
        )


def test_normalization_service_uses_markitdown_for_csv_inputs(tmp_path: Path) -> None:
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
    source_path = tmp_path / "sample.bin"
    source_path.write_text("bits", encoding="utf-8")

    with pytest.raises(ValueError, match="Supported ingest inputs are canonical text"):
        NormalizationService().normalize_path(source_path)


def test_validate_conversion_output_rejects_truncated_multi_page_text() -> None:
    truncated = ("This is a long sentence without an ending " * 40).strip()

    with pytest.raises(ValueError, match="appears truncated"):
        _validate_conversion_output(
            truncated,
            page_count=5,
            source_name="sample.pdf",
        )


def test_validate_conversion_output_allows_terminal_markdown_link_line() -> None:
    contents = (
        "This paragraph has enough words and ends with punctuation.\n\n" * 60
    ) + "[tbl-14.md](tbl-14.md)\n"

    _validate_conversion_output(
        contents,
        page_count=5,
        source_name="sample.pdf",
    )


def test_validate_conversion_output_rejects_empty_output() -> None:
    with pytest.raises(ValueError, match="conversion produced empty output"):
        _validate_conversion_output("", page_count=1, source_name="empty.pdf")


def test_validate_conversion_output_rejects_implausibly_short_multi_page_output() -> None:
    with pytest.raises(ValueError, match="implausibly short"):
        _validate_conversion_output(
            "Enough words to clear the tiny text gate but not enough for five pages.",
            page_count=5,
            source_name="short.pdf",
        )


def test_extract_title_prefers_heading_and_ignores_abstract() -> None:
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
    assert _extract_title("", Path("my-research.md")) == "My Research"


def test_extract_title_rejects_numbered_generic_heading() -> None:
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
    long_line = "A" * 200
    title = _extract_title(long_line, Path("source.md"))

    assert len(title) == 160
    assert title == "A" * 160


def test_extract_title_preserves_long_valid_title() -> None:
    title = (
        "Leveraging Passage Retrieval with Generative Models for Open Domain "
        "Question Answering"
    )

    assert _extract_title(title, Path("fid.pdf")) == title


def test_resolve_wkhtmltopdf_binary_uses_configured_path(tmp_path: Path) -> None:
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
    config = {"conversion": {"html": "oops"}}

    assert resolve_wkhtmltopdf_binary(config) is None


def test_mistral_ocr_converter_uses_client_for_document_and_image() -> None:
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
    converter = MistralOcrConverter(client=FakeOcrClient())

    with pytest.raises(ValueError, match="empty document"):
        converter.convert_document_bytes(b"", mime_type="application/pdf")
    with pytest.raises(ValueError, match="empty image"):
        converter.convert_image_bytes(b"", mime_type="image/png")


def test_mistral_ocr_converter_requires_api_key_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    converter = MistralOcrConverter()

    with pytest.raises(ValueError, match="Mistral OCR requires environment variable"):
        converter.convert_document_bytes(b"document", mime_type="application/pdf")


def test_wkhtmltopdf_renderer_validates_binary_paths(tmp_path: Path) -> None:
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


def test_wkhtmltopdf_renderer_rejects_empty_pdf_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    source_path = tmp_path / "no-newline.md"
    source_path.write_text("# Title\n\nNo trailing newline", encoding="utf-8")

    result = NormalizationService().normalize_path(source_path)

    assert result.normalized_text.endswith("\n")
    assert not result.normalized_text.endswith("\n\n")


def test_normalization_service_newline_only_file(tmp_path: Path) -> None:
    source_path = tmp_path / "newline-only.md"
    source_path.write_text("\n", encoding="utf-8")

    result = NormalizationService().normalize_path(source_path)

    assert result.normalized_text == "\n"


def test_normalization_service_html_falls_back_when_wkhtmltopdf_missing(
    tmp_path: Path,
) -> None:
    """HTML input with broken renderer should fall back to MarkItDown."""
    source_path = tmp_path / "report.html"
    source_path.write_text("<h1>Report</h1><p>Body text.</p>", encoding="utf-8")

    class BrokenRenderer:
        def resolve_binary(self) -> str:
            return "/nonexistent/wkhtmltopdf"

        def render_file(self, source_path: Path) -> bytes:
            raise ValueError("wkhtmltopdf not found")

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
        def __init__(self) -> None:
            self.ocr = self

        def process(self, **kwargs):
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
