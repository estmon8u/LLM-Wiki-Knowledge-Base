from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Optional

from markitdown import __version__ as markitdown_version
from markitdown import MarkItDown, MarkItDownException


SUPPORTED_MARKDOWN_SUFFIXES = {".md", ".markdown"}
SUPPORTED_PLAIN_TEXT_SUFFIXES = {".txt"}
SUPPORTED_DOCLING_PDF_SUFFIXES = {".pdf"}
SUPPORTED_MARKITDOWN_SUFFIXES = {
    ".csv",
    ".docx",
    ".epub",
    ".htm",
    ".html",
    ".ipynb",
    ".pptx",
    ".xls",
    ".xlsx",
}

DIRECT_CANONICAL_TEXT_INGEST_MODE = "direct-canonical-text"
DOCLING_PDF_CONVERSION_INGEST_MODE = "docling-pdf-convert"
MARKITDOWN_CONVERSION_INGEST_MODE = "markitdown-convert"

MARKDOWN_PASSTHROUGH_ROUTE = "markdown-passthrough"
PLAIN_TEXT_PASSTHROUGH_ROUTE = "plain-text-passthrough"
DOCLING_PDF_ROUTE = "docling-pdf"
MARKITDOWN_ROUTE = "markitdown-born-digital"


@dataclass
class NormalizationResult:
    normalized_text: str
    normalized_suffix: str
    title: str
    metadata: dict[str, Any]


class PdfDocumentConverter:
    def __init__(self, converter: Optional[Any] = None) -> None:
        self._converter = converter

    def convert_local(self, source_path: Path) -> str:
        try:
            result = self._converter_instance().convert(source_path)
        except Exception as error:
            raise ValueError(
                f"Docling could not convert {source_path.name}: {error}"
            ) from error
        return result.document.export_to_markdown()

    def _converter_instance(self) -> Any:
        if self._converter is None:
            from docling.document_converter import DocumentConverter

            self._converter = DocumentConverter()
        return self._converter


class NormalizationService:
    def __init__(
        self,
        converter: Optional[MarkItDown] = None,
        pdf_converter: Optional[PdfDocumentConverter] = None,
    ) -> None:
        self._converter = converter
        self._pdf_converter = pdf_converter

    def normalize_path(self, source_path: Path) -> NormalizationResult:
        suffix = source_path.suffix.lower()
        if suffix in SUPPORTED_MARKDOWN_SUFFIXES:
            return self._normalize_direct_text(
                source_path,
                normalized_suffix=".md",
                route=MARKDOWN_PASSTHROUGH_ROUTE,
            )

        if suffix in SUPPORTED_PLAIN_TEXT_SUFFIXES:
            return self._normalize_direct_text(
                source_path,
                normalized_suffix=".txt",
                route=PLAIN_TEXT_PASSTHROUGH_ROUTE,
            )

        if suffix in SUPPORTED_DOCLING_PDF_SUFFIXES:
            return self._normalize_pdf(source_path)

        if suffix not in SUPPORTED_MARKITDOWN_SUFFIXES:
            raise ValueError(
                "Supported ingest inputs are canonical text (.md, .markdown, .txt), "
                "Docling-backed PDFs (.pdf), and MarkItDown-backed formats "
                "(.csv, .docx, .epub, .htm, .html, .ipynb, .pptx, .xls, .xlsx)."
            )

        return self._normalize_with_markitdown(source_path)

    def _normalize_direct_text(
        self,
        source_path: Path,
        *,
        normalized_suffix: str,
        route: str,
    ) -> NormalizationResult:
        contents = source_path.read_text(encoding="utf-8")
        normalized_text = _ensure_trailing_newline(contents)
        return NormalizationResult(
            normalized_text=normalized_text,
            normalized_suffix=normalized_suffix,
            title=_extract_title(normalized_text, source_path),
            metadata={
                "converter": "direct-copy",
                "ingest_mode": DIRECT_CANONICAL_TEXT_INGEST_MODE,
                "canonical_text_format": normalized_suffix,
                "normalization_route": route,
            },
        )

    def _normalize_pdf(self, source_path: Path) -> NormalizationResult:
        normalized_text = _ensure_trailing_newline(
            self._pdf_converter_instance().convert_local(source_path)
        )
        return NormalizationResult(
            normalized_text=normalized_text,
            normalized_suffix=".md",
            title=_extract_title(normalized_text, source_path),
            metadata={
                "converter": "docling",
                "converter_version": package_version("docling"),
                "ingest_mode": DOCLING_PDF_CONVERSION_INGEST_MODE,
                "canonical_text_format": ".md",
                "normalization_route": DOCLING_PDF_ROUTE,
            },
        )

    def _normalize_with_markitdown(self, source_path: Path) -> NormalizationResult:
        try:
            result = self._converter_instance().convert_local(source_path)
        except MarkItDownException as error:
            raise ValueError(
                f"MarkItDown could not convert {source_path.name}: {error}"
            ) from error
        except Exception as error:
            raise ValueError(
                f"MarkItDown could not convert {source_path.name}: {error}"
            ) from error

        normalized_text = _ensure_trailing_newline(result.markdown)
        title = (
            result.title.strip()
            if result.title
            else _extract_title(normalized_text, source_path)
        )
        return NormalizationResult(
            normalized_text=normalized_text,
            normalized_suffix=".md",
            title=title,
            metadata={
                "converter": "markitdown",
                "converter_version": markitdown_version,
                "ingest_mode": MARKITDOWN_CONVERSION_INGEST_MODE,
                "canonical_text_format": ".md",
                "normalization_route": MARKITDOWN_ROUTE,
            },
        )

    def _converter_instance(self) -> MarkItDown:
        if self._converter is None:
            self._converter = MarkItDown(enable_plugins=False)
        return self._converter

    def _pdf_converter_instance(self) -> PdfDocumentConverter:
        if self._pdf_converter is None:
            self._pdf_converter = PdfDocumentConverter()
        return self._pdf_converter


def _ensure_trailing_newline(contents: str) -> str:
    return contents.rstrip() + "\n"


def _extract_title(contents: str, source_path: Path) -> str:
    for line in contents.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("# ").strip()
        if stripped:
            return stripped[:80]
    return source_path.stem.replace("_", " ").replace("-", " ").title()
