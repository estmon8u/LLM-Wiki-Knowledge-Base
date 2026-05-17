"""Document normalization and converter routing for source ingestion."""

from __future__ import annotations

import base64
import logging
import os
import re
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from importlib.metadata import version as package_version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from markitdown import MarkItDown

try:
    import pdfkit
except ImportError:  # pragma: no cover - optional dependency absent
    pdfkit = None

from graphwiki_kb.services.config_service import DEFAULT_CONFIG
from graphwiki_kb.services.markdown_document import (
    normalize_newlines as markdown_normalize_newlines,
)
from graphwiki_kb.services.markdown_document import (
    paragraphs as markdown_paragraphs,
)
from graphwiki_kb.services.markdown_document import (
    parse_document,
)
from graphwiki_kb.services.markdown_document import (
    plain_text as markdown_plain_text,
)

SUPPORTED_MARKDOWN_SUFFIXES = {".md", ".markdown"}
SUPPORTED_PLAIN_TEXT_SUFFIXES = {".txt"}
SUPPORTED_MISTRAL_DOCUMENT_SUFFIXES = {".pdf", ".docx", ".pptx"}
SUPPORTED_MISTRAL_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".avif"}
SUPPORTED_HTML_SUFFIXES = {".htm", ".html"}
SUPPORTED_MARKITDOWN_SUFFIXES = {
    ".csv",
    ".epub",
    ".ipynb",
    ".xls",
    ".xlsx",
}
SUPPORTED_SOURCE_SUFFIXES = (
    SUPPORTED_MARKDOWN_SUFFIXES
    | SUPPORTED_PLAIN_TEXT_SUFFIXES
    | SUPPORTED_MISTRAL_DOCUMENT_SUFFIXES
    | SUPPORTED_MISTRAL_IMAGE_SUFFIXES
    | SUPPORTED_HTML_SUFFIXES
    | SUPPORTED_MARKITDOWN_SUFFIXES
)

DIRECT_CANONICAL_TEXT_INGEST_MODE = "direct-canonical-text"
MISTRAL_OCR_CONVERSION_INGEST_MODE = "mistral-ocr-convert"
DOCLING_PDF_CONVERSION_INGEST_MODE = "docling-pdf-convert"
MARKITDOWN_CONVERSION_INGEST_MODE = "markitdown-convert"

MARKDOWN_PASSTHROUGH_ROUTE = "markdown-passthrough"
PLAIN_TEXT_PASSTHROUGH_ROUTE = "plain-text-passthrough"
MISTRAL_DOCUMENT_ROUTE = "mistral-ocr-document"
MISTRAL_IMAGE_ROUTE = "mistral-ocr-image"
HTML_RENDERED_OCR_ROUTE = "wkhtmltopdf-mistral-ocr"
HTML_XHTML2PDF_OCR_ROUTE = "xhtml2pdf-mistral-ocr"
DOCX_PPTX_FALLBACK_ROUTE = "mistral-ocr-document->markitdown-fallback"
HTML_FALLBACK_ROUTE = "wkhtmltopdf-mistral-ocr->markitdown-fallback"
PDF_FALLBACK_ROUTE = "mistral-ocr-document->docling-fallback"
PDF_MARKITDOWN_FALLBACK_ROUTE = "mistral-ocr-document->markitdown-fallback"
MARKITDOWN_ROUTE = "markitdown-born-digital"

MISTRAL_DOCUMENT_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
MISTRAL_IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".avif": "image/avif",
}


def _markitdown_exception_types() -> tuple[type[Exception], ...]:
    try:
        from markitdown import MarkItDownException
    except ImportError:
        return ()
    return (MarkItDownException,)


def _markitdown_version() -> str:
    try:
        from markitdown import __version__ as version
    except ImportError:
        return "unavailable"
    return str(version)


MAX_INLINE_OCR_PAYLOAD_BYTES = 100 * 1024 * 1024
_GENERIC_TITLE_HEADINGS = {
    "abstract",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "background",
    "conclusion",
    "contents",
    "discussion",
    "experiments",
    "introduction",
    "method",
    "methods",
    "related work",
    "references",
    "table of contents",
}
_AFFILIATION_HINTS = (
    "university",
    "department",
    "institute",
    "laboratory",
    "school",
    "college",
)
_CAPTION_PATTERN = re.compile(r"^(figure|fig\.?|table)\s+\d+\b", re.IGNORECASE)
_NUMBERED_SECTION_PREFIX_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?|[ivxlcdm]+\.?)\s+",
    re.IGNORECASE,
)

logger = logging.getLogger(__name__)


@dataclass
class NormalizationResult:
    """Final normalized content, title, suffix, and converter metadata."""

    normalized_text: str
    normalized_suffix: str
    title: str
    metadata: dict[str, Any]


@dataclass
class _ConvertedText:
    """Normalized text plus converter metadata before it becomes an ingest result."""

    normalized_text: str
    title: str | None = None
    page_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DoclingPdfConverter:
    """Adapter for Docling's local PDF conversion API."""

    def __init__(self, converter: Any | None = None) -> None:
        self._converter = converter

    def convert_local(self, source_path: Path) -> _ConvertedText:
        """Convert one local PDF through Docling."""
        conversion_path, temp_dir = self._prepare_conversion_path(source_path)
        try:
            result = self._converter_instance().convert(conversion_path)
        except Exception as error:
            raise ValueError(
                f"Docling could not convert {source_path.name}: {error}"
            ) from error
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

        status = getattr(result, "status", None)
        status_value = getattr(status, "value", str(status or "")).lower()
        errors = [str(error) for error in (getattr(result, "errors", None) or [])]
        if (status is not None and status_value != "success") or errors:
            detail = f"status={status_value or 'unknown'}"
            if errors:
                detail = f"{detail}; errors={'; '.join(errors)}"
            raise ValueError(
                f"Docling could not convert {source_path.name} cleanly: {detail}"
            )

        markdown = result.document.export_to_markdown()
        page_count = len(getattr(result, "pages", None) or [])
        return _ConvertedText(normalized_text=markdown, page_count=page_count)

    def _prepare_conversion_path(
        self, source_path: Path
    ) -> tuple[Path, TemporaryDirectory[str] | None]:
        if str(source_path).isascii():
            return source_path, None

        temp_dir = TemporaryDirectory(prefix="kb-docling-")
        temp_path = Path(temp_dir.name) / f"input{source_path.suffix.lower()}"
        shutil.copyfile(source_path, temp_path)
        return temp_path, temp_dir

    def _converter_instance(self) -> Any:
        if self._converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pipeline_options = PdfPipelineOptions(
                do_ocr=False,
                force_backend_text=True,
                ocr_batch_size=1,
                layout_batch_size=1,
                table_batch_size=1,
            )
            try:
                self._converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(
                            pipeline_options=pipeline_options
                        )
                    }
                )
            except TypeError:
                self._converter = DocumentConverter()
        return self._converter


class MistralOcrConverter:
    """Thin wrapper around the Mistral OCR SDK for document and image bytes."""

    def __init__(
        self,
        *,
        model: str = "mistral-ocr-latest",
        api_key_env: str = "MISTRAL_API_KEY",
        table_format: str = "markdown",
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.table_format = table_format
        self._client = client

    def convert_document_bytes(
        self,
        document_bytes: bytes,
        *,
        mime_type: str,
        document_name: str = "",
    ) -> _ConvertedText:
        """Convert document bytes through Mistral OCR."""
        if not document_bytes:
            raise ValueError("Mistral OCR cannot process an empty document.")
        document_url = _data_uri(mime_type, document_bytes)
        response = self._ocr_process(
            document={
                "type": "document_url",
                "document_url": document_url,
                "document_name": document_name or None,
            },
        )
        pages = getattr(response, "pages", []) or []
        logger.info(
            "Mistral OCR processed %s: %d page(s)",
            document_name or "document",
            len(pages),
        )
        return _ConvertedText(
            normalized_text=_join_ocr_pages(pages),
            page_count=len(pages),
        )

    def convert_image_bytes(
        self,
        image_bytes: bytes,
        *,
        mime_type: str,
    ) -> _ConvertedText:
        """Convert image bytes through Mistral OCR."""
        if not image_bytes:
            raise ValueError("Mistral OCR cannot process an empty image.")
        image_url = _data_uri(mime_type, image_bytes)
        response = self._ocr_process(
            document={
                "type": "image_url",
                "image_url": image_url,
            },
        )
        pages = getattr(response, "pages", []) or []
        return _ConvertedText(
            normalized_text=_join_ocr_pages(pages),
            page_count=len(pages),
        )

    def _ocr_process(self, *, document: dict[str, Any]) -> Any:
        """Call ``client.ocr.process`` with transient-error retry."""
        from graphwiki_kb.providers.retry import provider_retry

        @provider_retry()
        def _call() -> Any:
            return self._client_instance().ocr.process(
                model=self.model,
                document=document,
                table_format=self.table_format,
            )

        return _call()

    def _client_instance(self) -> Any:
        if self._client is None:
            api_key = os.environ.get(self.api_key_env, "").strip()
            if not api_key:
                raise ValueError(
                    f"Mistral OCR requires environment variable {self.api_key_env}."
                )
            from mistralai.client import Mistral

            self._client = Mistral(api_key=api_key)
        return self._client


class WkhtmltopdfRenderer:
    """Renders local HTML files to PDF through `wkhtmltopdf`."""

    def __init__(
        self,
        wkhtmltopdf_path: str | None = None,
        *,
        allow_local_file_access: bool = False,
    ) -> None:
        self._wkhtmltopdf_path = wkhtmltopdf_path
        self.allow_local_file_access = allow_local_file_access

    def resolve_binary(self) -> str:
        """Resolve binary.

        Returns:
            str produced by the operation.
        """
        configured = (self._wkhtmltopdf_path or "").strip()
        if configured:
            candidate = Path(configured)
            if candidate.exists():
                return str(candidate)
            raise ValueError(
                f"wkhtmltopdf binary not found at configured path: {configured}"
            )

        discovered = shutil.which("wkhtmltopdf")
        if discovered:
            return discovered
        raise ValueError(
            "wkhtmltopdf is required to convert .html and .htm inputs before OCR."
        )

    def render_file(self, source_path: Path) -> bytes:
        """Render file.

        Args:
            source_path: Source path being processed.

        Returns:
            bytes produced by the operation.
        """
        binary = self.resolve_binary()
        if pdfkit is None:
            raise ValueError(
                "pdfkit is required to render HTML with wkhtmltopdf. "
                "Install the html or all extra."
            )
        try:
            configuration = pdfkit.configuration(wkhtmltopdf=binary)
            options = {
                "encoding": "UTF-8",
                "quiet": None,
                "no-stop-slow-scripts": None,
                "disable-javascript": None,
                "no-images": None,
            }
            if self.allow_local_file_access:
                options["enable-local-file-access"] = None
            pdf_bytes = pdfkit.from_file(
                str(source_path),
                False,
                options=options,
                configuration=configuration,
                verbose=False,
            )
        except Exception as error:
            raise ValueError(
                f"wkhtmltopdf could not render {source_path.name}: {error}"
            ) from error

        if isinstance(pdf_bytes, str):
            pdf_bytes = pdf_bytes.encode("utf-8")
        if not isinstance(pdf_bytes, (bytes, bytearray)) or not pdf_bytes:
            raise ValueError(
                f"wkhtmltopdf produced no PDF bytes for {source_path.name}."
            )
        return bytes(pdf_bytes)


class Xhtml2pdfRenderer:
    """HTML→PDF renderer using xhtml2pdf (pure-Python, no external binary)."""

    @staticmethod
    def available() -> bool:
        """Available.

        Returns:
            bool produced by the operation.
        """
        try:
            import xhtml2pdf  # noqa: F401

            return True
        except ImportError:
            return False

    def render_file(self, source_path: Path) -> bytes:
        """Render file.

        Args:
            source_path: Source path being processed.

        Returns:
            bytes produced by the operation.
        """
        try:
            from xhtml2pdf import pisa
        except ImportError as exc:
            raise ValueError(
                "xhtml2pdf is not installed. Install it with: pip install xhtml2pdf"
            ) from exc

        import io
        import logging
        import os
        import sys

        html_text = source_path.read_text(encoding="utf-8", errors="replace")
        pdf_buffer = io.BytesIO()

        # xhtml2pdf and reportlab emit noisy CSS/font warnings to
        # stdout/stderr and via the logging system.  Suppress them all.
        xhtml2pdf_logger = logging.getLogger("xhtml2pdf")
        prev_level = xhtml2pdf_logger.level
        xhtml2pdf_logger.setLevel(logging.CRITICAL)
        saved_stdout, saved_stderr = sys.stdout, sys.stderr
        devnull = open(os.devnull, "w")  # noqa: SIM115
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            status = pisa.CreatePDF(html_text, dest=pdf_buffer, encoding="utf-8")
        except Exception as error:
            raise ValueError(
                f"xhtml2pdf could not render {source_path.name}: {error}"
            ) from error
        finally:
            sys.stdout = saved_stdout
            sys.stderr = saved_stderr
            devnull.close()
            xhtml2pdf_logger.setLevel(prev_level)

        if status.err:
            raise ValueError(f"xhtml2pdf reported errors rendering {source_path.name}.")
        pdf_bytes = pdf_buffer.getvalue()
        if not pdf_bytes:
            raise ValueError(f"xhtml2pdf produced no PDF bytes for {source_path.name}.")
        return pdf_bytes


class NormalizationService:
    """Chooses the right converter path for each supported source file."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        converter: MarkItDown | None = None,
        pdf_converter: DoclingPdfConverter | None = None,
        mistral_ocr_converter: MistralOcrConverter | None = None,
        html_renderer: WkhtmltopdfRenderer | None = None,
    ) -> None:
        self._config = _merged_config(config)
        self._converter = converter
        self._pdf_converter = pdf_converter
        self._mistral_ocr_converter = mistral_ocr_converter
        self._html_renderer = html_renderer

    def normalize_path(self, source_path: Path) -> NormalizationResult:
        """Normalize path.

        Args:
            source_path: Source path being processed.

        Returns:
            NormalizationResult produced by the operation.
        """
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

        if suffix in SUPPORTED_MISTRAL_DOCUMENT_SUFFIXES:
            return self._normalize_mistral_document(source_path)

        if suffix in SUPPORTED_MISTRAL_IMAGE_SUFFIXES:
            return self._normalize_mistral_image(source_path)

        if suffix in SUPPORTED_HTML_SUFFIXES:
            return self._normalize_html(source_path)

        if suffix in SUPPORTED_MARKITDOWN_SUFFIXES:
            return self._normalize_with_markitdown(
                source_path,
                route=MARKITDOWN_ROUTE,
            )

        raise ValueError(
            "Supported ingest inputs are canonical text (.md, .markdown, .txt), "
            "Mistral OCR-native documents (.pdf, .docx, .pptx), Mistral OCR-native "
            "images (.png, .jpg, .jpeg, .avif), HTML rendered through wkhtmltopdf "
            "(.htm, .html), and MarkItDown-backed formats "
            "(.csv, .epub, .ipynb, .xls, .xlsx)."
        )

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

    def _normalize_mistral_document(self, source_path: Path) -> NormalizationResult:
        suffix = source_path.suffix.lower()
        try:
            candidate = self._candidate_from_mistral_document(source_path)
            return self._build_result(source_path, candidate)
        except ValueError as primary_error:
            fallbacks = self._fallback_names(suffix)
            logger.warning(
                "Primary Mistral OCR route rejected %s; trying fallback chain %s: %s",
                source_path.name,
                ", ".join(fallbacks) or "none configured",
                primary_error,
            )
            fallback_errors: list[str] = []
            for fallback in fallbacks:
                try:
                    if fallback == "docling" and suffix == ".pdf":
                        return self._build_fallback_result(
                            source_path,
                            self._candidate_from_docling_pdf(source_path),
                            fallback_name=fallback,
                            fallback_route=PDF_FALLBACK_ROUTE,
                            primary_error=primary_error,
                        )
                    if fallback == "markitdown":
                        fallback_route = (
                            PDF_MARKITDOWN_FALLBACK_ROUTE
                            if suffix == ".pdf"
                            else DOCX_PPTX_FALLBACK_ROUTE
                        )
                        return self._build_fallback_result(
                            source_path,
                            self._candidate_from_markitdown(
                                source_path,
                                route=fallback_route,
                            ),
                            fallback_name=fallback,
                            fallback_route=fallback_route,
                            primary_error=primary_error,
                        )
                    fallback_errors.append(f"{fallback} is not supported for {suffix}")
                except ValueError as fallback_error:
                    fallback_errors.append(f"{fallback} failed ({fallback_error})")
            if fallback_errors:
                raise ValueError(
                    f"All conversion routes failed for {source_path.name}: "
                    f"primary Mistral OCR failed ({primary_error}); "
                    f"fallbacks failed: {'; '.join(fallback_errors)}."
                ) from primary_error
            raise ValueError(
                f"Mistral OCR could not convert {source_path.name}: {primary_error}"
            ) from primary_error

    def _normalize_mistral_image(self, source_path: Path) -> NormalizationResult:
        candidate = self._candidate_from_mistral_image(source_path)
        return self._build_result(source_path, candidate)

    def _normalize_html(self, source_path: Path) -> NormalizationResult:
        try:
            candidate = self._candidate_from_html_mistral(source_path)
            return self._build_result(source_path, candidate)
        except ValueError as primary_error:
            fallback = self._first_fallback_name(source_path.suffix.lower())
            logger.warning(
                "Primary HTML OCR route rejected %s; trying %s fallback: %s",
                source_path.name,
                fallback or "no configured",
                primary_error,
            )
            if fallback != "markitdown":
                raise ValueError(
                    f"HTML conversion failed for {source_path.name}: {primary_error}"
                ) from primary_error
            return self._build_result_with_fallback(
                source_path,
                self._candidate_from_markitdown(
                    source_path,
                    route=HTML_FALLBACK_ROUTE,
                ),
                fallback_name=fallback,
                fallback_route=HTML_FALLBACK_ROUTE,
                primary_error=primary_error,
            )

    def _normalize_with_markitdown(
        self,
        source_path: Path,
        *,
        route: str,
    ) -> NormalizationResult:
        candidate = self._candidate_from_markitdown(source_path, route=route)
        return self._build_result(source_path, candidate)

    def _candidate_from_mistral_document(self, source_path: Path) -> _ConvertedText:
        suffix = source_path.suffix.lower()
        mistral_config = self._mistral_config()
        converted = self._mistral_ocr_converter_instance().convert_document_bytes(
            _read_inline_ocr_payload(source_path),
            mime_type=MISTRAL_DOCUMENT_MIME_TYPES[suffix],
            document_name=source_path.name,
        )
        converted = _coerce_converted_text(converted)
        converted.metadata.update(
            {
                "converter": "mistral-ocr",
                "converter_version": package_version("mistralai"),
                "ingest_mode": MISTRAL_OCR_CONVERSION_INGEST_MODE,
                "canonical_text_format": ".md",
                "normalization_route": MISTRAL_DOCUMENT_ROUTE,
                "ocr_model": mistral_config["model"],
                "table_format": mistral_config["table_format"],
            }
        )
        return converted

    def _candidate_from_mistral_image(self, source_path: Path) -> _ConvertedText:
        suffix = source_path.suffix.lower()
        mistral_config = self._mistral_config()
        converted = self._mistral_ocr_converter_instance().convert_image_bytes(
            _read_inline_ocr_payload(source_path),
            mime_type=MISTRAL_IMAGE_MIME_TYPES[suffix],
        )
        converted = _coerce_converted_text(converted)
        converted.metadata.update(
            {
                "converter": "mistral-ocr",
                "converter_version": package_version("mistralai"),
                "ingest_mode": MISTRAL_OCR_CONVERSION_INGEST_MODE,
                "canonical_text_format": ".md",
                "normalization_route": MISTRAL_IMAGE_ROUTE,
                "ocr_model": mistral_config["model"],
                "table_format": mistral_config["table_format"],
            }
        )
        return converted

    def _candidate_from_html_mistral(self, source_path: Path) -> _ConvertedText:
        mistral_config = self._mistral_config()
        renderer = self._html_renderer_instance()
        rendered_pdf: bytes | None = None
        html_renderer_name = "wkhtmltopdf"
        route = HTML_RENDERED_OCR_ROUTE

        # Try wkhtmltopdf first (silently).
        try:
            rendered_pdf = renderer.render_file(source_path)
        except (ValueError, Exception) as wk_err:
            logger.debug("wkhtmltopdf failed for %s: %s", source_path.name, wk_err)

        # Fall back to xhtml2pdf silently.
        if rendered_pdf is None and Xhtml2pdfRenderer.available():
            try:
                rendered_pdf = Xhtml2pdfRenderer().render_file(source_path)
                html_renderer_name = "xhtml2pdf"
                route = HTML_XHTML2PDF_OCR_ROUTE
            except (ValueError, Exception) as xh_err:
                logger.debug("xhtml2pdf failed for %s: %s", source_path.name, xh_err)

        if rendered_pdf is None:
            raise ValueError(
                f"No HTML renderer could produce a PDF for {source_path.name}. "
                "Ensure wkhtmltopdf or xhtml2pdf is available."
            )

        _ensure_inline_ocr_payload_size(
            len(rendered_pdf),
            f"{source_path.stem}.pdf",
        )
        converted = self._mistral_ocr_converter_instance().convert_document_bytes(
            rendered_pdf,
            mime_type="application/pdf",
            document_name=f"{source_path.stem}.pdf",
        )
        converted = _coerce_converted_text(converted)
        renderer_meta: dict[str, Any] = {
            "converter": "mistral-ocr",
            "converter_version": package_version("mistralai"),
            "ingest_mode": MISTRAL_OCR_CONVERSION_INGEST_MODE,
            "canonical_text_format": ".md",
            "normalization_route": route,
            "ocr_model": mistral_config["model"],
            "table_format": mistral_config["table_format"],
            "html_renderer": html_renderer_name,
        }
        if html_renderer_name == "wkhtmltopdf":
            renderer_meta["wkhtmltopdf_path"] = renderer.resolve_binary()
        converted.metadata.update(renderer_meta)
        return converted

    def _candidate_from_docling_pdf(self, source_path: Path) -> _ConvertedText:
        converted = _coerce_converted_text(
            self._pdf_converter_instance().convert_local(source_path)
        )
        converted.metadata.update(
            {
                "converter": "docling",
                "converter_version": package_version("docling"),
                "ingest_mode": DOCLING_PDF_CONVERSION_INGEST_MODE,
                "canonical_text_format": ".md",
                "normalization_route": PDF_FALLBACK_ROUTE,
            }
        )
        return converted

    def _candidate_from_markitdown(
        self,
        source_path: Path,
        *,
        route: str,
    ) -> _ConvertedText:
        try:
            result = self._converter_instance().convert_local(source_path)
        except _markitdown_exception_types() as error:
            raise ValueError(
                f"MarkItDown could not convert {source_path.name}: {error}"
            ) from error
        except Exception as error:
            raise ValueError(
                f"MarkItDown could not convert {source_path.name}: {error}"
            ) from error

        title = result.title.strip() if getattr(result, "title", None) else None
        return _ConvertedText(
            normalized_text=result.markdown,
            title=title,
            metadata={
                "converter": "markitdown",
                "converter_version": _markitdown_version(),
                "ingest_mode": MARKITDOWN_CONVERSION_INGEST_MODE,
                "canonical_text_format": ".md",
                "normalization_route": route,
            },
        )

    def _build_result(
        self,
        source_path: Path,
        candidate: _ConvertedText,
    ) -> NormalizationResult:
        normalized_text = _ensure_trailing_newline(candidate.normalized_text)
        _validate_conversion_output(
            normalized_text,
            page_count=candidate.page_count,
            source_name=source_path.name,
        )
        title = (candidate.title or "").strip() or _extract_title(
            normalized_text,
            source_path,
        )
        return NormalizationResult(
            normalized_text=normalized_text,
            normalized_suffix=".md",
            title=title,
            metadata=candidate.metadata,
        )

    def _build_result_with_fallback(
        self,
        source_path: Path,
        candidate: _ConvertedText,
        *,
        fallback_name: str,
        fallback_route: str,
        primary_error: ValueError,
    ) -> NormalizationResult:
        try:
            return self._build_fallback_result(
                source_path,
                candidate,
                fallback_name=fallback_name,
                fallback_route=fallback_route,
                primary_error=primary_error,
            )
        except ValueError as fallback_error:
            raise ValueError(
                f"All conversion routes failed for {source_path.name}: "
                f"primary Mistral OCR failed ({primary_error}); "
                f"fallback {fallback_name} failed ({fallback_error})."
            ) from fallback_error

    def _build_fallback_result(
        self,
        source_path: Path,
        candidate: _ConvertedText,
        *,
        fallback_name: str,
        fallback_route: str,
        primary_error: ValueError,
    ) -> NormalizationResult:
        result = self._build_result(source_path, candidate)
        result.metadata["fallback_used"] = True
        result.metadata["fallback_converter"] = fallback_name
        result.metadata["fallback_route"] = fallback_route
        result.metadata["primary_converter"] = "mistral-ocr"
        result.metadata["primary_error"] = str(primary_error)
        return result

    def _converter_instance(self) -> MarkItDown:
        if self._converter is None:
            from markitdown import MarkItDown

            self._converter = MarkItDown(enable_plugins=False)
        return self._converter

    def _pdf_converter_instance(self) -> DoclingPdfConverter:
        if self._pdf_converter is None:
            self._pdf_converter = DoclingPdfConverter()
        return self._pdf_converter

    def _mistral_ocr_converter_instance(self) -> MistralOcrConverter:
        if self._mistral_ocr_converter is None:
            mistral_config = self._mistral_config()
            self._mistral_ocr_converter = MistralOcrConverter(
                model=mistral_config["model"],
                api_key_env=mistral_config["api_key_env"],
                table_format=mistral_config["table_format"],
            )
        return self._mistral_ocr_converter

    def _html_renderer_instance(self) -> WkhtmltopdfRenderer:
        if self._html_renderer is None:
            html_config = self._html_config()
            wkhtmltopdf_path = html_config.get("wkhtmltopdf_path")
            self._html_renderer = WkhtmltopdfRenderer(
                wkhtmltopdf_path,
                allow_local_file_access=bool(
                    html_config.get("allow_local_file_access", False)
                ),
            )
        return self._html_renderer

    def _mistral_config(self) -> dict[str, Any]:
        conversion = self._config.get("conversion", {})
        mistral_ocr = conversion.get("mistral_ocr", {})
        return mistral_ocr if isinstance(mistral_ocr, dict) else {}

    def _html_config(self) -> dict[str, Any]:
        conversion = self._config.get("conversion", {})
        html = conversion.get("html", {})
        return html if isinstance(html, dict) else {}

    def _first_fallback_name(self, suffix: str) -> str:
        names = self._fallback_names(suffix)
        return names[0] if names else ""

    def _fallback_names(self, suffix: str) -> tuple[str, ...]:
        conversion = self._config.get("conversion", {})
        fallbacks = conversion.get("fallbacks", {})
        if not isinstance(fallbacks, dict):
            return ()
        if suffix == ".pdf":
            key = "pdf"
        elif suffix == ".docx":
            key = "docx"
        elif suffix == ".pptx":
            key = "pptx"
        else:
            key = "html"
        value = fallbacks.get(key, "")
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple)):
            values = list(value)
        else:
            return ()
        names: list[str] = []
        for raw_name in values:
            if not isinstance(raw_name, str):
                continue
            name = raw_name.strip().lower()
            if name == "none":
                continue
            if name and name not in names:
                names.append(name)
        return tuple(names)


def is_supported_source_path(source_path: Path) -> bool:
    """Is supported source path.

    Args:
        source_path: Source path being processed.

    Returns:
        bool produced by the operation.
    """
    return source_path.suffix.lower() in SUPPORTED_SOURCE_SUFFIXES


def resolve_wkhtmltopdf_binary(config: dict[str, Any] | None = None) -> str | None:
    """Resolve wkhtmltopdf binary.

    Args:
        config: Loaded knowledge-base configuration mapping.

    Returns:
        str | None produced by the operation.
    """
    merged = _merged_config(config)
    html = merged.get("conversion", {}).get("html", {})
    if not isinstance(html, dict):
        return None
    configured = html.get("wkhtmltopdf_path")
    if isinstance(configured, str) and configured.strip():
        return str(Path(configured))
    return shutil.which("wkhtmltopdf")


def _merged_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = deepcopy(DEFAULT_CONFIG)
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = _merge_nested(merged[key], value)
            else:
                merged[key] = deepcopy(value)
    return merged


def _merge_nested(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _data_uri(mime_type: str, payload: bytes) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _read_inline_ocr_payload(path: Path) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(
            f"Could not read {path.name} for OCR conversion: {exc}"
        ) from exc
    _ensure_inline_ocr_payload_size(size, path.name)
    return path.read_bytes()


def _ensure_inline_ocr_payload_size(size: int, source_name: str) -> None:
    if size <= MAX_INLINE_OCR_PAYLOAD_BYTES:
        return
    limit_mb = MAX_INLINE_OCR_PAYLOAD_BYTES // (1024 * 1024)
    actual_mb = size / (1024 * 1024)
    raise ValueError(
        f"{source_name} is {actual_mb:.1f} MB, which exceeds the {limit_mb} MB "
        "inline OCR upload limit. Split the file or use a provider upload/document-url "
        "workflow before ingesting it."
    )


def _join_ocr_pages(pages: list[Any]) -> str:
    ordered = sorted(pages, key=lambda page: getattr(page, "index", 0))
    parts = [str(getattr(page, "markdown", "") or "").strip() for page in ordered]
    return "\n\n".join(part for part in parts if part)


def _validate_conversion_output(
    contents: str,
    *,
    page_count: int | None,
    source_name: str,
) -> None:
    text = _normalize_newlines(contents).strip()
    if not text:
        raise ValueError(f"{source_name} conversion produced empty output.")

    plain = _plain_text(contents)
    if len(plain) < 20:
        raise ValueError(f"{source_name} conversion produced too little text.")

    if not _usable_paragraphs(contents) and len(plain.split()) < 5:
        raise ValueError(f"{source_name} conversion produced no usable body text.")

    if page_count and page_count > 1 and len(plain) < page_count * 60:
        raise ValueError(
            f"{source_name} conversion output is implausibly short for {page_count} pages."
        )

    if page_count and _looks_truncated(contents, plain, page_count=page_count):
        raise ValueError(f"{source_name} conversion output appears truncated.")


def _looks_truncated(contents: str, plain_text: str, *, page_count: int) -> bool:
    """Return True only for clearly undersized multi-page conversions.

    Converter status/errors and the page-count ratio are more reliable than
    inspecting the final line for sentence punctuation. Markdown often ends in
    tables, bullets, references, or captions that are valid but not prose.
    """
    if page_count <= 1:
        return False
    return len(plain_text) < page_count * 60


def _usable_paragraphs(contents: str) -> list[str]:
    return [
        paragraph
        for paragraph in markdown_paragraphs(
            contents,
            content_only=False,
            trim_leading_boilerplate=False,
        )
        if len(_plain_text(paragraph).split()) >= 5
    ]


def _plain_text(contents: str) -> str:
    """Extract plain text from markdown using markdown-it-py AST."""
    return " ".join(markdown_plain_text(contents).split()).strip()


def _ensure_trailing_newline(contents: str) -> str:
    return _normalize_newlines(contents).rstrip() + "\n"


def _filename_title(source_path: Path) -> str:
    return source_path.stem.replace("_", " ").replace("-", " ").title()


def _extract_title(contents: str, source_path: Path) -> str:
    document = parse_document(contents)
    heading_candidates: list[str] = []
    pre_heading_fallbacks: list[str] = []
    post_heading_fallbacks: list[str] = []
    saw_heading = False
    body = document.body if document.valid_frontmatter else contents
    for line in _normalize_newlines(body).splitlines()[:60]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("---"):
            continue
        is_heading = stripped.startswith("#")
        if is_heading:
            saw_heading = True
        candidate = stripped.lstrip("#").strip() if is_heading else stripped
        cleaned = _clean_title_candidate(candidate)
        if not cleaned or not _is_probable_title(cleaned):
            continue
        if is_heading:
            heading_candidates.append(cleaned)
        elif not saw_heading:
            pre_heading_fallbacks.append(cleaned)
        else:
            post_heading_fallbacks.append(cleaned)
    if pre_heading_fallbacks:
        return pre_heading_fallbacks[0]
    if heading_candidates:
        return heading_candidates[0]
    if post_heading_fallbacks:
        return post_heading_fallbacks[0]
    return _filename_title(source_path)


def _clean_title_candidate(candidate: str) -> str:
    cleaned = re.sub(r"\s+", " ", candidate).strip()
    return cleaned[:160].strip()


def _coerce_converted_text(converted: _ConvertedText | str) -> _ConvertedText:
    if isinstance(converted, _ConvertedText):
        return converted
    return _ConvertedText(normalized_text=str(converted))


def _is_probable_title(candidate: str) -> bool:
    normalized = candidate.strip().strip(":").strip()
    if len(normalized) < 3:
        return False
    lower = normalized.casefold()
    de_numbered = _NUMBERED_SECTION_PREFIX_PATTERN.sub("", lower).strip()
    if lower in _GENERIC_TITLE_HEADINGS:
        return False
    if de_numbered in _GENERIC_TITLE_HEADINGS:
        return False
    if _CAPTION_PATTERN.match(normalized):
        return False
    if "@" in normalized:
        return False
    if len(normalized.split()) > 20:
        return False
    if normalized.endswith(".") and len(normalized.split()) > 10:
        return False
    if (
        any(hint in lower for hint in _AFFILIATION_HINTS)
        and len(normalized.split()) <= 8
    ):
        return False
    return True


def _normalize_newlines(contents: str) -> str:
    return markdown_normalize_newlines(contents).lstrip("\ufeff")
