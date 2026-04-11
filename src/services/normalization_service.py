from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from markitdown import __version__ as markitdown_version
from markitdown import MarkItDown, MarkItDownException


SUPPORTED_CANONICAL_TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
SUPPORTED_MARKITDOWN_SUFFIXES = {
    ".csv",
    ".docx",
    ".epub",
    ".htm",
    ".html",
    ".ipynb",
    ".pdf",
    ".pptx",
    ".xls",
    ".xlsx",
}

DIRECT_CANONICAL_TEXT_INGEST_MODE = "direct-canonical-text"
MARKITDOWN_CONVERSION_INGEST_MODE = "markitdown-convert"


@dataclass
class NormalizationResult:
    normalized_text: str
    normalized_suffix: str
    title: str
    metadata: dict[str, Any]


class NormalizationService:
    def __init__(self, converter: Optional[MarkItDown] = None) -> None:
        self._converter = converter

    def normalize_path(self, source_path: Path) -> NormalizationResult:
        suffix = source_path.suffix.lower()
        if suffix in SUPPORTED_CANONICAL_TEXT_SUFFIXES:
            contents = source_path.read_text(encoding="utf-8")
            normalized_text = _ensure_trailing_newline(contents)
            normalized_suffix = ".md" if suffix in {".md", ".markdown"} else ".txt"
            return NormalizationResult(
                normalized_text=normalized_text,
                normalized_suffix=normalized_suffix,
                title=_extract_title(normalized_text, source_path),
                metadata={
                    "converter": "direct-copy",
                    "ingest_mode": DIRECT_CANONICAL_TEXT_INGEST_MODE,
                    "canonical_text_format": normalized_suffix,
                },
            )

        if suffix not in SUPPORTED_MARKITDOWN_SUFFIXES:
            raise ValueError(
                "Supported ingest inputs are canonical text (.md, .markdown, .txt) "
                "and MarkItDown-backed formats "
                "(.csv, .docx, .epub, .htm, .html, .ipynb, .pdf, .pptx, .xls, .xlsx)."
            )

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
            },
        )

    def _converter_instance(self) -> MarkItDown:
        if self._converter is None:
            self._converter = MarkItDown(enable_plugins=False)
        return self._converter


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
