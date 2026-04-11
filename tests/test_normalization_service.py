from __future__ import annotations

from pathlib import Path

import pytest

from src.services.normalization_service import NormalizationService


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
    assert "HTML Research Note" in result.normalized_text
    assert "Traceability matters." in result.normalized_text


def test_normalization_service_rejects_unsupported_suffix(tmp_path: Path) -> None:
    source_path = tmp_path / "sample.bin"
    source_path.write_text("bits", encoding="utf-8")

    with pytest.raises(ValueError, match="Supported ingest inputs are canonical text"):
        NormalizationService().normalize_path(source_path)
