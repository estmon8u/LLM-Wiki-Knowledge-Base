"""PDF staging tests for web source acquisition."""

from __future__ import annotations

from graphwiki_kb.agents.models import SourceRecommendation
from graphwiki_kb.services.web_source_acquisition_service import (
    WebSourceAcquisitionService,
)


def test_stage_pdf_download(test_project, monkeypatch) -> None:
    service = WebSourceAcquisitionService(test_project.paths)

    def _fake_binary(url: str, dest, *, timeout: float = 60.0) -> None:
        dest.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(service, "_download_binary", _fake_binary)
    recommendation = SourceRecommendation(
        id=1,
        title="Paper",
        url="https://example.com/paper.pdf",
        source_type="paper",
        retrieved_at="2026-05-19T00:00:00+00:00",
        why_add="test",
        knowledge_gap="gap",
        novelty="medium",
        confidence="medium",
        ingestable=True,
    )
    staged = service.stage(recommendation, run_id="run_pdf")
    assert staged.staged_path.suffix == ".pdf"
    assert staged.staged_path.exists()
