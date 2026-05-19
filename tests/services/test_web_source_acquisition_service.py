"""Tests for web source acquisition staging."""

from __future__ import annotations

from graphwiki_kb.agents.models import SourceRecommendation
from graphwiki_kb.services.web_source_acquisition_service import (
    WebSourceAcquisitionService,
)


def test_stage_writes_markdown(test_project, monkeypatch) -> None:
    service = WebSourceAcquisitionService(test_project.paths)

    def _fake_download(url: str, *, timeout: float = 30.0) -> str:
        return "<html><body><p>Hello web</p></body></html>"

    monkeypatch.setattr(service, "_download_text", _fake_download)
    recommendation = SourceRecommendation(
        id=1,
        title="Example Article",
        url="https://example.com/article",
        source_type="article",
        retrieved_at="2026-05-19T00:00:00+00:00",
        why_add="test",
        knowledge_gap="gap",
        novelty="medium",
        confidence="medium",
        ingestable=True,
    )
    staged = service.stage(recommendation, run_id="research_test")
    assert staged.staged_path.exists()
    assert staged.staged_path.suffix == ".md"
    assert "Hello web" in staged.staged_path.read_text(encoding="utf-8")
