"""Tests for the ingest_recommendation agent tool and acquisition service."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphwiki_kb.agents.models import (
    IngestRecommendationInput,
    ResearchRunRecord,
    SourceRecommendation,
)
from graphwiki_kb.agents.tools.ingest_recommendation import (
    run_ingest_recommendation,
)
from graphwiki_kb.services.source_recommendation_store import (
    SourceRecommendationStore,
)
from graphwiki_kb.services.web_source_acquisition_service import (
    WebSourceAcquisitionError,
    WebSourceAcquisitionService,
)


def _seed_run(store: SourceRecommendationStore) -> ResearchRunRecord:
    record = ResearchRunRecord(
        run_id="research_20260519T120000Z_q",
        question="benchmark question",
        created_at="2026-05-19T12:00:00+00:00",
        local_answer={"answer": "..."},
        kb_gaps=["a gap"],
        web_findings=[],
        recommendations=[
            SourceRecommendation(
                id=1,
                title="Paper One",
                url="https://example.com/paper.md",
                source_type="paper",
                retrieved_at="2026-05-19T12:00:00+00:00",
                why_add="r",
            ),
            SourceRecommendation(
                id=2,
                title="Paper Two",
                url="https://example.com/two.md",
                source_type="paper",
                retrieved_at="2026-05-19T12:00:00+00:00",
                why_add="r",
                ingestable=False,
            ),
        ],
    )
    store.save(record)
    return record


def _stub_fetcher(payload: bytes, *, content_type: str = "text/markdown"):
    def _fetch(url: str, *, timeout: int, max_bytes: int) -> tuple[bytes, str]:
        return payload, content_type

    return _fetch


def test_ingest_recommendation_pauses_for_approval(runtime, test_project) -> None:
    _seed_run(runtime.recommendation_store)
    payload = IngestRecommendationInput(run_id="latest", ids=[1])

    output = run_ingest_recommendation(runtime, payload)

    assert output.results == []
    assert len(runtime.pending_approvals) == 1
    approval = runtime.pending_approvals[0]
    assert approval.tool_name == "ingest_recommendation"
    assert approval.payload["ids"] == [1]


def test_ingest_recommendation_with_auto_approve_stages_and_ingests(
    runtime, test_project
) -> None:
    _seed_run(runtime.recommendation_store)
    runtime.auto_approve = True
    acquisition = WebSourceAcquisitionService(
        test_project.paths,
        http_fetcher=_stub_fetcher(b"# Paper\n\nbody"),
    )
    runtime.metadata["web_source_acquisition"] = acquisition

    output = run_ingest_recommendation(
        runtime,
        IngestRecommendationInput(run_id="latest", ids=[1]),
    )

    assert len(output.results) == 1
    item = output.results[0]
    assert item.id == 1
    assert item.created is True
    assert item.source_id is not None
    assert item.staged_path is not None
    assert Path(item.staged_path).exists()
    assert output.next_command == "kb update"


def test_ingest_recommendation_skips_non_ingestable(runtime, test_project) -> None:
    _seed_run(runtime.recommendation_store)
    runtime.auto_approve = True
    acquisition = WebSourceAcquisitionService(
        test_project.paths,
        http_fetcher=_stub_fetcher(b"# x"),
    )
    runtime.metadata["web_source_acquisition"] = acquisition

    output = run_ingest_recommendation(
        runtime,
        IngestRecommendationInput(run_id="latest", ids=[2]),
    )

    assert len(output.results) == 1
    item = output.results[0]
    assert item.created is False
    assert "not ingestable" in item.message.lower()


def test_acquisition_service_rejects_non_http_urls(test_project) -> None:
    service = WebSourceAcquisitionService(
        test_project.paths,
        http_fetcher=_stub_fetcher(b"x"),
    )
    rec = SourceRecommendation(
        id=1,
        title="Bad",
        url="ftp://nope/x",
        source_type="paper",
        retrieved_at="2026-05-19T12:00:00+00:00",
        why_add="r",
    )
    with pytest.raises(WebSourceAcquisitionError):
        service.stage_recommendation(rec, run_id="r")


def test_ingest_recommendation_reports_acquisition_failure(
    runtime, test_project
) -> None:
    _seed_run(runtime.recommendation_store)
    runtime.auto_approve = True

    def _failing_fetcher(url, *, timeout, max_bytes):
        raise RuntimeError("network down")

    runtime.metadata["web_source_acquisition"] = WebSourceAcquisitionService(
        test_project.paths,
        http_fetcher=_failing_fetcher,
    )

    output = run_ingest_recommendation(
        runtime,
        IngestRecommendationInput(run_id="latest", ids=[1]),
    )
    assert len(output.results) == 1
    assert output.results[0].created is False
    assert "Failed to stage" in output.results[0].message


def test_ingest_recommendation_unknown_run_returns_empty(runtime) -> None:
    output = run_ingest_recommendation(
        runtime,
        IngestRecommendationInput(run_id="does-not-exist", ids=[1]),
    )
    assert output.results == []
    assert runtime.tool_results[-1].ok is False


def test_acquisition_service_detects_html_by_leading_tag(test_project) -> None:
    service = WebSourceAcquisitionService(
        test_project.paths,
        http_fetcher=_stub_fetcher(b"<html><body>Hi</body></html>", content_type=""),
    )
    rec = SourceRecommendation(
        id=1,
        title="HTML",
        url="https://example.com/x",
        source_type="article",
        retrieved_at="2026-05-19T12:00:00+00:00",
        why_add="r",
    )
    staged = service.stage_recommendation(rec, run_id="r")
    assert staged.suffix == ".html"
    text = staged.staged_path.read_text(encoding="utf-8")
    assert "Hi" in text  # stripped HTML content


def test_acquisition_service_uses_url_suffix_when_unknown(test_project) -> None:
    service = WebSourceAcquisitionService(
        test_project.paths,
        http_fetcher=_stub_fetcher(b"plain text body", content_type=""),
    )
    rec = SourceRecommendation(
        id=1,
        title="Text",
        url="https://example.com/note.txt",
        source_type="docs",
        retrieved_at="2026-05-19T12:00:00+00:00",
        why_add="r",
    )
    staged = service.stage_recommendation(rec, run_id="r")
    assert staged.suffix == ".txt"


def test_acquisition_service_rejects_oversized_body(test_project) -> None:
    body = b"x" * 100
    service = WebSourceAcquisitionService(
        test_project.paths,
        http_fetcher=_stub_fetcher(body, content_type="text/plain"),
        max_bytes=10,
    )
    rec = SourceRecommendation(
        id=1,
        title="Big",
        url="https://example.com/x",
        source_type="docs",
        retrieved_at="2026-05-19T12:00:00+00:00",
        why_add="r",
    )
    with pytest.raises(WebSourceAcquisitionError):
        service.stage_recommendation(rec, run_id="r")


def test_acquisition_service_rejects_empty_body(test_project) -> None:
    service = WebSourceAcquisitionService(
        test_project.paths,
        http_fetcher=_stub_fetcher(b"", content_type="text/plain"),
    )
    rec = SourceRecommendation(
        id=1,
        title="Empty",
        url="https://example.com/x",
        source_type="docs",
        retrieved_at="2026-05-19T12:00:00+00:00",
        why_add="r",
    )
    with pytest.raises(WebSourceAcquisitionError):
        service.stage_recommendation(rec, run_id="r")


def test_acquisition_service_detects_pdf_by_magic_bytes(test_project) -> None:
    service = WebSourceAcquisitionService(
        test_project.paths,
        http_fetcher=_stub_fetcher(b"%PDF-1.4 fake body", content_type=""),
    )
    rec = SourceRecommendation(
        id=1,
        title="PDF",
        url="https://example.com/x",
        source_type="paper",
        retrieved_at="2026-05-19T12:00:00+00:00",
        why_add="r",
    )
    staged = service.stage_recommendation(rec, run_id="r")
    assert staged.suffix == ".pdf"
    assert staged.staged_path.read_bytes().startswith(b"%PDF-")
