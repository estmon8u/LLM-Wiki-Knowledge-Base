from __future__ import annotations

import threading

import pytest

from src.providers import ProviderConfigurationError, ProviderExecutionError
from src.providers.base import ProviderRequest, ProviderResponse, TextProvider
from src.services.review_service import ReviewService


class SequencedReviewProvider(TextProvider):
    name = "fake-review"

    def __init__(
        self, responses: list[object], model_name: str = "fake-review-v1"
    ) -> None:
        self._responses = list(responses)
        self._model_name = model_name
        self._lock = threading.Lock()
        self.requests: list[ProviderRequest] = []

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        with self._lock:
            self.requests.append(request)
            if not self._responses:
                raise AssertionError("No review response remaining.")
            response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return ProviderResponse(text=str(response), model_name=self._model_name)


def test_review_service_finds_no_issues_on_empty_wiki(test_project) -> None:
    report = test_project.services["review"].review()

    assert report.issue_count == 0
    assert report.mode == "no-sources"


def test_review_service_detects_overlapping_source_topics(test_project) -> None:
    # Two source pages with heavily overlapping terminology
    test_project.write_file(
        "wiki/sources/alpha.md",
        "knowledge base traceability citation markdown wiki compile ingest lint",
    )
    test_project.write_file(
        "wiki/sources/beta.md",
        "knowledge base traceability citation markdown wiki compile query lint",
    )

    report = test_project.services["review"].review()

    overlap_issues = [i for i in report.issues if i.code == "overlapping-topics"]
    assert len(overlap_issues) >= 1
    assert "wiki/sources/alpha.md" in overlap_issues[0].pages
    assert "wiki/sources/beta.md" in overlap_issues[0].pages


def test_review_service_does_not_flag_distinct_source_pages(test_project) -> None:
    test_project.write_file(
        "wiki/sources/alpha.md",
        "knowledge base traceability citation provenance compile",
    )
    test_project.write_file(
        "wiki/sources/beta.md",
        "neural network gradient descent optimizer transformer attention",
    )

    report = test_project.services["review"].review()

    overlap_issues = [i for i in report.issues if i.code == "overlapping-topics"]
    assert len(overlap_issues) == 0


def test_review_service_check_summary_overlap_skips_empty_union(test_project) -> None:
    service = ReviewService(test_project.paths)

    issues = service._check_summary_overlap(
        {
            "wiki/sources/a.md": [],
            "wiki/sources/b.md": [],
        }
    )

    assert issues == []


def test_review_service_detects_terminology_variants(test_project) -> None:
    test_project.write_file(
        "wiki/sources/page-a.md",
        "The knowledge-base system handles traceability.",
    )
    test_project.write_file(
        "wiki/sources/page-b.md",
        "The knowledgebase system handles compilation.",
    )

    report = test_project.services["review"].review()

    variant_issues = [i for i in report.issues if i.code == "terminology-variant"]
    assert len(variant_issues) >= 1
    variants_text = " ".join(i.message for i in variant_issues)
    assert "knowledge-base" in variants_text or "knowledgebase" in variants_text


def test_review_report_properties() -> None:
    from src.models.wiki_models import ReviewIssue, ReviewReport

    report = ReviewReport(
        issues=[
            ReviewIssue("suggestion", "overlapping-topics", ["a.md", "b.md"], "test"),
            ReviewIssue("suggestion", "terminology-variant", ["a.md"], "variant"),
        ],
        mode="provider:stub-1",
    )

    assert report.issue_count == 2
    assert report.mode == "provider:stub-1"


# --- P1 boundary/negative tests ---


def test_review_overlap_exactly_at_threshold(test_project) -> None:
    # Jaccard = 11/20 = 0.55 (exactly at _OVERLAP_THRESHOLD)
    # Shared 11: knowledge system traceability citation markdown compile query ingest lint review vault
    # A-only 4: normalize config schema provider  → set_a = 15
    # B-only 5: export search status model deploy  → set_b = 16, union = 20
    test_project.write_file(
        "wiki/sources/overlap-a.md",
        "knowledge system traceability citation markdown "
        "compile query ingest lint review vault "
        "normalize config schema provider",
    )
    test_project.write_file(
        "wiki/sources/overlap-b.md",
        "knowledge system traceability citation markdown "
        "compile query ingest lint review vault "
        "export search status model deploy",
    )

    report = test_project.services["review"].review()
    overlap_issues = [i for i in report.issues if i.code == "overlapping-topics"]

    assert len(overlap_issues) >= 1


def test_review_overlap_below_threshold(test_project) -> None:
    # Jaccard = 6/11 ≈ 0.5454 (below 0.55 threshold)
    # Shared 6: knowledge system traceability citation markdown compile
    # A-only 2: normalize config  → set_a = 8
    # B-only 3: export search status  → set_b = 9, union = 11
    test_project.write_file(
        "wiki/sources/below-a.md",
        "knowledge system traceability citation markdown compile normalize config",
    )
    test_project.write_file(
        "wiki/sources/below-b.md",
        "knowledge system traceability citation markdown compile export search status",
    )

    report = test_project.services["review"].review()
    overlap_issues = [
        i
        for i in report.issues
        if i.code == "overlapping-topics"
        and "below-a" in " ".join(i.pages)
        and "below-b" in " ".join(i.pages)
    ]

    assert len(overlap_issues) == 0


def test_review_multi_agent_terminology_variant(test_project) -> None:
    test_project.write_file(
        "wiki/sources/variant-a.md",
        "The multi-agent system handles reasoning.",
    )
    test_project.write_file(
        "wiki/sources/variant-b.md",
        "The multiagent system handles compilation.",
    )

    report = test_project.services["review"].review()
    variant_issues = [i for i in report.issues if i.code == "terminology-variant"]

    assert len(variant_issues) >= 1
    variants_text = " ".join(i.message for i in variant_issues)
    assert "multi-agent" in variants_text or "multiagent" in variants_text


def test_review_concept_only_pages_no_overlap(test_project) -> None:
    test_project.write_file(
        "wiki/concepts/topic-a.md",
        "knowledge base traceability citation markdown wiki compile ingest lint query",
    )
    test_project.write_file(
        "wiki/concepts/topic-b.md",
        "knowledge base traceability citation markdown wiki compile ingest lint query",
    )

    report = test_project.services["review"].review()
    overlap_issues = [i for i in report.issues if i.code == "overlapping-topics"]

    assert len(overlap_issues) == 0


def test_review_single_source_page_no_overlap(test_project) -> None:
    test_project.write_file(
        "wiki/sources/solo.md",
        "knowledge base traceability citation markdown wiki compile ingest lint query",
    )

    report = test_project.services["review"].review()
    overlap_issues = [i for i in report.issues if i.code == "overlapping-topics"]

    assert len(overlap_issues) == 0


def test_review_service_provider_review_on_empty_sources_dir_returns_no_sources(
    test_project,
) -> None:
    provider = SequencedReviewProvider([])
    service = ReviewService(test_project.paths, provider=provider)
    test_project.paths.wiki_sources_dir.mkdir(parents=True, exist_ok=True)

    issues, mode = service._provider_review()

    assert issues == []
    assert mode == "no-sources"


def test_review_service_without_provider_raises_configuration_error(
    test_project,
) -> None:
    service = ReviewService(test_project.paths, provider=None)

    with pytest.raises(ProviderConfigurationError, match="kb review"):
        service.review()


def test_review_service_parse_provider_issues_skips_malformed_lines() -> None:
    raw = (
        "ISSUE|error|contradiction|a.md, b.md|Pages disagree\n"
        "ISSUE|too-few-parts|only-three\n"
        "random noise\n"
        "NO_ISSUES\n"
        "ISSUE|badlevel|stale|c.md|Stale content\n"
        "ISSUE|warning|term-drift|d.md|Term mismatch\n"
    )

    issues = ReviewService._parse_provider_issues(raw)

    assert len(issues) == 3
    assert issues[0].severity == "error"
    assert issues[0].code == "contradiction"
    assert issues[0].pages == ["a.md", "b.md"]
    assert issues[1].severity == "suggestion"  # badlevel → suggestion fallback
    assert issues[2].severity == "warning"
    assert issues[2].code == "term-drift"


def test_review_service_provider_review_raises_on_provider_failure(
    test_project,
) -> None:
    test_project.write_file("wiki/sources/alpha.md", "# Alpha\n\nContent.\n")
    provider = SequencedReviewProvider([RuntimeError("provider crash")])
    service = ReviewService(test_project.paths, provider=provider)

    with pytest.raises(ProviderExecutionError, match="provider crash"):
        service._provider_review()
