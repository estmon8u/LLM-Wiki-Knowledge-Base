from __future__ import annotations

import threading

import pytest

from src.providers import ProviderConfigurationError, ProviderExecutionError
from src.providers.base import ProviderRequest, ProviderResponse, TextProvider
from src.schemas.review import ReviewFinding, Verdict
from src.services.review_service import (
    PageSnapshot,
    ReviewPair,
    ReviewService,
)
from src.storage.run_store import RunStore


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


def test_review_service_adversarial_on_empty_wiki_returns_provider_mode(
    test_project,
) -> None:
    provider = SequencedReviewProvider([])
    service = ReviewService(test_project.paths, provider=provider)

    report = service.review(adversarial=True)

    assert report.mode == "adversarial:fake-review"
    assert report.findings == []


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


def test_review_service_adversarial_without_provider_raises_configuration_error(
    test_project,
) -> None:
    service = ReviewService(test_project.paths, provider=None)

    with pytest.raises(ProviderConfigurationError, match="kb review"):
        service.review(adversarial=True)


def test_review_service_build_candidate_pairs_uses_headings_years_and_variants(
    test_project,
) -> None:
    test_project.write_file(
        "wiki/sources/alpha.md",
        "# Alpha\n\n## Timeline\n\nThe multi-agent workflow changed in 2026.\n",
    )
    test_project.write_file(
        "wiki/sources/beta.md",
        "# Beta\n\n## Timeline\n\nThe multiagent process changed in 2026.\n",
    )
    service = ReviewService(test_project.paths)

    snapshots = service._load_source_page_snapshots()
    pairs = service._build_candidate_pairs(snapshots)

    assert len(pairs) == 1
    reasons = " | ".join(pairs[0].reasons)
    assert "shared headings:" in reasons
    assert "shared years:" in reasons
    assert "variant terms:" in reasons


def test_review_service_extract_page_title_falls_back_to_stem(test_project) -> None:
    service = ReviewService(test_project.paths)

    title = service._extract_page_title("alpha-page", "No top heading here.")

    assert title == "Alpha Page"


def test_review_service_provider_review_on_empty_sources_dir_returns_no_sources(
    test_project,
) -> None:
    provider = SequencedReviewProvider([])
    service = ReviewService(test_project.paths, provider=provider)
    test_project.paths.wiki_sources_dir.mkdir(parents=True, exist_ok=True)

    issues, mode = service._provider_review()

    assert issues == []
    assert mode == "no-sources"


def test_review_service_parse_extracted_claims_skips_malformed_lines() -> None:
    claims = ReviewService._parse_extracted_claims(
        "CLAIM|Claim A|a.md, b.md\n"
        "CLAIM|too-few\n"
        "random text\n"
        "NO_CLAIMS\n"
        "CLAIM|Claim B|b.md"
    )

    assert [claim.text for claim in claims] == ["Claim A", "Claim B"]
    assert claims[0].citations == ("a.md", "b.md")


def test_review_service_parse_adversarial_findings_defaults_unknown_values() -> None:
    service = ReviewService.__new__(ReviewService)
    pair = ReviewPair(
        left=PageSnapshot(
            path="wiki/sources/a.md",
            title="A",
            text="",
            tokens=set(),
            headings=set(),
            years=set(),
        ),
        right=PageSnapshot(
            path="wiki/sources/b.md",
            title="B",
            text="",
            tokens=set(),
            headings=set(),
            years=set(),
        ),
        reasons=("shared headings: timeline",),
    )

    findings = service._parse_adversarial_findings(
        "FINDING||unknown|oops|Claim text|For evidence|Against evidence|",
        pair,
    )

    assert len(findings) == 1
    assert findings[0].verdict == Verdict.NEEDS_REVIEW
    assert findings[0].confidence == 0.5
    assert findings[0].issue_type == "needs-review"
    assert findings[0].citations == ["wiki/sources/a.md", "wiki/sources/b.md"]


def test_review_service_findings_to_issues_skips_consistent_and_maps_severity(
    test_project,
) -> None:
    service = ReviewService(test_project.paths)
    findings = [
        ReviewFinding(
            issue_type="consistent",
            affected_pages=["a.md", "b.md"],
            claim="Pages agree",
            verdict=Verdict.CONSISTENT,
            confidence=0.9,
        ),
        ReviewFinding(
            issue_type="contradiction",
            affected_pages=["a.md", "b.md"],
            claim="Pages disagree about the release date",
            verdict=Verdict.CONTRADICTORY,
            confidence=0.9,
        ),
        ReviewFinding(
            issue_type="term-drift",
            affected_pages=["a.md", "b.md"],
            claim="Pages use different terminology for the same concept",
            verdict=Verdict.TERM_DRIFT,
            confidence=0.7,
        ),
        ReviewFinding(
            issue_type="needs-review",
            affected_pages=["a.md", "b.md"],
            claim="Evidence remains ambiguous",
            verdict=Verdict.NEEDS_REVIEW,
            confidence=0.5,
        ),
    ]

    issues = service._findings_to_issues(findings)

    assert [issue.code for issue in issues] == [
        "contradiction",
        "term-drift",
        "needs-review",
    ]
    assert [issue.severity for issue in issues] == [
        "error",
        "suggestion",
        "warning",
    ]


def test_review_service_review_summary_counts_verdicts(test_project) -> None:
    service = ReviewService(test_project.paths)

    summary = service._review_summary(
        [
            ReviewFinding(issue_type="a", verdict=Verdict.CONTRADICTORY),
            ReviewFinding(issue_type="b", verdict=Verdict.CONTRADICTORY),
            ReviewFinding(issue_type="c", verdict=Verdict.NEEDS_REVIEW),
        ]
    )

    assert "contradictory=2" in summary
    assert "needs_review=1" in summary


def test_review_service_review_summary_empty(test_project) -> None:
    service = ReviewService(test_project.paths)

    assert service._review_summary([]) == "No adversarial review findings."


def test_review_service_pair_pages_block_and_claims_block(test_project) -> None:
    service = ReviewService(test_project.paths)
    pair = ReviewPair(
        left=PageSnapshot(
            path="wiki/sources/a.md",
            title="Alpha",
            text="Alpha body",
            tokens=set(),
            headings=set(),
            years=set(),
        ),
        right=PageSnapshot(
            path="wiki/sources/b.md",
            title="Beta",
            text="Beta body",
            tokens=set(),
            headings=set(),
            years=set(),
        ),
        reasons=("shared headings: timeline",),
    )
    claims_block = service._claims_block([])
    pages_block = service._pair_pages_block(pair)

    assert claims_block == "NO_CLAIMS"
    assert "wiki/sources/a.md" in pages_block
    assert "wiki/sources/b.md" in pages_block
    assert "Alpha" in pages_block
    assert "Beta" in pages_block


def test_review_service_finding_message_uses_evidence_fields_when_claim_missing(
    test_project,
) -> None:
    service = ReviewService(test_project.paths)

    assert (
        service._finding_message(
            ReviewFinding(
                issue_type="needs-review",
                evidence_against="Conflicting evidence remains",
                verdict=Verdict.NEEDS_REVIEW,
            )
        )
        == "Conflicting evidence remains"
    )
    assert (
        service._finding_message(
            ReviewFinding(
                issue_type="needs-review",
                evidence_for="Supporting evidence remains",
                verdict=Verdict.NEEDS_REVIEW,
            )
        )
        == "Supporting evidence remains"
    )
    assert (
        service._finding_message(
            ReviewFinding(issue_type="needs-review", verdict=Verdict.NEEDS_REVIEW)
        )
        == "Needs Review"
    )


def test_review_service_persist_review_run_without_store_returns_none(
    test_project,
) -> None:
    service = ReviewService(test_project.paths)
    evidence_bundle = service._build_review_evidence_bundle([], [])

    run_id = service._persist_review_run(
        evidence_bundle=evidence_bundle,
        findings=[],
        model_id="fake-model",
        wall_time_ms=0,
        unresolved_disagreement=False,
    )

    assert run_id is None


def test_review_service_review_model_name_falls_back_to_provider_name(
    test_project,
) -> None:
    provider = SequencedReviewProvider([], model_name="fake-review-v1")
    service = ReviewService(test_project.paths, provider=provider)

    model_name = service._review_model_name([])

    assert model_name == "fake-review"


def test_review_service_adversarial_persists_run_and_emits_issue(test_project) -> None:
    test_project.write_file(
        "wiki/sources/alpha.md",
        "# Alpha\n\n## Timeline\n\nIn 2026 the traceability workflow is complete.\n",
    )
    test_project.write_file(
        "wiki/sources/beta.md",
        "# Beta\n\n## Timeline\n\nIn 2026 the traceability workflow is incomplete.\n",
    )
    provider = SequencedReviewProvider(
        [
            "CLAIM|The pages disagree about whether the traceability workflow is complete|wiki/sources/alpha.md, wiki/sources/beta.md",
            "CRITIQUE|contradiction|The pages disagree about whether the traceability workflow is complete|Beta states the workflow is incomplete.|wiki/sources/alpha.md, wiki/sources/beta.md",
            "FINDING|contradiction|contradictory|0.92|The pages disagree about whether the traceability workflow is complete|Alpha states the workflow is complete.|Beta states the workflow is incomplete.|wiki/sources/alpha.md, wiki/sources/beta.md",
        ]
    )
    run_store = RunStore(test_project.paths.graph_exports_dir / "review-adversarial.db")
    service = ReviewService(
        test_project.paths,
        provider=provider,
        run_store=run_store,
    )

    report = service.review(adversarial=True)

    contradiction_issues = [
        issue for issue in report.issues if issue.code == "contradiction"
    ]
    assert report.mode == "adversarial:fake-review-v1"
    assert report.run_id is not None
    assert len(report.findings) == 1
    assert report.findings[0].verdict == Verdict.CONTRADICTORY
    assert len(contradiction_issues) == 1
    assert contradiction_issues[0].severity == "error"
    record = run_store.get_run(report.run_id)
    assert record is not None
    assert record.command == "review"
    assert len(record.review_findings) == 1
    assert record.review_findings[0].issue_type == "contradiction"


def test_review_service_adversarial_keeps_consistent_findings_out_of_issue_output(
    test_project,
) -> None:
    test_project.write_file(
        "wiki/sources/alpha.md",
        "# Alpha\n\n## Timeline\n\nIn 2026 the compiler stores source hashes.\n",
    )
    test_project.write_file(
        "wiki/sources/beta.md",
        "# Beta\n\n## Timeline\n\nIn 2026 the vault export copies markdown pages.\n",
    )
    provider = SequencedReviewProvider(
        [
            "CLAIM|The pages both describe maintenance workflow steps|wiki/sources/alpha.md, wiki/sources/beta.md",
            "NO_CRITIQUES",
            "FINDING|consistent|consistent|0.88|The pages both describe maintenance workflow steps|Both pages describe maintenance steps||wiki/sources/alpha.md, wiki/sources/beta.md",
        ]
    )
    service = ReviewService(test_project.paths, provider=provider)

    report = service.review(adversarial=True)

    assert report.mode == "adversarial:fake-review-v1"
    assert len(report.findings) == 1
    assert report.findings[0].verdict == Verdict.CONSISTENT
    assert report.issues == []


def test_review_service_adversarial_raises_when_all_pairs_fail(test_project) -> None:
    test_project.write_file(
        "wiki/sources/alpha.md",
        "# Alpha\n\n## Timeline\n\nIn 2026 the compiler stores source hashes.\n",
    )
    test_project.write_file(
        "wiki/sources/beta.md",
        "# Beta\n\n## Timeline\n\nIn 2026 the vault export copies markdown pages.\n",
    )
    provider = SequencedReviewProvider([RuntimeError("extractor failed")])
    service = ReviewService(test_project.paths, provider=provider)

    with pytest.raises(ProviderExecutionError, match="extractor failed"):
        service.review(adversarial=True)


def test_review_service_adversarial_without_candidate_pairs_returns_empty_findings(
    test_project,
) -> None:
    test_project.write_file(
        "wiki/sources/alpha.md",
        "# Alpha\n\nGraph storage and adjacency lists.\n",
    )
    test_project.write_file(
        "wiki/sources/beta.md",
        "# Beta\n\nTransformer attention and gradient descent.\n",
    )
    provider = SequencedReviewProvider([])
    service = ReviewService(test_project.paths, provider=provider)

    report = service.review(adversarial=True)

    assert report.mode == "adversarial:fake-review"
    assert report.findings == []
    assert report.issues == []
