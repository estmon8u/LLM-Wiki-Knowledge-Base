from __future__ import annotations

import json
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

    # knowledge-base vs knowledgebase is hyphenation-only, suppressed
    variant_issues = [i for i in report.issues if i.code == "terminology-variant"]
    variant_messages = " ".join(i.message for i in variant_issues)
    assert "knowledge-base" not in variant_messages
    assert "knowledgebase" not in variant_messages


def test_review_service_suppresses_simple_inflection_variants(test_project) -> None:
    test_project.write_file(
        "wiki/sources/page-a.md",
        "The retriever indexes sources questions datasets benchmarks documents.",
    )
    test_project.write_file(
        "wiki/sources/page-b.md",
        "The retrievers index source question dataset benchmark document.",
    )

    report = test_project.services["review"].review()

    variant_messages = " ".join(
        issue.message for issue in report.issues if issue.code == "terminology-variant"
    )
    assert "retriever" not in variant_messages
    assert "source" not in variant_messages
    assert "question" not in variant_messages
    assert "dataset" not in variant_messages


def test_review_service_suppresses_false_positive_fuzzy_variants(
    test_project,
) -> None:
    test_project.write_file(
        "wiki/sources/page-a.md",
        "supervised accuracy billions open-domain knowledge-intensive passage-retrieval",
    )
    test_project.write_file(
        "wiki/sources/page-b.md",
        "unsupervised inaccuracy millions open-domain-qa "
        "knowledge-intensive-tasks dense-passage-retrieval",
    )

    report = test_project.services["review"].review()

    variant_messages = " ".join(
        issue.message for issue in report.issues if issue.code == "terminology-variant"
    )
    assert "unsupervised" not in variant_messages
    assert "inaccuracy" not in variant_messages
    assert "millions" not in variant_messages
    assert "open-domain-qa" not in variant_messages
    assert "knowledge-intensive-tasks" not in variant_messages
    assert "dense-passage-retrieval" not in variant_messages


def test_review_service_suppresses_near_spellings_with_different_roots(
    test_project,
) -> None:
    test_project.write_file(
        "wiki/sources/page-a.md",
        "contexts produce retrieval evidence",
    )
    test_project.write_file(
        "wiki/sources/page-b.md",
        "contents product retrieval evidence",
    )

    report = test_project.services["review"].review()

    variant_messages = " ".join(
        issue.message for issue in report.issues if issue.code == "terminology-variant"
    )
    assert "contexts" not in variant_messages
    assert "produce" not in variant_messages


def test_review_service_suppresses_hyphenation_only_variants(
    test_project,
) -> None:
    test_project.write_file(
        "wiki/sources/page-a.md",
        "pretraining fine-tuning reranking pre-trained",
    )
    test_project.write_file(
        "wiki/sources/page-b.md",
        "pre-training finetuning re-ranking pretrained",
    )

    report = test_project.services["review"].review()

    variant_messages = " ".join(
        issue.message for issue in report.issues if issue.code == "terminology-variant"
    )
    # Hyphenation-only variants should be suppressed via collapsed stemming
    assert "pretraining" not in variant_messages
    assert "pre-training" not in variant_messages
    assert "finetuning" not in variant_messages
    assert "fine-tuning" not in variant_messages
    assert "reranking" not in variant_messages
    assert "re-ranking" not in variant_messages
    assert "pre-trained" not in variant_messages
    assert "pretrained" not in variant_messages


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


def test_review_multi_agent_prefix_hyphenation_suppressed(test_project) -> None:
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

    # multi-agent vs multiagent is a prefix-hyphenation pair, suppressed
    variant_messages = " ".join(i.message for i in variant_issues)
    assert "multi-agent" not in variant_messages
    assert "multiagent" not in variant_messages


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


# ---------------------------------------------------------------------------
# rapidfuzz-based terminology variant detection
# ---------------------------------------------------------------------------


def test_review_detects_spelling_variants_with_rapidfuzz(test_project) -> None:
    """rapidfuzz should catch near-duplicate terms beyond hyphenation."""
    test_project.write_file(
        "wiki/sources/a.md",
        "---\ntitle: A\nsummary: s\ntype: source\nsource_id: a\n"
        "source_hash: h\nraw_path: raw/a.md\norigin: local\n"
        "compiled_at: t\ningested_at: t\ntags: []\n---\n"
        "retriever strategies work great\n",
    )
    test_project.write_file(
        "wiki/sources/b.md",
        "---\ntitle: B\nsummary: s\ntype: source\nsource_id: b\n"
        "source_hash: h\nraw_path: raw/b.md\norigin: local\n"
        "compiled_at: t\ningested_at: t\ntags: []\n---\n"
        "retriver strategies work great\n",
    )

    report = test_project.services["review"].review()
    variant_issues = [i for i in report.issues if i.code == "terminology-variant"]
    assert len(variant_issues) >= 1
    msgs = " ".join(i.message for i in variant_issues)
    assert "retriever" in msgs or "retriver" in msgs


def test_review_variant_ignores_short_terms(test_project) -> None:
    """Terms shorter than 4 characters should not trigger variant detection."""
    test_project.write_file(
        "wiki/sources/a.md",
        "---\ntitle: A\nsummary: s\ntype: source\nsource_id: a\n"
        "source_hash: h\nraw_path: raw/a.md\norigin: local\n"
        "compiled_at: t\ningested_at: t\ntags: []\n---\n"
        "the cat sat\n",
    )
    test_project.write_file(
        "wiki/sources/b.md",
        "---\ntitle: B\nsummary: s\ntype: source\nsource_id: b\n"
        "source_hash: h\nraw_path: raw/b.md\norigin: local\n"
        "compiled_at: t\ningested_at: t\ntags: []\n---\n"
        "the car sat\n",
    )

    report = test_project.services["review"].review()
    variant_issues = [i for i in report.issues if i.code == "terminology-variant"]
    # "cat" vs "car" are 3-char tokens so should be ignored
    assert len(variant_issues) == 0


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


def test_review_service_parse_provider_issues_reads_structured_json() -> None:
    raw = json.dumps(
        {
            "issues": [
                {
                    "severity": "error",
                    "code": "contradiction",
                    "pages": ["a.md", "b.md"],
                    "message": "Pages disagree",
                },
                {
                    "severity": "warning",
                    "code": "term-drift",
                    "pages": ["d.md"],
                    "message": "Term mismatch",
                },
            ]
        }
    )

    issues = ReviewService._parse_provider_issues(raw)

    assert len(issues) == 2
    assert issues[0].severity == "error"
    assert issues[0].code == "contradiction"
    assert issues[0].pages == ["a.md", "b.md"]
    assert issues[1].severity == "warning"
    assert issues[1].code == "term-drift"


def test_review_service_provider_review_raises_on_provider_failure(
    test_project,
) -> None:
    test_project.write_file("wiki/sources/alpha.md", "# Alpha\n\nContent.\n")
    provider = SequencedReviewProvider([RuntimeError("provider crash")])
    service = ReviewService(test_project.paths, provider=provider)

    with pytest.raises(ProviderExecutionError, match="provider crash"):
        service._provider_review()


def test_provider_review_filters_truncation_and_uses_curated_excerpts(
    test_project,
) -> None:
    test_project.write_file(
        "wiki/sources/alpha.md",
        "---\ntitle: Alpha\ntype: source\n---\n\n"
        "# Alpha\n\n"
        "## Summary\n\nUseful summary.\n\n"
        "## Source Details\n\n- Raw file: `raw/sources/alpha.pdf`\n\n"
        "## Key Excerpt\n\nUseful evidence excerpt.\n",
    )
    provider = SequencedReviewProvider(
        [
            json.dumps(
                {
                    "issues": [
                        {
                            "severity": "error",
                            "code": "TRUNCATED_CONTENT",
                            "pages": ["wiki/sources/alpha.md"],
                            "message": "Content appears truncated mid-sentence.",
                        },
                        {
                            "severity": "warning",
                            "code": "real-issue",
                            "pages": ["wiki/sources/alpha.md"],
                            "message": "A real issue remains.",
                        },
                    ]
                }
            )
        ]
    )
    service = ReviewService(test_project.paths, provider=provider)

    issues, mode = service._provider_review()

    assert mode == "provider:fake-review-v1"
    assert [issue.code for issue in issues] == ["real-issue"]
    request = provider.requests[0]
    assert request.max_tokens == 4096
    assert request.reasoning_effort == "low"
    assert "Useful summary" in request.prompt
    assert "Useful evidence excerpt" in request.prompt
    assert "Source Details" not in request.prompt
