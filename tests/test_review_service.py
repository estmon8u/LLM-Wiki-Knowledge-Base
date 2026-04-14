from __future__ import annotations


def test_review_service_finds_no_issues_on_empty_wiki(test_project) -> None:
    report = test_project.services["review"].review()

    assert report.issue_count == 0
    assert report.mode == "heuristic"


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
        mode="heuristic",
    )

    assert report.issue_count == 2
    assert report.mode == "heuristic"
