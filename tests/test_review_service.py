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
