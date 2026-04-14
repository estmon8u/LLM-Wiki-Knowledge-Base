from __future__ import annotations

import json

from src.models.wiki_models import LintIssue, LintReport
from src.services.compile_service import _markdown_paragraphs, _strip_frontmatter
from src.services.lint_service import _split_frontmatter


def _ingest_source(test_project, relative_path: str, content: str):
    path = test_project.write_file(relative_path, content)
    return test_project.services["ingest"].ingest_path(path).source


def _compiled_page(title: str, body: str, *, summary: str = "Summary") -> str:
    return (
        "---\n"
        f"title: {title}\n"
        f"summary: {summary}\n"
        "source_id: source-1\n"
        "raw_path: raw/source.md\n"
        "source_hash: hash-1\n"
        "compiled_at: 2026-04-14T00:00:00Z\n"
        "---\n\n"
        f"# {title}\n\n"
        f"{body}\n"
    )


def test_markdown_paragraphs_strip_frontmatter_and_headings() -> None:
    contents = (
        "---\n"
        "title: Example\n"
        "---\n\n"
        "# Heading\n\n"
        "First paragraph line one.\n"
        "Still first paragraph.\n\n"
        "Second paragraph.\n"
    )

    assert _markdown_paragraphs(contents) == [
        "First paragraph line one. Still first paragraph.",
        "Second paragraph.",
    ]


def test_strip_frontmatter_returns_original_text_when_marker_is_invalid() -> None:
    invalid = "---\ntitle: Example\nNo closing marker"

    assert _strip_frontmatter(invalid) == invalid
    assert _strip_frontmatter("plain text") == "plain text"


def test_compile_service_compiles_source_pages_index_and_log(test_project) -> None:
    source = _ingest_source(
        test_project,
        "notes/doc.md",
        "# Source Doc\n\nParagraph one.\n\nParagraph two.\n",
    )

    result = test_project.services["compile"].compile()

    assert result.compiled_count == 1
    article_path = test_project.root / "wiki" / "sources" / "source-doc.md"
    assert article_path.exists()
    article_text = article_path.read_text(encoding="utf-8")
    assert "source_id" in article_text
    assert "## Summary" in article_text
    index_payload = json.loads(
        test_project.paths.wiki_index_file.read_text(encoding="utf-8")
    )
    assert index_payload["source_pages"][0]["slug"] == source.slug
    assert "compiled 1 source page(s)" in test_project.paths.wiki_log_file.read_text(
        encoding="utf-8"
    )


def test_compile_service_skips_unchanged_sources_and_force_rebuilds(
    test_project,
) -> None:
    _ingest_source(test_project, "notes/doc.md", "# Doc\n\nText\n")
    compile_service = test_project.services["compile"]

    first = compile_service.compile()
    second = compile_service.compile()
    forced = compile_service.compile(force=True)

    assert first.compiled_count == 1
    assert second.compiled_count == 0
    assert second.skipped_count == 1
    assert forced.compiled_count == 1


def test_compile_service_handles_empty_manifest(test_project) -> None:
    result = test_project.services["compile"].compile()

    assert result.compiled_count == 0
    assert test_project.paths.wiki_index_file.exists()
    assert (
        "No source pages compiled yet."
        in test_project.paths.wiki_index_markdown.read_text(encoding="utf-8")
    )


def test_compile_service_uses_fallback_summary_and_excerpt_when_no_paragraphs(
    test_project,
) -> None:
    source = _ingest_source(test_project, "notes/empty.md", "# Empty\n")

    test_project.services["compile"].compile()

    article_text = (
        test_project.root / "wiki" / "sources" / (source.slug + ".md")
    ).read_text(encoding="utf-8")
    assert "No summary available yet." in article_text
    assert "No excerpt available yet." in article_text


def test_split_frontmatter_parses_valid_yaml_and_invalid_text() -> None:
    frontmatter, content = _split_frontmatter("---\ntitle: Test\n---\nBody\n")

    assert frontmatter == {"title": "Test"}
    assert content == "Body\n"
    assert _split_frontmatter("No frontmatter") == (None, "No frontmatter")
    assert _split_frontmatter("---\ntitle: broken\nBody\n") == (
        None,
        "---\ntitle: broken\nBody\n",
    )


def test_lint_service_reports_stale_and_missing_compiled_pages(test_project) -> None:
    source = _ingest_source(test_project, "notes/stale.md", "# Stale\n\nBody\n")
    lint_service = test_project.services["lint"]

    stale_report = lint_service.lint()
    assert any(issue.code == "stale-source-page" for issue in stale_report.issues)

    test_project.services["compile"].compile()
    article_path = test_project.root / "wiki" / "sources" / (source.slug + ".md")
    article_path.unlink()
    missing_report = lint_service.lint()

    assert any(issue.code == "missing-compiled-page" for issue in missing_report.issues)


def test_lint_service_reports_missing_frontmatter_and_broken_links(
    test_project,
) -> None:
    test_project.write_file(
        "wiki/sources/manual.md",
        "# Manual\n\nBroken link [[Missing Target]]\n",
    )

    report = test_project.services["lint"].lint()
    codes = {issue.code for issue in report.issues}

    assert "missing-frontmatter" in codes
    assert "broken-link" in codes
    assert "orphan-page" in codes


def test_lint_service_reports_missing_fields_empty_summary_and_orphan_page(
    test_project,
) -> None:
    test_project.write_file(
        "wiki/sources/incomplete.md",
        "---\n" "title: Incomplete\n" "summary: ''\n" "---\n\n" "# Incomplete\n",
    )

    report = test_project.services["lint"].lint()
    codes = [issue.code for issue in report.issues]

    assert "missing-field" in codes
    assert "empty-summary" in codes
    assert "orphan-page" in codes


def test_lint_service_ignores_malformed_timestamp_markdown_links(
    test_project,
) -> None:
    test_project.write_file(
        "wiki/sources/target-page.md",
        "---\n"
        "title: Target Page\n"
        "summary: A real wiki target\n"
        "source_id: target-1\n"
        "source_hash: hash-1\n"
        "---\n\n"
        "# Target Page\n",
    )
    test_project.write_file(
        "wiki/sources/reference.md",
        "---\n"
        "title: Reference\n"
        "summary: References a valid page and a timestamp markdown link\n"
        "source_id: reference-1\n"
        "source_hash: hash-2\n"
        "---\n\n"
        "# Reference\n\n"
        "See [[00:00](http://example.com)] and [[target-page]].\n",
    )

    report = test_project.services["lint"].lint()

    assert not any(
        issue.code == "broken-link" and "00:00" in issue.message
        for issue in report.issues
    )
    assert not any(
        issue.code == "broken-link" and "target-page" in issue.message
        for issue in report.issues
    )


def test_lint_service_reports_markdown_link_and_empty_target_errors(
    test_project,
) -> None:
    test_project.write_file(
        "wiki/sources/bad.md",
        _compiled_page(
            "Bad Link Page",
            "See [missing](missing.md) and [empty]().",
        ),
    )

    report = test_project.services["lint"].lint()
    codes = [issue.code for issue in report.issues]

    assert "broken-markdown-link" in codes
    assert "empty-markdown-link" in codes


def test_lint_service_reports_broken_fragments_for_wiki_and_markdown_links(
    test_project,
) -> None:
    test_project.write_file(
        "wiki/sources/target-page.md",
        _compiled_page("Target Page", "## Present Section\n\nBody."),
    )
    test_project.write_file(
        "wiki/sources/reference.md",
        _compiled_page(
            "Reference Page",
            (
                "See [[target-page#Missing Section]] and "
                "[target](target-page.md#Also Missing)."
            ),
        ),
    )

    report = test_project.services["lint"].lint()
    fragment_issues = [
        issue for issue in report.issues if issue.code == "broken-fragment"
    ]

    assert len(fragment_issues) >= 2
    messages = " ".join(issue.message for issue in fragment_issues)
    assert "[[target-page#Missing Section]]" in messages
    assert "[target](target-page.md#Also Missing)" in messages


def test_lint_service_counts_markdown_links_for_orphan_detection(test_project) -> None:
    test_project.write_file(
        "wiki/sources/target-page.md",
        _compiled_page("Target Page", "Body."),
    )
    test_project.write_file(
        "wiki/sources/reference.md",
        _compiled_page("Reference Page", "See [target](target-page.md)."),
    )

    report = test_project.services["lint"].lint()

    assert not any(
        issue.code == "orphan-page" and issue.path == "wiki/sources/target-page.md"
        for issue in report.issues
    )


def test_lint_service_reports_heading_structure_and_duplicate_titles(
    test_project,
) -> None:
    test_project.write_file(
        "wiki/sources/alpha.md",
        _compiled_page(
            "Shared Title",
            "### Skipped Level\n\n# Another H1\n\n## Repeated\n\n## Repeated\n",
        ),
    )
    test_project.write_file(
        "wiki/concepts/beta.md",
        _compiled_page("Shared Title", "Body."),
    )

    report = test_project.services["lint"].lint()
    codes = [issue.code for issue in report.issues]

    assert "heading-level-skip" in codes
    assert "multiple-h1" in codes
    assert "duplicate-heading" in codes
    assert "duplicate-title" in codes


def test_lint_report_properties_count_issue_severities() -> None:
    report = LintReport(
        issues=[
            LintIssue("error", "broken-link", "a.md", "bad"),
            LintIssue("warning", "orphan-page", "b.md", "warn"),
            LintIssue("suggestion", "cross-link", "c.md", "suggest"),
        ]
    )

    assert report.error_count == 1
    assert report.warning_count == 1
    assert report.suggestion_count == 1
