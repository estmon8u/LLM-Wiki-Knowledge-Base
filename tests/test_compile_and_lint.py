from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models.wiki_models import LintIssue, LintReport
from src.providers.base import ProviderRequest, ProviderResponse, TextProvider
from src.services.compile_service import (
    SOURCE_PAGE_CONTRACT_VERSION,
    _abstract_paragraphs,
    _deterministic_summary,
    _discover_concept_pages,
    _is_content_paragraph,
    _is_link_only_inline,
    _is_weak_summary,
    _markdown_paragraphs,
    _normalize_newlines,
    _parse_frontmatter,
    _plain_text_fallback,
    _safe_int,
    _sorted_sources,
    _split_sentences,
    _strip_frontmatter,
    _truncate_with_boundary,
)
from src.services.update_service import UpdateOptions, UpdateService
from src.services.lint_service import (
    _split_frontmatter,
    _split_markdown_target,
    _strip_excerpt_section,
    _strip_fenced_code_blocks,
    _page_title,
    _extract_headings,
    _fragment_exists,
    _resolve_markdown_target,
    _PageState,
)


def _ingest_source(test_project, relative_path: str, content: str):
    path = test_project.write_file(relative_path, content)
    return test_project.services["ingest"].ingest_path(path).source


def _compiled_page(
    title: str,
    body: str,
    *,
    summary: str = "Summary",
    page_type: str | None = "source",
) -> str:
    type_block = f"type: {page_type}\n" if page_type else ""
    return (
        "---\n"
        f"title: {title}\n"
        f"summary: {summary}\n"
        f"{type_block}"
        "source_id: source-1\n"
        "raw_path: raw/source.md\n"
        "source_hash: hash-1\n"
        "compiled_at: 2026-04-14T00:00:00Z\n"
        "---\n\n"
        f"# {title}\n\n"
        f"{body}\n"
    )


class _FailOnSecondSummaryProvider(TextProvider):
    name = "fail-on-second"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("summary failure")
        return ProviderResponse(
            text="Stub summary of the document.", model_name="fail-on-second-v1"
        )


class _StableSummaryProvider(TextProvider):
    name = "stable-summary"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            text="Stub summary of the document.", model_name="stable-summary-v1"
        )


class _RecordingSummaryProvider(TextProvider):
    name = "recording-summary"

    def __init__(self) -> None:
        self.requests: list[ProviderRequest] = []

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(
            text="Stub summary of the document.", model_name="recording-summary-v1"
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
    assert (
        "update | 1 compiled, 0 skipped"
        in test_project.paths.wiki_log_file.read_text(encoding="utf-8")
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


def test_compile_service_plan_and_progress_callback_track_work(test_project) -> None:
    _ingest_source(test_project, "notes/alpha.md", "# Alpha\n\nBody\n")
    _ingest_source(test_project, "notes/beta.md", "# Beta\n\nBody\n")
    compile_service = test_project.services["compile"]

    plan = compile_service.plan()
    seen = []
    result = compile_service.compile(
        progress_callback=lambda source: seen.append(source.slug)
    )
    post_plan = compile_service.plan()

    assert plan.pending_count == 2
    assert plan.skipped_count == 0
    assert result.compiled_count == 2
    assert seen == ["alpha", "beta"]
    assert post_plan.pending_count == 0
    assert post_plan.skipped_count == 2


def test_compile_service_handles_empty_manifest(test_project) -> None:
    result = test_project.services["compile"].compile()

    assert result.compiled_count == 0
    assert test_project.paths.wiki_index_file.exists()
    assert (
        "No source pages compiled yet."
        in test_project.paths.wiki_index_markdown.read_text(encoding="utf-8")
    )


def test_compile_service_uses_provider_summary_for_minimal_content(
    test_project,
) -> None:
    source = _ingest_source(test_project, "notes/empty.md", "# Empty\n")

    test_project.services["compile"].compile()

    article_text = (
        test_project.root / "wiki" / "sources" / (source.slug + ".md")
    ).read_text(encoding="utf-8")
    assert "Stub summary of the document." in article_text
    assert "## Summary" in article_text
    assert "## Key Excerpt" in article_text


def test_compile_service_resume_requires_failed_run(test_project) -> None:
    with pytest.raises(ValueError, match="No failed compile run"):
        test_project.services["compile"].compile(resume=True)


def test_compile_service_plan_rejects_force_resume_combination(test_project) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        test_project.services["compile"].plan(force=True, resume=True)


def test_compile_service_records_failure_and_resumes_remaining_sources(
    test_project,
) -> None:
    _ingest_source(test_project, "notes/alpha.md", "# Alpha\n\nBody\n")
    _ingest_source(test_project, "notes/beta.md", "# Beta\n\nBody\n")
    compile_service = test_project.services["compile"]
    compile_service.provider = _FailOnSecondSummaryProvider()

    result = compile_service.compile()

    assert result.compiled_count == 2
    resume_record = compile_service.compile_run_store.resume_candidate()
    assert resume_record is None
    assert (test_project.root / "wiki/sources/alpha.md").exists()
    beta_page = (test_project.root / "wiki/sources/beta.md").read_text(encoding="utf-8")
    assert "## Summary" in beta_page
    assert "Body" in beta_page


def test_compile_service_resume_handles_post_loop_failure_with_no_pending_sources(
    monkeypatch, test_project
) -> None:
    _ingest_source(test_project, "notes/doc.md", "# Doc\n\nBody\n")
    compile_service = test_project.services["compile"]
    original_write_index = compile_service._write_index
    fail_once = {"value": True}

    def flaky_write_index(*args, **kwargs):
        if fail_once["value"]:
            fail_once["value"] = False
            raise OSError("index write failure")
        return original_write_index(*args, **kwargs)

    monkeypatch.setattr(compile_service, "_write_index", flaky_write_index)

    with pytest.raises(OSError, match="index write failure"):
        compile_service.compile()

    resume_record = compile_service.compile_run_store.resume_candidate()
    assert resume_record is not None
    assert resume_record.pending_source_slugs == []

    result = compile_service.compile(resume=True)

    assert result.compiled_count == 0
    assert result.resumed_from_run_id == resume_record.run_id
    assert test_project.paths.wiki_index_file.exists()
    assert compile_service.compile_run_store.resume_candidate() is None


def test_compile_service_append_log_adds_separator_for_non_newline_file(
    test_project,
) -> None:
    compile_service = test_project.services["compile"]
    test_project.paths.wiki_log_file.parent.mkdir(parents=True, exist_ok=True)
    test_project.paths.wiki_log_file.write_text("# Activity Log", encoding="utf-8")

    compile_service._append_log(1, 0, False, resumed=True)

    log_text = test_project.paths.wiki_log_file.read_text(encoding="utf-8")
    assert "## [" in log_text
    assert "update |" in log_text
    assert "(resume)" in log_text


def test_is_content_paragraph_rejects_link_heavy_navigation_text() -> None:
    assert _is_content_paragraph("[one](a) [two](b) [three](c) leftover text") is False


def test_parse_frontmatter_returns_empty_dict_for_invalid_inputs() -> None:
    assert _parse_frontmatter("plain text") == {}
    assert _parse_frontmatter("---\ntitle: broken") == {}
    assert _parse_frontmatter("---\ntitle: [broken\n---\nBody\n") == {}


def test_discover_concept_pages_handles_missing_and_unreadable_files(
    monkeypatch, test_project
) -> None:
    concept_dir = test_project.paths.wiki_concepts_dir
    for path in concept_dir.glob("*.md"):
        path.unlink()
    concept_dir.rmdir()

    assert _discover_concept_pages(test_project.paths) == []

    concept_dir.mkdir(parents=True, exist_ok=True)
    readable = concept_dir / "good.md"
    unreadable = concept_dir / "bad.md"
    readable.write_text("---\ntitle: Good Concept\n---\nBody\n", encoding="utf-8")
    unreadable.write_text("content", encoding="utf-8")
    original_read_text = Path.read_text

    def flaky_read_text(self, *args, **kwargs):
        if self == unreadable:
            raise OSError("locked")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", flaky_read_text)

    assert _discover_concept_pages(test_project.paths) == [
        {
            "title": "Good Concept",
            "slug": "good",
            "path": "wiki/concepts/good.md",
        }
    ]


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


def test_lint_service_reports_invalid_frontmatter_types(test_project) -> None:
    test_project.write_file(
        "wiki/sources/bad-types.md",
        "---\n"
        "title: 123\n"
        "summary: Valid string\n"
        "source_id: id-1\n"
        "raw_path: raw/file.md\n"
        "source_hash: hash-1\n"
        "compiled_at: 2026-04-14T00:00:00Z\n"
        "tags: not-a-list\n"
        "---\n\n"
        "# Bad Types\n\nBody.\n",
    )

    report = test_project.services["lint"].lint()
    type_issues = [i for i in report.issues if i.code == "invalid-field-type"]

    assert len(type_issues) >= 2
    fields = " ".join(i.message for i in type_issues)
    assert "title" in fields
    assert "tags" in fields


def test_lint_service_reports_invalid_date_format(test_project) -> None:
    test_project.write_file(
        "wiki/sources/bad-date.md",
        "---\n"
        "title: Bad Date\n"
        "summary: Summary\n"
        "source_id: id-1\n"
        "raw_path: raw/file.md\n"
        "source_hash: hash-1\n"
        "compiled_at: not-a-date\n"
        "ingested_at: also bad\n"
        "---\n\n"
        "# Bad Date\n\nBody.\n",
    )

    report = test_project.services["lint"].lint()
    date_issues = [i for i in report.issues if i.code == "invalid-date-format"]

    assert len(date_issues) >= 2
    fields = " ".join(i.message for i in date_issues)
    assert "compiled_at" in fields
    assert "ingested_at" in fields


def test_lint_service_reports_empty_page(test_project) -> None:
    test_project.write_file(
        "wiki/sources/empty-body.md",
        _compiled_page("Empty Body", ""),
    )
    test_project.write_file(
        "wiki/sources/headings-only.md",
        "---\n"
        "title: Headings Only\n"
        "summary: Summary\n"
        "source_id: id-1\n"
        "raw_path: raw/file.md\n"
        "source_hash: hash-1\n"
        "compiled_at: 2026-04-14T00:00:00Z\n"
        "---\n\n"
        "# Headings Only\n\n"
        "## Sub Heading\n",
    )

    report = test_project.services["lint"].lint()
    empty_issues = [i for i in report.issues if i.code == "empty-page"]

    assert len(empty_issues) >= 2
    paths = {i.path for i in empty_issues}
    assert "wiki/sources/empty-body.md" in paths
    assert "wiki/sources/headings-only.md" in paths


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


# --- P0 coverage tests: lint helpers and uncovered branches ---


def test_strip_fenced_code_blocks_removes_backtick_and_tilde_fences() -> None:
    text = (
        "Before\n"
        "```python\n"
        "[[hidden-link]]\n"
        "```\n"
        "Between\n"
        "~~~\n"
        "# fake heading inside tilde fence\n"
        "~~~\n"
        "After\n"
    )

    result = _strip_fenced_code_blocks(text)

    assert "[[hidden-link]]" not in result
    assert "fake heading inside tilde fence" not in result
    assert "Before" in result
    assert "Between" in result
    assert "After" in result


def test_split_markdown_target_handles_angle_brackets_and_space_title() -> None:
    dest, frag = _split_markdown_target("<page.md#section>")
    assert dest == "page.md"
    assert frag == "section"

    dest2, frag2 = _split_markdown_target('page.md "link title"')
    assert dest2 == "page.md"
    assert frag2 == ""


def test_page_title_falls_back_to_file_stem() -> None:
    title = _page_title(Path("wiki/sources/my-page.md"), None, [])
    assert title == "my-page"

    title_with_empty_fm = _page_title(Path("wiki/sources/other.md"), {"title": ""}, [])
    assert title_with_empty_fm != ""


def test_lint_ignores_links_inside_fenced_code_blocks(test_project) -> None:
    test_project.write_file(
        "wiki/sources/code-page.md",
        _compiled_page(
            "Code Page",
            (
                "Body text.\n\n"
                "```markdown\n"
                "[[nonexistent-page]]\n"
                "[broken](missing.md)\n"
                "```\n"
            ),
        ),
    )

    report = test_project.services["lint"].lint()

    assert not any(
        issue.code in ("broken-link", "broken-markdown-link")
        and "code-page" in issue.path
        for issue in report.issues
    )


def test_lint_ignores_image_links() -> None:
    from src.services.lint_service import MARKDOWN_LINK_PATTERN

    image = "![alt text](image.png)"
    assert MARKDOWN_LINK_PATTERN.search(image) is None

    link = "[text](page.md)"
    assert MARKDOWN_LINK_PATTERN.search(link) is not None


def test_lint_empty_page_skips_non_source_directories(test_project) -> None:
    test_project.write_file(
        "wiki/index.md",
        "---\ntitle: Index\n---\n\n# Index\n",
    )

    report = test_project.services["lint"].lint()

    assert not any(
        issue.code == "empty-page" and issue.path == "wiki/index.md"
        for issue in report.issues
    )


def test_lint_fragment_only_link_checks_current_page_anchors(test_project) -> None:
    test_project.write_file(
        "wiki/sources/self-ref.md",
        _compiled_page(
            "Self Ref",
            (
                "## Present Section\n\n"
                "See [above](#present-section) and [missing](#no-such-heading).\n"
            ),
        ),
    )

    report = test_project.services["lint"].lint()
    fragment_issues = [
        i for i in report.issues if i.code == "broken-fragment" and "self-ref" in i.path
    ]

    assert len(fragment_issues) == 1
    assert "no-such-heading" in fragment_issues[0].message


# --- P0 remaining: defensive branches and edge-case helpers ---


def test_extract_headings_skips_empty_normalized_titles() -> None:
    content = "# Valid Heading\n\n#  \n\n## Another\n"

    headings = _extract_headings(content)

    assert len(headings) == 2
    assert headings[0] == (1, "Valid Heading")
    assert headings[1] == (2, "Another")


def test_fragment_exists_returns_false_for_empty_fragment() -> None:
    state = _PageState(
        file_path=Path("wiki/sources/test.md"),
        relative_path="wiki/sources/test.md",
        frontmatter=None,
        content="",
        analysis_text="",
        headings=[],
        anchors={"some-heading"},
        page_title="test",
    )

    assert _fragment_exists(state, "") is False
    assert _fragment_exists(state, "   ") is False
    assert _fragment_exists(state, "some-heading") is True


def test_resolve_markdown_target_handles_absolute_path() -> None:
    current = Path("wiki/sources/page.md")
    result = _resolve_markdown_target(current, "/absolute/path.md")

    assert result.is_absolute()
    assert result.name == "path.md"


def test_lint_heading_structure_returns_empty_for_no_headings(test_project) -> None:
    test_project.write_file(
        "wiki/sources/no-headings.md",
        "---\ntitle: No Headings\nsummary: Summary\nsource_id: id\n"
        "raw_path: raw/f.md\nsource_hash: h\ncompiled_at: 2026-04-14T00:00:00Z\n"
        "---\n\nJust body text with no headings at all.\n",
    )

    report = test_project.services["lint"].lint()

    assert not any(
        issue.code in ("heading-level-skip", "multiple-h1", "duplicate-heading")
        and "no-headings" in issue.path
        for issue in report.issues
    )


# --- P1 boundary/negative tests ---


def test_lint_valid_frontmatter_produces_no_type_issues(test_project) -> None:
    test_project.write_file(
        "wiki/sources/valid-types.md",
        "---\n"
        "title: Valid Page\n"
        "summary: A valid summary\n"
        "source_id: id-1\n"
        "raw_path: raw/file.md\n"
        "source_hash: hash-1\n"
        "origin: origin.md\n"
        "normalized_path: raw/normalized/file.md\n"
        "compiled_at: 2026-04-14T00:00:00Z\n"
        "ingested_at: 2026-04-14T00:00:00Z\n"
        "tags:\n"
        "  - research\n"
        "  - capstone\n"
        "---\n\n"
        "# Valid Page\n\nBody content.\n",
    )

    report = test_project.services["lint"].lint()
    type_issues = [
        i
        for i in report.issues
        if i.code in ("invalid-field-type", "invalid-date-format")
        and "valid-types" in i.path
    ]

    assert type_issues == []


def test_lint_yaml_datetime_coercion_passes_date_check(test_project) -> None:
    test_project.write_file(
        "wiki/sources/bare-date.md",
        "---\n"
        "title: Bare Date\n"
        "summary: Summary\n"
        "source_id: id-1\n"
        "raw_path: raw/file.md\n"
        "source_hash: hash-1\n"
        "compiled_at: 2026-04-14\n"
        "---\n\n"
        "# Bare Date\n\nBody.\n",
    )

    report = test_project.services["lint"].lint()
    date_issues = [
        i
        for i in report.issues
        if i.code == "invalid-date-format" and "bare-date" in i.path
    ]

    assert date_issues == []


def test_lint_case_insensitive_duplicate_heading(test_project) -> None:
    test_project.write_file(
        "wiki/sources/case-dup.md",
        _compiled_page("Case Dup", "## Setup\n\nFirst.\n\n## setup\n\nSecond.\n"),
    )

    report = test_project.services["lint"].lint()
    dup_issues = [
        i
        for i in report.issues
        if i.code == "duplicate-heading" and "case-dup" in i.path
    ]

    assert len(dup_issues) >= 1


def test_compile_requires_provider(test_project) -> None:
    from src.providers import ProviderConfigurationError

    test_project.services["compile"].provider = None
    _ingest_source(
        test_project,
        "notes/multi.md",
        "# Multi\n\nFirst para.\n\nSecond para.\n",
    )

    with pytest.raises(
        ProviderConfigurationError, match="requires a configured provider"
    ):
        test_project.services["compile"].compile()


def test_compile_custom_excerpt_character_limit(test_project) -> None:
    test_project.config["compile"]["excerpt_character_limit"] = 30
    long_para = "A" * 100
    _ingest_source(test_project, "notes/long.md", f"# Long\n\n{long_para}\n")

    test_project.services["compile"].compile()

    article = (test_project.root / "wiki" / "sources" / "long.md").read_text(
        encoding="utf-8"
    )
    excerpt_start = article.index("## Key Excerpt")
    excerpt_section = article[excerpt_start + len("## Key Excerpt") :]
    assert len(excerpt_section.strip()) <= 30


def test_compile_includes_canonical_file_line_when_normalized_path_set(
    test_project,
) -> None:
    _ingest_source(test_project, "notes/canon.md", "# Canon\n\nBody text.\n")

    test_project.services["compile"].compile()

    article = (test_project.root / "wiki" / "sources" / "canon.md").read_text(
        encoding="utf-8"
    )
    assert "Canonical file:" in article


def test_compile_omits_canonical_file_line_when_normalized_path_none(
    test_project,
) -> None:
    _ingest_source(test_project, "notes/nocanon.md", "# No Canon\n\nBody text.\n")
    sources = test_project.services["manifest"].list_sources()
    for source in sources:
        if source.slug == "no-canon":
            source.normalized_path = None
            test_project.services["manifest"].save_source(source)

    test_project.services["compile"].compile()

    article = (test_project.root / "wiki" / "sources" / "no-canon.md").read_text(
        encoding="utf-8"
    )
    assert "Canonical file:" not in article


def test_compile_appends_to_existing_log(test_project) -> None:
    test_project.paths.wiki_log_file.write_text(
        "# Activity Log\n\nPre-existing entry.\n", encoding="utf-8"
    )
    _ingest_source(test_project, "notes/log.md", "# Log\n\nBody.\n")

    test_project.services["compile"].compile()

    log_text = test_project.paths.wiki_log_file.read_text(encoding="utf-8")
    assert "Pre-existing entry." in log_text
    assert "update | 1 compiled, 0 skipped" in log_text


# --- P4 boundary tests ---


def test_lint_circular_wiki_links_terminates(test_project) -> None:
    test_project.write_file(
        "wiki/sources/page-a.md",
        _compiled_page("Page A", "See [[page-b]]."),
    )
    test_project.write_file(
        "wiki/sources/page-b.md",
        _compiled_page("Page B", "See [[page-a]]."),
    )

    report = test_project.services["lint"].lint()

    assert not any(i.code == "broken-link" for i in report.issues)
    assert not any(
        i.code == "orphan-page" and "page-a" in i.path for i in report.issues
    )


def test_lint_page_with_many_links_performs_reasonably(test_project) -> None:
    links = " ".join(f"[[target-{i}]]" for i in range(200))
    test_project.write_file(
        "wiki/sources/many-links.md",
        _compiled_page("Many Links", links),
    )
    for i in range(200):
        test_project.write_file(
            f"wiki/sources/target-{i}.md",
            _compiled_page(f"Target {i}", "Body."),
        )

    report = test_project.services["lint"].lint()

    assert report is not None


def test_frontmatter_yaml_injection_blocked_by_safe_load() -> None:
    import yaml
    from src.services.lint_service import _split_frontmatter

    malicious = (
        "---\ntitle: !!python/object/apply:os.system ['echo pwned']\n---\nBody\n"
    )

    with pytest.raises(yaml.constructor.ConstructorError):
        _split_frontmatter(malicious)


def test_markdown_paragraphs_preserves_content_before_first_heading() -> None:
    contents = (
        "Lead paragraph.\n\n"
        "# Real Title\n\n"
        "Actual content paragraph.\n\n"
        "Second paragraph.\n"
    )

    result = _markdown_paragraphs(contents)

    assert result == [
        "Lead paragraph.",
        "Actual content paragraph.",
        "Second paragraph.",
    ]


def test_markdown_paragraphs_falls_back_when_no_headings() -> None:
    contents = "First paragraph.\n\nSecond paragraph.\n"

    result = _markdown_paragraphs(contents)

    assert result == ["First paragraph.", "Second paragraph."]


def test_markdown_paragraphs_trims_leading_toc_boilerplate() -> None:
    contents = (
        "[![Site](banner.svg)](index.html)\n\n"
        "Small. Fast. Reliable. Choose any three.\n\n"
        "Search Documentation Search Changelog\n\n"
        "SQLite FTS5 Extension\n\n"
        "Table Of Contents\n\n"
        "[1. Overview](#overview)\n\n"
        "# 1. Overview\n\n"
        "FTS5 is an SQLite virtual table module.\n\n"
        "It supports full-text search.\n"
    )

    result = _markdown_paragraphs(contents)

    assert result == [
        "FTS5 is an SQLite virtual table module.",
        "It supports full-text search.",
    ]


def test_is_content_paragraph_rejects_image_only() -> None:
    assert not _is_content_paragraph("[![SQLite](img.svg)](index.html)")


def test_is_content_paragraph_rejects_nav_links() -> None:
    nav = "* [Home](index.html)* [About](about.html)* [Docs](docs.html)"
    assert not _is_content_paragraph(nav)


def test_is_content_paragraph_rejects_toc_links() -> None:
    toc = "[1. Overview](#overview) [2. Setup](#setup)"
    assert not _is_content_paragraph(toc)


def test_is_content_paragraph_rejects_single_word() -> None:
    assert not _is_content_paragraph("CREATE")
    assert not _is_content_paragraph("hide")


def test_is_content_paragraph_accepts_real_content() -> None:
    assert _is_content_paragraph("FTS5 is a virtual table module for full-text search.")
    assert _is_content_paragraph("Second paragraph.")


def test_compile_index_lists_concept_pages(test_project) -> None:
    _ingest_source(test_project, "notes/doc.md", "# Doc\n\nBody text.\n")
    test_project.services["compile"].compile()
    test_project.write_file(
        "wiki/concepts/my-topic.md",
        "---\ntitle: My Topic\ntype: analysis\n---\n\n# My Topic\n\nAnswer.\n",
    )

    test_project.services["compile"].compile(force=True)

    index_text = test_project.paths.wiki_index_markdown.read_text(encoding="utf-8")
    assert "[[my-topic]]" in index_text
    assert "No concept pages compiled yet." not in index_text
    index_json = json.loads(
        test_project.paths.wiki_index_file.read_text(encoding="utf-8")
    )
    slugs = [cp["slug"] for cp in index_json["concept_pages"]]
    assert "my-topic" in slugs


def test_update_refreshes_index_after_removing_stale_concept_page(test_project) -> None:
    _ingest_source(test_project, "notes/doc.md", "# Doc\n\nBody text.\n")
    test_project.config["provider"] = {"name": "openai"}
    test_project.write_file(
        "wiki/concepts/stale-topic.md",
        (
            "---\n"
            "title: Stale Topic\n"
            "type: concept\n"
            "generator: concept-service-v1\n"
            "---\n\n"
            "# Stale Topic\n\n"
            "Old generated concept.\n"
        ),
    )

    update_service = UpdateService(
        ingest_service=test_project.services["ingest"],
        compile_service=test_project.services["compile"],
        concept_service=test_project.services["concepts"],
        search_service=test_project.services["search"],
        config=test_project.config,
    )

    result = update_service.run(UpdateOptions())

    assert result.concept_result.removed_paths == ["wiki/concepts/stale-topic.md"]
    index_text = test_project.paths.wiki_index_markdown.read_text(encoding="utf-8")
    assert "[[stale-topic]]" not in index_text
    assert "No concept pages compiled yet." in index_text

    report = test_project.services["lint"].lint()
    assert not any(
        issue.code == "broken-link" and issue.path == "wiki/index.md"
        for issue in report.issues
    )


def test_markdown_paragraphs_skips_fenced_code_blocks() -> None:
    contents = (
        "Real paragraph before code.\n\n"
        "```python\n"
        "This is code and should not appear.\n"
        "```\n\n"
        "Real paragraph after code.\n"
    )

    result = _markdown_paragraphs(contents)

    assert result == ["Real paragraph before code.", "Real paragraph after code."]


def test_markdown_paragraphs_skips_tilde_fenced_code_blocks() -> None:
    contents = (
        "Before tilde block.\n\n"
        "~~~\n"
        "code inside\n"
        "~~~\n\n"
        "After tilde block.\n"
    )

    result = _markdown_paragraphs(contents)

    assert result == ["Before tilde block.", "After tilde block."]


def test_markdown_paragraphs_skips_html_comments() -> None:
    contents = (
        "Before comment.\n\n"
        "<!-- This is a multi-line\n"
        "HTML comment that spans lines -->\n\n"
        "After comment.\n"
    )

    result = _markdown_paragraphs(contents)

    assert result == ["Before comment.", "After comment."]


def test_markdown_paragraphs_skips_single_line_html_comment() -> None:
    contents = "Real content before comment.\n\n<!-- single line -->\n\nReal content after comment.\n"

    result = _markdown_paragraphs(contents)

    assert result == ["Real content before comment.", "Real content after comment."]


def test_markdown_paragraphs_skips_horizontal_rules() -> None:
    contents = "Before rule.\n\n---\n\nAfter rule.\n"

    result = _markdown_paragraphs(contents)

    assert result == ["Before rule.", "After rule."]


def test_normalize_newlines_strips_bom_and_crlf() -> None:
    assert _normalize_newlines("\ufeffHello\r\nWorld\rEnd") == "Hello\nWorld\nEnd"


def test_normalize_newlines_passes_clean_input() -> None:
    assert _normalize_newlines("clean\ninput") == "clean\ninput"


def test_plain_text_fallback_strips_markdown() -> None:
    contents = (
        "---\ntitle: Test\n---\n\n"
        "# Heading\n\n"
        "Some **bold** and [link](http://example.com) text.\n\n"
        "```\ncode block\n```\n"
    )

    result = _plain_text_fallback(contents)

    assert "bold" in result
    assert "link" in result
    assert "code block" not in result
    assert "#" not in result
    assert "**" not in result


def test_safe_int_returns_parsed_value() -> None:
    assert _safe_int(5, default=2, minimum=1) == 5
    assert _safe_int("10", default=2, minimum=1) == 10


def test_safe_int_returns_default_for_invalid() -> None:
    assert _safe_int(None, default=2, minimum=1) == 2
    assert _safe_int("abc", default=3, minimum=1) == 3


def test_safe_int_enforces_minimum() -> None:
    assert _safe_int(0, default=2, minimum=1) == 1
    assert _safe_int(-5, default=2, minimum=1) == 1


def test_sorted_sources_deterministic_order() -> None:
    from src.models.source_models import RawSourceRecord

    sources = [
        RawSourceRecord(
            source_id="id-2",
            title="Zebra",
            slug="zebra",
            raw_path="raw/zebra.md",
            origin="local",
            source_type="md",
            content_hash="h2",
            ingested_at="2026-01-01",
        ),
        RawSourceRecord(
            source_id="id-1",
            title="Alpha",
            slug="alpha",
            raw_path="raw/alpha.md",
            origin="local",
            source_type="md",
            content_hash="h1",
            ingested_at="2026-01-01",
        ),
    ]

    result = _sorted_sources(sources)

    assert [s.slug for s in result] == ["alpha", "zebra"]


def test_compile_raises_when_source_file_missing(test_project) -> None:
    source = _ingest_source(test_project, "notes/doc.md", "# Doc\n\nBody text.\n")
    canonical = source.normalized_path or source.raw_path
    canonical_full = test_project.root / canonical
    canonical_full.unlink()

    with pytest.raises(FileNotFoundError, match="does not exist"):
        test_project.services["compile"].compile()


def test_lint_concept_page_does_not_require_source_fields(test_project) -> None:
    test_project.write_file(
        "wiki/concepts/analysis.md",
        "---\ntitle: My Analysis\nsummary: An analysis page.\ntype: analysis\n"
        "question: How?\nsaved_at: '2026-04-20T12:00:00+00:00'\n"
        "citations:\n- wiki/sources/alpha.md#chunk-0\n"
        "insufficient_evidence: false\nclaim_count: 0\ncitation_count: 1\n"
        "---\n\n# My Analysis\n\nSome answer.\n",
    )

    report = test_project.services["lint"].lint()
    field_issues = [
        i for i in report.issues if i.code == "missing-field" and "analysis" in i.path
    ]

    assert field_issues == []


def test_lint_generated_concept_page_requires_concept_fields_not_source_fields(
    test_project,
) -> None:
    test_project.write_file(
        "wiki/concepts/concept.md",
        "---\n"
        "title: Retrieval and Question Answering\n"
        "summary: Generated concept page.\n"
        "type: concept\n"
        "generated_at: 2026-04-15T00:00:00Z\n"
        "source_pages:\n"
        "- wiki/sources/a.md\n"
        "topic_terms:\n"
        "- retrieval\n"
        "- question-answering\n"
        "---\n\n"
        "# Retrieval and Question Answering\n\nOverview.\n",
    )

    report = test_project.services["lint"].lint()
    field_issues = [
        i for i in report.issues if i.code == "missing-field" and "concept" in i.path
    ]

    assert field_issues == []


def test_lint_source_page_still_requires_all_fields(test_project) -> None:
    test_project.write_file(
        "wiki/sources/incomplete.md",
        "---\ntitle: Incomplete\nsummary: S\n---\n\n# Incomplete\n\nBody.\n",
    )

    report = test_project.services["lint"].lint()
    field_issues = [
        i for i in report.issues if i.code == "missing-field" and "incomplete" in i.path
    ]

    assert len(field_issues) >= 1


def test_strip_excerpt_section_removes_key_excerpt() -> None:
    content = (
        "# Title\n\n"
        "## Summary\n\nGood content.\n\n"
        "## Key Excerpt\n\nBroken [link](missing.html) in excerpt.\n"
    )

    result = _strip_excerpt_section(content)

    assert "## Summary" in result
    assert "## Key Excerpt" not in result
    assert "missing.html" not in result


def test_strip_excerpt_section_preserves_content_without_excerpt() -> None:
    content = "# Title\n\n## Summary\n\nBody.\n"

    assert _strip_excerpt_section(content) == content


def test_lint_ignores_broken_links_inside_key_excerpt(test_project) -> None:
    test_project.write_file(
        "wiki/sources/excerpt-page.md",
        "---\n"
        "title: Excerpt Page\n"
        "summary: Summary\n"
        "source_id: id-1\n"
        "raw_path: raw/source.md\n"
        "source_hash: hash-1\n"
        "compiled_at: 2026-04-14T00:00:00Z\n"
        "---\n\n"
        "# Excerpt Page\n\n"
        "## Summary\n\nReal summary.\n\n"
        "## Key Excerpt\n\n"
        "[Home](index.html) [About](about.html)\n",
    )

    report = test_project.services["lint"].lint()
    link_issues = [
        i
        for i in report.issues
        if i.code == "broken-markdown-link" and "excerpt-page" in i.path
    ]

    assert link_issues == []


# --- Provider-backed compile tests ---


def test_compile_provider_empty_content_returns_no_content_message(
    test_project,
) -> None:
    from src.services.compile_service import CompileService

    service = test_project.services["compile"]
    summary = service._extract_summary("")

    assert summary == "No content available for summarization."


def test_compile_provider_structured_summary_fields(test_project) -> None:
    from src.providers.base import ProviderRequest, ProviderResponse, TextProvider

    class StructuredSummaryProvider(TextProvider):
        name = "structured-summary"

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            return ProviderResponse(
                text=json.dumps(
                    {
                        "summary": "The document explains provenance traceability through compiled pages.",
                        "key_points": ["Compiled pages preserve provenance."],
                        "open_questions": ["How should stale pages be reviewed?"],
                        "title_suggestion": "Traceability Overview",
                    }
                ),
                model_name="structured-summary-v1",
            )

    service = test_project.services["compile"]
    service.provider = StructuredSummaryProvider()

    result = service._extract_summary_result(
        "# Traceability\n\nCompiled pages preserve provenance."
    )

    assert (
        result.summary
        == "The document explains provenance traceability through compiled pages."
    )
    assert result.key_points == ["Compiled pages preserve provenance."]
    assert result.open_questions == ["How should stale pages be reviewed?"]
    assert result.title_suggestion == "Traceability Overview"


def test_compile_provider_empty_response_falls_back(test_project) -> None:
    from src.providers.base import ProviderRequest, ProviderResponse, TextProvider

    class EmptyProvider(TextProvider):
        name = "empty"

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            return ProviderResponse(text="", model_name="empty-v1")

    test_project.services["compile"].provider = EmptyProvider()
    _ingest_source(test_project, "notes/doc.md", "# Doc\n\nSome content.\n")

    test_project.services["compile"].compile()

    article = (test_project.root / "wiki" / "sources" / "doc.md").read_text(
        encoding="utf-8"
    )
    assert "Some content." in article
    assert "No summary available yet." not in article


def test_compile_provider_error_falls_back_to_deterministic_summary(
    test_project,
) -> None:
    from src.providers.base import ProviderRequest, ProviderResponse, TextProvider

    class ErrorProvider(TextProvider):
        name = "error"

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            raise RuntimeError("API timeout")

    test_project.services["compile"].provider = ErrorProvider()
    _ingest_source(test_project, "notes/doc.md", "# Doc\n\nSome content.\n")

    test_project.services["compile"].compile()

    article = (test_project.root / "wiki" / "sources" / "doc.md").read_text(
        encoding="utf-8"
    )
    assert "Some content." in article


def test_compile_prompt_echo_summary_falls_back_to_deterministic_summary(
    test_project,
) -> None:
    from src.providers.base import ProviderRequest, ProviderResponse, TextProvider

    class EchoProvider(TextProvider):
        name = "echo"

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            return ProviderResponse(
                text=(
                    "source_id: source_1\n"
                    "raw_path: unknown\n"
                    "content_hash: unknown\n"
                    "summary: The document argues that dense retrieval improves QA."
                ),
                model_name="echo-v1",
            )

    test_project.services["compile"].provider = EchoProvider()
    _ingest_source(
        test_project,
        "notes/doc.md",
        "# Doc\n\n## Abstract\n\nSome content about retrieval.\n",
    )

    test_project.services["compile"].compile()

    article = (test_project.root / "wiki" / "sources" / "doc.md").read_text(
        encoding="utf-8"
    )
    assert "source_id: source_1" not in article
    assert "Some content about retrieval." in article


def test_compile_unavailable_provider_raises_configuration_error(
    test_project,
) -> None:
    from src.providers import ProviderConfigurationError, UnavailableProvider

    test_project.services["compile"].provider = UnavailableProvider(
        "No API key set", provider_name="test"
    )
    _ingest_source(test_project, "notes/doc.md", "# Doc\n\nBody.\n")

    with pytest.raises(ProviderConfigurationError, match="No API key set"):
        test_project.services["compile"].compile()


def test_abstract_paragraphs_extracts_from_abstract_heading() -> None:
    contents = (
        "## Title\n\n"
        "Author names and affiliations\n\n"
        "## Abstract\n\n"
        "This paper proposes a new method.\n\n"
        "It outperforms baselines.\n\n"
        "## 1. Introduction\n\n"
        "Background here.\n"
    )

    result = _abstract_paragraphs(contents)

    assert result == [
        "This paper proposes a new method.",
        "It outperforms baselines.",
    ]


def test_abstract_paragraphs_returns_empty_without_abstract_heading() -> None:
    contents = "## Title\n\nBody text.\n\n## Methods\n\nMore text.\n"

    assert _abstract_paragraphs(contents) == []


def test_compile_excerpt_prefers_abstract_section(test_project) -> None:
    _ingest_source(
        test_project,
        "notes/paper.md",
        "## Paper Title\n\n"
        "John Doe, University\n\n"
        "## Abstract\n\n"
        "This paper proposes a new retrieval method.\n\n"
        "## 1. Introduction\n\n"
        "Background.\n",
    )

    test_project.services["compile"].compile()

    article = (test_project.root / "wiki" / "sources" / "paper-title.md").read_text(
        encoding="utf-8"
    )
    excerpt_start = article.index("## Key Excerpt")
    excerpt_section = article[excerpt_start + len("## Key Excerpt") :]
    assert "This paper proposes a new retrieval method." in excerpt_section
    assert "John Doe" not in excerpt_section


def test_abstract_paragraphs_extracts_without_following_heading() -> None:
    contents = (
        "## Abstract\n\n"
        "Only abstract content here.\n\n"
        "Second paragraph of abstract.\n"
    )

    result = _abstract_paragraphs(contents)

    assert result == [
        "Only abstract content here.",
        "Second paragraph of abstract.",
    ]


def test_abstract_paragraphs_strips_frontmatter() -> None:
    contents = (
        "---\ntitle: Paper\n---\n\n" "## Abstract\n\n" "Content after frontmatter.\n"
    )

    result = _abstract_paragraphs(contents)

    assert result == ["Content after frontmatter."]


# --- Phase 7: schema/index centrality tests ---


def test_compiled_source_page_has_type_source(test_project) -> None:
    _ingest_source(test_project, "notes/typed.md", "# Typed\n\nBody.\n")

    test_project.services["compile"].compile()

    page_text = (test_project.root / "wiki/sources/typed.md").read_text(
        encoding="utf-8"
    )
    assert "type: source" in page_text


def test_lint_warns_on_old_source_page_missing_type(test_project) -> None:
    source = _ingest_source(test_project, "notes/old.md", "# Old\n\nBody.\n")
    source.compiled_at = "2026-04-21T00:00:00Z"
    source.compiled_from_hash = source.content_hash
    source.metadata = {}
    test_project.services["manifest"].save_source(source)
    test_project.write_file(
        "wiki/sources/old.md",
        _compiled_page("Old", "No type field.", page_type=None),
    )

    report = test_project.services["lint"].lint()
    codes = [i.code for i in report.issues]

    assert "missing-type" in codes
    issue = next(i for i in report.issues if i.code == "missing-type")
    assert issue.severity == "warning"
    assert "kb update --force" in issue.message


def test_lint_errors_when_current_source_page_loses_type(test_project) -> None:
    _ingest_source(test_project, "notes/current.md", "# Current\n\nBody.\n")

    test_project.services["compile"].compile()

    page_path = test_project.root / "wiki/sources/current.md"
    page_text = page_path.read_text(encoding="utf-8").replace("type: source\n", "", 1)
    page_path.write_text(page_text, encoding="utf-8")

    source = test_project.services["manifest"].list_sources()[0]
    assert (
        source.metadata["source_page_contract_version"] == SOURCE_PAGE_CONTRACT_VERSION
    )

    report = test_project.services["lint"].lint()
    issue = next(i for i in report.issues if i.code == "missing-type")

    assert issue.severity == "error"
    assert "required frontmatter field: type" in issue.message


def test_lint_no_missing_type_warning_when_type_present(test_project) -> None:
    test_project.write_file(
        "wiki/sources/new.md",
        "---\ntitle: New\nsummary: S\ntype: source\nsource_id: s\n"
        "raw_path: raw/s.md\nsource_hash: h\ncompiled_at: 2026-04-21\n---\n\n# New\n\nBody.\n",
    )

    report = test_project.services["lint"].lint()
    missing_type = [i for i in report.issues if i.code == "missing-type"]

    assert missing_type == []


def test_index_includes_analysis_pages_section(test_project) -> None:
    test_project.write_file(
        "wiki/analysis/my-question.md",
        "---\ntitle: My Question\ntype: analysis\n---\n\n# My Question\n",
    )

    test_project.services["compile"]._write_index([])

    index_text = test_project.paths.wiki_index_markdown.read_text(encoding="utf-8")
    assert "## Analysis Pages" in index_text
    assert "[[my-question]]" in index_text

    index_json = json.loads(
        test_project.paths.wiki_index_file.read_text(encoding="utf-8")
    )
    assert "analysis_pages" in index_json
    assert any(e["slug"] == "my-question" for e in index_json["analysis_pages"])


def test_index_shows_empty_analysis_section_when_none(test_project) -> None:
    test_project.services["compile"]._write_index([])

    index_text = test_project.paths.wiki_index_markdown.read_text(encoding="utf-8")
    assert "## Analysis Pages" in index_text
    assert "No analysis pages saved yet." in index_text


def test_compile_prompt_includes_schema_excerpt(test_project) -> None:
    from src.services.config_service import DEFAULT_SCHEMA

    provider = _RecordingSummaryProvider()
    compile_service = test_project.services["compile"]
    compile_service.provider = provider
    compile_service.schema_text = DEFAULT_SCHEMA
    _ingest_source(test_project, "notes/schema.md", "# Schema\n\nBody.\n")

    compile_service.compile()

    assert provider.requests
    system_prompt = provider.requests[0].system_prompt or ""
    assert "## Source Pages" in system_prompt
    assert "Create one source page" in system_prompt
    assert "## Query Behavior" not in system_prompt


def test_log_entry_uses_heading_format(test_project) -> None:
    compile_service = test_project.services["compile"]
    compile_service._append_log(3, 7, True)

    log_text = test_project.paths.wiki_log_file.read_text(encoding="utf-8")
    # Entry should be grep-parseable heading format
    assert "## [" in log_text
    assert "update | 3 compiled, 7 skipped (force)" in log_text


def test_log_entry_no_flags_when_not_force_or_resume(test_project) -> None:
    compile_service = test_project.services["compile"]
    compile_service._append_log(2, 0, False)

    log_text = test_project.paths.wiki_log_file.read_text(encoding="utf-8")
    assert "update | 2 compiled, 0 skipped\n" in log_text
    assert "(force)" not in log_text
    assert "(resume)" not in log_text


def test_truncate_with_boundary_fits_entirely() -> None:
    text = "Short text."
    assert _truncate_with_boundary(text, 100, add_ellipsis=True) == text


def test_truncate_with_boundary_cuts_at_sentence() -> None:
    text = "First sentence. Second sentence. Third sentence that is longer."
    result = _truncate_with_boundary(text, 35, add_ellipsis=True)
    assert result.endswith("Second sentence....")


def test_truncate_with_boundary_cuts_at_word_when_no_sentence() -> None:
    text = "one two three four five six seven eight nine ten eleven twelve"
    result = _truncate_with_boundary(text, 30, add_ellipsis=True)
    assert not result.rstrip(".").endswith("elev")
    assert "..." in result


def test_truncate_with_boundary_empty_input() -> None:
    assert _truncate_with_boundary("", 100, add_ellipsis=True) == ""


def test_is_weak_summary_detects_placeholders() -> None:
    assert _is_weak_summary("") is True
    assert _is_weak_summary("No summary available yet.") is True
    assert _is_weak_summary("ok") is True
    assert _is_weak_summary("The paper presents a novel approach.") is False


def test_deterministic_summary_uses_abstract() -> None:
    text = "# Paper\n\n## Abstract\n\nFirst sentence. Second sentence. Third.\n"
    result = _deterministic_summary(text)
    assert "First sentence." in result
    assert "Second sentence." in result


def test_deterministic_summary_falls_back_to_paragraphs() -> None:
    text = "# Paper\n\nSome first paragraph. With details.\n\nSecond paragraph.\n"
    result = _deterministic_summary(text)
    assert "Some first paragraph" in result


# ---------------------------------------------------------------------------
# markdown-it-py based helpers
# ---------------------------------------------------------------------------


def test_plain_text_fallback_strips_markdown_via_ast() -> None:
    md = "# Heading\n\nSome **bold** text with [a link](http://example.com).\n"
    result = _plain_text_fallback(md)
    assert "bold" in result
    assert "a link" in result
    assert "**" not in result
    assert "http" not in result
    assert "#" not in result


def test_plain_text_fallback_skips_fenced_code() -> None:
    md = "Hello.\n\n```python\ncode = True\n```\n\nWorld.\n"
    result = _plain_text_fallback(md)
    assert "Hello." in result
    assert "World." in result
    assert "code = True" not in result


def test_plain_text_fallback_skips_html_comments() -> None:
    md = "Before.\n\n<!-- hidden comment -->\n\nAfter.\n"
    result = _plain_text_fallback(md)
    assert "Before." in result
    assert "After." in result
    assert "hidden" not in result


def test_plain_text_fallback_skips_images() -> None:
    md = "Text before.\n\n![alt](image.png)\n\nText after.\n"
    result = _plain_text_fallback(md)
    assert "Text before." in result
    assert "Text after." in result
    assert "alt" not in result or "image.png" not in result


def test_markdown_paragraphs_uses_ast() -> None:
    md = (
        "# Title\n\n"
        "First real paragraph.\n\n"
        "```python\ncode_block = 1\n```\n\n"
        "Second paragraph.\n"
    )
    result = _markdown_paragraphs(md)
    assert "First real paragraph." in result
    assert "Second paragraph." in result
    assert not any("code_block" in p for p in result)


def test_markdown_paragraphs_skips_link_only() -> None:
    md = "[Go here](http://example.com)\n\nReal content paragraph.\n"
    result = _markdown_paragraphs(md)
    assert "Real content paragraph." in result
    assert not any("Go here" in p for p in result)


def test_is_link_only_inline_detects_pure_links() -> None:
    from markdown_it import MarkdownIt

    parser = MarkdownIt()
    tokens = parser.parse("[Link](http://example.com)")
    inline = [t for t in tokens if t.type == "inline"][0]
    assert _is_link_only_inline(inline)


def test_is_link_only_inline_rejects_mixed_content() -> None:
    from markdown_it import MarkdownIt

    parser = MarkdownIt()
    tokens = parser.parse("Some text and [Link](http://example.com).")
    inline = [t for t in tokens if t.type == "inline"][0]
    assert not _is_link_only_inline(inline)


# ---------------------------------------------------------------------------
# NLTK sentence splitting
# ---------------------------------------------------------------------------


def test_split_sentences_with_nltk() -> None:
    text = "First sentence. Second sentence. Third one."
    result = _split_sentences(text)
    assert len(result) >= 2
    assert result[0] == "First sentence."


def test_split_sentences_handles_abbreviations() -> None:
    text = "Dr. Smith found results. The test passed."
    result = _split_sentences(text)
    # NLTK should not split on "Dr."
    assert any("Dr." in s for s in result)


def test_split_sentences_empty_input() -> None:
    assert _split_sentences("") == []
    assert _split_sentences("   ") == []
