from __future__ import annotations

from src.providers.base import ProviderRequest, ProviderResponse, TextProvider
from src.services.search_service import _extract_snippet


def test_extract_snippet_uses_matching_window_and_fallback() -> None:
    text = "Alpha beta gamma delta epsilon zeta"

    assert "gamma" in _extract_snippet(text, ["gamma"])
    assert _extract_snippet(text, ["missing"]).startswith("Alpha")


def test_search_service_returns_ranked_results_and_limit(test_project) -> None:
    test_project.write_file("wiki/sources/first.md", "alpha alpha beta")
    test_project.write_file("wiki/sources/second.md", "alpha")
    test_project.write_file("wiki/index.md", "alpha alpha alpha")

    results = test_project.services["search"].search("alpha beta", limit=2)

    assert len(results) == 2
    assert results[0].score >= results[1].score
    assert results[0].path in {"wiki/index.md", "wiki/sources/first.md"}


def test_search_service_returns_empty_for_blank_query(test_project) -> None:
    assert test_project.services["search"].search("!!!") == []


def test_query_service_returns_fallback_when_no_matches(test_project) -> None:
    answer = test_project.services["query"].answer_question("What is missing?")

    assert answer.citations == []
    assert "No compiled wiki pages matched" in answer.answer


def test_query_service_returns_answer_with_citations(test_project) -> None:
    test_project.write_file("wiki/sources/citations.md", "traceability appears here")

    answer = test_project.services["query"].answer_question("traceability")

    assert answer.citations
    assert answer.citations[0].path == "wiki/sources/citations.md"
    assert "traceability appears here" in answer.answer


def test_query_service_save_answer_writes_analysis_page(test_project) -> None:
    test_project.write_file("wiki/sources/citations.md", "traceability appears here")
    answer = test_project.services["query"].answer_question("traceability")

    saved_path = test_project.services["query"].save_answer(
        "How does traceability work?", answer
    )

    assert saved_path.startswith("wiki/concepts/")
    assert saved_path.endswith(".md")
    full_path = test_project.root / saved_path
    assert full_path.exists()
    content = full_path.read_text(encoding="utf-8")
    assert "How does traceability work?" in content
    assert "type: analysis" in content
    assert "Citations" in content


def test_query_service_save_answer_uses_fallback_slug_for_empty_question(
    test_project,
) -> None:
    test_project.write_file("wiki/sources/citations.md", "traceability appears here")
    answer = test_project.services["query"].answer_question("traceability")

    saved_path = test_project.services["query"].save_answer("???", answer)

    assert saved_path.startswith("wiki/concepts/analysis-")
    assert (test_project.root / saved_path).exists()


# --- P1 boundary/negative tests ---


def test_search_special_characters_split_on_word_boundaries(test_project) -> None:
    test_project.write_file("wiki/sources/llm.md", "LLM-based research from 2026")

    results = test_project.services["search"].search("LLM-based (2026)")

    assert len(results) >= 1
    assert any("llm" in r.path for r in results)


def test_search_respects_limit_with_many_pages(test_project) -> None:
    for i in range(12):
        test_project.write_file(f"wiki/sources/page-{i}.md", f"alpha content {i}")

    results = test_project.services["search"].search("alpha", limit=3)

    assert len(results) == 3


def test_search_matches_terms_in_frontmatter(test_project) -> None:
    test_project.write_file(
        "wiki/sources/fm-match.md",
        "---\ntitle: traceability overview\n---\n\nBody without the keyword.\n",
    )

    results = test_project.services["search"].search("traceability")

    assert len(results) >= 1
    assert any("fm-match" in r.path for r in results)


def test_extract_snippet_term_at_position_zero() -> None:
    text = "gamma delta epsilon zeta theta"

    snippet = _extract_snippet(text, ["gamma"])

    assert snippet.startswith("gamma")


def test_save_answer_creates_parent_directory(test_project) -> None:
    import shutil

    concepts_dir = test_project.paths.wiki_concepts_dir
    if concepts_dir.exists():
        shutil.rmtree(concepts_dir)
    test_project.write_file("wiki/sources/citations.md", "traceability appears here")
    answer = test_project.services["query"].answer_question("traceability")

    saved_path = test_project.services["query"].save_answer(
        "What is traceability?", answer
    )

    assert (test_project.root / saved_path).exists()


def test_export_service_copies_all_markdown_files(test_project) -> None:
    test_project.write_file("wiki/sources/a.md", "A")
    test_project.write_file("wiki/index.md", "Index")

    result = test_project.services["export"].export_vault()

    assert set(result.exported_paths) == {
        "vault/obsidian/index.md",
        "vault/obsidian/sources/a.md",
    }


def test_status_service_counts_sources_compiled_pages_and_last_compile(
    test_project,
) -> None:
    source_path = test_project.write_file("notes/status.md", "# Status\n\nBody\n")
    test_project.services["ingest"].ingest_path(source_path)
    test_project.services["compile"].compile()
    test_project.write_file("wiki/concepts/topic.md", "# Topic\n")

    snapshot = test_project.services["status"].snapshot(initialized=True)

    assert snapshot.initialized is True
    assert snapshot.source_count == 1
    assert snapshot.compiled_source_count == 1
    assert snapshot.concept_page_count == 1
    assert snapshot.last_compile_at is not None


def test_diff_service_reports_new_source_before_compile(test_project) -> None:
    source_path = test_project.write_file("notes/diff.md", "# Diff\n\nBody\n")
    test_project.services["ingest"].ingest_path(source_path)

    report = test_project.services["diff"].diff()

    assert report.new_count == 1
    assert report.changed_count == 0
    assert report.up_to_date_count == 0
    assert report.entries[0].status == "new"
    assert report.entries[0].title == "Diff"


def test_diff_service_reports_up_to_date_after_compile(test_project) -> None:
    source_path = test_project.write_file("notes/diff.md", "# Diff\n\nBody\n")
    test_project.services["ingest"].ingest_path(source_path)
    test_project.services["compile"].compile()

    report = test_project.services["diff"].diff()

    assert report.new_count == 0
    assert report.changed_count == 0
    assert report.up_to_date_count == 1
    assert report.entries[0].status == "up_to_date"


def test_diff_service_reports_changed_after_source_modification(test_project) -> None:
    source_path = test_project.write_file("notes/diff.md", "# Diff\n\nBody\n")
    test_project.services["ingest"].ingest_path(source_path)
    test_project.services["compile"].compile()

    # Modify the normalized file to simulate a source change
    sources = test_project.services["manifest"].list_sources()
    record = sources[0]
    record.content_hash = "changed-hash"
    test_project.services["manifest"].save_source(record)

    report = test_project.services["diff"].diff()

    assert report.new_count == 0
    assert report.changed_count == 1
    assert report.up_to_date_count == 0
    assert report.entries[0].status == "changed"


def test_diff_service_handles_empty_manifest(test_project) -> None:
    report = test_project.services["diff"].diff()

    assert report.new_count == 0
    assert report.changed_count == 0
    assert report.up_to_date_count == 0
    assert report.entries == []


def test_provider_dataclasses_and_base_provider_behavior() -> None:
    request = ProviderRequest(prompt="hello", system_prompt="system")
    response = ProviderResponse(text="world", model_name="demo")

    assert request.prompt == "hello"
    assert response.model_name == "demo"

    provider = TextProvider()
    try:
        provider.generate(request)
    except NotImplementedError:
        pass
    else:
        raise AssertionError(
            "Expected TextProvider.generate to raise NotImplementedError"
        )
