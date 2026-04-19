from __future__ import annotations

from pathlib import Path
import sqlite3
import threading
from unittest.mock import AsyncMock, patch

import pytest

from src.providers import ProviderConfigurationError, ProviderExecutionError
from src.providers.base import ProviderRequest, ProviderResponse, TextProvider
from src.schemas.claims import CandidateAnswer, Claim, EvidenceBundle, EvidenceItem
from src.services.query_service import QueryService
from src.services.search_service import _extract_snippet
from src.storage.run_store import RunStore


class SequencedProvider(TextProvider):
    name = "fake-provider"

    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self._lock = threading.Lock()
        self.requests: list[ProviderRequest] = []

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        with self._lock:
            self.requests.append(request)
            if not self._responses:
                raise AssertionError("No fake response remaining.")
            response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return ProviderResponse(text=str(response), model_name="fake-model")


def _provider_query_service(test_project, *responses: object) -> QueryService:
    return QueryService(
        test_project.paths,
        test_project.services["search"],
        provider=SequencedProvider(list(responses)),
    )


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


def test_extract_snippet_uses_matching_window_and_fallback() -> None:
    text = "Alpha beta gamma delta epsilon zeta"

    assert "gamma" in _extract_snippet(text, ["gamma"])
    assert _extract_snippet(text, ["missing"]).startswith("Alpha")


def test_extract_frontmatter_handles_invalid_yaml_and_non_mapping_payload() -> None:
    from src.services.search_service import _extract_frontmatter

    assert _extract_frontmatter("---\ntitle: [oops\n---\n") == {}
    assert _extract_frontmatter("---\n- item\n---\n") == {}


def test_frontmatter_value_and_text_helpers_cover_scalar_and_nested_values() -> None:
    from src.services.search_service import _frontmatter_text, _frontmatter_value

    frontmatter = {
        "title": " Example Title ",
        "count": 3,
        "flags": [True, "tag"],
        "nested": {"question": "How does it work?"},
    }

    assert _frontmatter_value(frontmatter, "title") == "Example Title"
    assert _frontmatter_value(frontmatter, "count") == ""
    flattened = _frontmatter_text(frontmatter)
    assert "Example Title" in flattened
    assert "3" in flattened
    assert "True" in flattened
    assert "tag" in flattened
    assert "How does it work?" in flattened


def test_page_title_falls_back_to_heading_and_filename() -> None:
    from src.services.search_service import _page_title

    heading_title = _page_title(
        Path("wiki/sources/example.md"),
        "# Heading Title\n\nBody\n",
        {},
    )
    filename_title = _page_title(
        Path("wiki/sources/file-name.md"),
        "Body without headings\n",
        {},
    )

    assert heading_title == "Heading Title"
    assert filename_title == "File Name"


def test_chunk_markdown_body_handles_blank_and_long_sections() -> None:
    from src.services.search_service import _chunk_markdown_body

    assert _chunk_markdown_body("---\ntitle: Empty\n---\n", "Empty") == []

    long_body = (
        "# Chunked\n\n"
        + ("alpha beta gamma delta epsilon " * 80)
        + "\n\n"
        + ("traceability retrieval evidence snippet " * 80)
    )
    chunks = _chunk_markdown_body(long_body, "Chunked")

    assert len(chunks) >= 2
    assert all(chunk.body for chunk in chunks)


def test_paragraphs_skip_blank_and_heading_lines() -> None:
    from src.services.search_service import _paragraphs

    paragraphs = _paragraphs(
        "# Heading\n\nFirst line\nSecond line\n\n## Next\n\nThird line\n"
    )

    assert paragraphs == ["First line Second line", "Third line"]


def test_search_service_returns_ranked_results_and_limit(test_project) -> None:
    test_project.write_file("wiki/sources/first.md", "alpha alpha beta")
    test_project.write_file("wiki/sources/second.md", "alpha")
    test_project.write_file("wiki/index.md", "alpha alpha alpha")

    results = test_project.services["search"].search("alpha beta", limit=2)

    assert len(results) == 2
    assert results[0].score >= results[1].score
    # wiki/index.md is a maintenance page excluded from the FTS index
    assert results[0].path == "wiki/sources/first.md"


def test_search_service_refresh_is_noop_when_inventory_is_unchanged(
    test_project,
) -> None:
    test_project.write_file("wiki/sources/reindex.md", "alpha body")
    service = test_project.services["search"]

    assert service.refresh(force=True) is True
    assert service.refresh() is False


def test_search_service_refresh_short_circuits_when_fts_is_disabled(
    test_project,
) -> None:
    service = test_project.services["search"]
    service._fts_available = False

    assert service.refresh() is False


def test_search_service_refresh_marks_fts_unavailable_on_store_error(
    monkeypatch, test_project
) -> None:
    from src.storage.search_index_store import SearchIndexUnavailable

    service = test_project.services["search"]

    def raise_unavailable() -> dict[str, tuple[int, int]]:
        raise SearchIndexUnavailable("fts5 unavailable")

    monkeypatch.setattr(service.index_store, "load_indexed_files", raise_unavailable)

    assert service.refresh() is False
    assert service._fts_available is False


def test_search_service_search_falls_back_when_index_query_fails(
    monkeypatch, test_project
) -> None:
    from src.storage.search_index_store import SearchIndexUnavailable

    test_project.write_file("wiki/sources/fallback.md", "traceability body")
    service = test_project.services["search"]

    monkeypatch.setattr(service, "refresh", lambda force=False: False)

    def raise_unavailable(*_args, **_kwargs):
        raise SearchIndexUnavailable("fts5 unavailable")

    monkeypatch.setattr(service.index_store, "search", raise_unavailable)

    results = service.search("traceability")

    assert service._fts_available is False
    assert any(result.path == "wiki/sources/fallback.md" for result in results)


def test_search_service_builds_sqlite_index_and_returns_best_chunk(
    test_project,
) -> None:
    test_project.write_file(
        "wiki/sources/chunks.md",
        "---\ntitle: Chunked Page\nsummary: Example\n---\n\n"
        "# Chunked Page\n\n"
        "## Intro\n\n"
        "This section is about setup and does not mention the target term.\n\n"
        "## Retrieval\n\n"
        "SQLite FTS5 chunk search keeps the relevant retrieval snippet together.\n",
    )

    results = test_project.services["search"].search("SQLite retrieval", limit=3)

    assert (test_project.paths.graph_exports_dir / "search_index.sqlite3").exists()
    assert len(results) >= 1
    assert results[0].path == "wiki/sources/chunks.md"
    assert "SQLite FTS5 chunk search" in results[0].snippet


def test_search_service_returns_empty_for_blank_query(test_project) -> None:
    assert test_project.services["search"].search("!!!") == []


def test_search_service_inventory_returns_empty_for_missing_wiki_dir(
    uninitialized_project,
) -> None:
    assert uninitialized_project.services["search"]._wiki_inventory() == {}


def test_query_service_returns_fallback_when_no_matches(test_project) -> None:
    answer = _provider_query_service(test_project, "unused").answer_question(
        "What is missing?"
    )

    assert answer.citations == []
    assert "No compiled wiki pages matched" in answer.answer
    assert answer.mode == "no-matches"


def test_query_service_returns_answer_with_citations(test_project) -> None:
    test_project.write_file("wiki/sources/citations.md", "traceability appears here")

    answer = _provider_query_service(
        test_project,
        "Traceability appears here. [Citations]",
    ).answer_question("traceability")

    assert answer.citations
    assert answer.citations[0].path == "wiki/sources/citations.md"
    assert "traceability appears here" in answer.answer.lower()


def test_query_service_save_answer_writes_analysis_page(test_project) -> None:
    test_project.write_file("wiki/sources/citations.md", "traceability appears here")
    query_service = _provider_query_service(
        test_project,
        "Traceability appears here. [Citations]",
    )
    answer = query_service.answer_question("traceability")

    saved_path = query_service.save_answer("How does traceability work?", answer)

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
    query_service = _provider_query_service(
        test_project,
        "Traceability appears here. [Citations]",
    )
    answer = query_service.answer_question("traceability")

    saved_path = query_service.save_answer("???", answer)

    assert saved_path.startswith("wiki/concepts/analysis-")
    assert (test_project.root / saved_path).exists()


def test_query_service_normalizes_cited_sentences_into_claims(test_project) -> None:
    service = QueryService(test_project.paths, test_project.services["search"])
    evidence_bundle = EvidenceBundle(
        question="traceability",
        items=[
            EvidenceItem(
                page_path="wiki/sources/traceability-notes.md",
                title="Traceability Notes",
                snippet="Traceability preserves source links in the wiki.",
                score=3,
            )
        ],
    )

    claims = service._normalize_claims(
        (
            "Traceability preserves source links. [Traceability Notes]\n"
            "- Source pages remain reviewable. [Traceability Notes]"
        ),
        evidence_bundle,
    )

    assert [claim.text for claim in claims] == [
        "Traceability preserves source links",
        "Source pages remain reviewable",
    ]
    assert all(claim.grounded for claim in claims)
    assert all(
        claim.source_page == "wiki/sources/traceability-notes.md" for claim in claims
    )


def test_query_service_merge_collapses_near_duplicate_claims(test_project) -> None:
    service = QueryService(test_project.paths, test_project.services["search"])
    evidence_bundle = EvidenceBundle(
        question="traceability",
        items=[
            EvidenceItem(
                page_path="wiki/sources/traceability-notes.md",
                title="Traceability Notes",
                snippet="Traceability preserves source links in the wiki.",
                score=3,
            )
        ],
    )
    candidates = [
        CandidateAnswer(
            raw_text="A",
            claims=[
                Claim(
                    text="Traceability preserves source links",
                    source_page="wiki/sources/traceability-notes.md",
                    section="Traceability Notes",
                    confidence=1.0,
                    grounded=True,
                )
            ],
            model_name="fake-model",
        ),
        CandidateAnswer(
            raw_text="B",
            claims=[
                Claim(
                    text="Source links are preserved by traceability",
                    source_page="wiki/sources/traceability-notes.md",
                    section="Traceability Notes",
                    confidence=1.0,
                    grounded=True,
                )
            ],
            model_name="fake-model",
        ),
    ]

    merged = service._merge_candidates(candidates, evidence_bundle)

    assert merged.candidate_count == 2
    assert len(merged.accepted_claims) == 1
    assert len(merged.dropped_claims) == 0
    assert "[Traceability Notes]" in merged.text


def test_query_service_self_consistency_persists_run_and_merges_consensus(
    test_project,
) -> None:
    test_project.write_file(
        "wiki/sources/traceability-notes.md",
        _compiled_page(
            "Traceability Notes",
            "Traceability preserves source links in the wiki.",
        ),
    )
    provider = SequencedProvider(
        [
            "Traceability preserves source links. [Traceability Notes]",
            "Traceability preserves source links. [Traceability Notes]",
            "Traceability preserves source links. [Traceability Notes]",
        ]
    )
    run_store = RunStore(test_project.paths.graph_exports_dir / "self-consistency.db")
    service = QueryService(
        test_project.paths,
        test_project.services["search"],
        provider=provider,
        run_store=run_store,
    )

    answer = service.answer_question("traceability", self_consistency=3)

    assert answer.mode == "self-consistency:fake-model:3"
    assert answer.run_id is not None
    assert "Traceability preserves source links." in answer.answer
    record = run_store.get_run(answer.run_id)
    assert record is not None
    assert record.evidence_bundle is not None
    assert record.context_hash == record.evidence_bundle.context_hash
    assert len(record.candidates) == 3
    assert record.merged_answer is not None
    assert record.merged_answer.candidate_count == 3
    assert len(record.merged_answer.accepted_claims) == 1
    assert record.final_text == answer.answer


def test_query_service_self_consistency_drops_ungrounded_claims(test_project) -> None:
    test_project.write_file(
        "wiki/sources/traceability-notes.md",
        _compiled_page(
            "Traceability Notes",
            "Traceability preserves source links in the wiki.",
        ),
    )
    provider = SequencedProvider(
        [
            "The moon is made of cheese.",
            "Blue whales live on the moon.",
            "Moon cheese explains everything.",
        ]
    )
    run_store = RunStore(test_project.paths.graph_exports_dir / "no-consensus.db")
    service = QueryService(
        test_project.paths,
        test_project.services["search"],
        provider=provider,
        run_store=run_store,
    )

    answer = service.answer_question("traceability", self_consistency=3)

    assert answer.run_id is not None
    assert answer.answer == (
        "The sampled answers did not converge on a sufficiently grounded response "
        "from the available evidence."
    )
    record = run_store.get_run(answer.run_id)
    assert record is not None
    assert record.merged_answer is not None
    assert record.merged_answer.accepted_claims == []
    assert len(record.merged_answer.dropped_claims) == 3
    assert record.unresolved_disagreement is True


def test_query_service_self_consistency_fails_when_one_sample_fails(
    test_project,
) -> None:
    test_project.write_file(
        "wiki/sources/traceability-notes.md",
        _compiled_page(
            "Traceability Notes",
            "Traceability preserves source links in the wiki.",
        ),
    )
    provider = SequencedProvider(
        [
            RuntimeError("transient provider failure"),
            "Traceability preserves source links. [Traceability Notes]",
            "Traceability preserves source links. [Traceability Notes]",
        ]
    )
    run_store = RunStore(test_project.paths.graph_exports_dir / "partial-success.db")
    service = QueryService(
        test_project.paths,
        test_project.services["search"],
        provider=provider,
        run_store=run_store,
    )

    with pytest.raises(ProviderExecutionError, match="transient provider failure"):
        service.answer_question("traceability", self_consistency=3)


def test_query_service_without_provider_raises_configuration_error(
    test_project,
) -> None:
    test_project.write_file("wiki/sources/citations.md", "traceability appears here")

    with pytest.raises(ProviderConfigurationError, match="kb query requires"):
        test_project.services["query"].answer_question("traceability")


def test_query_service_self_consistency_raises_when_sampling_crashes(
    test_project,
) -> None:
    test_project.write_file("wiki/sources/citations.md", "traceability appears here")
    provider = SequencedProvider(["unused"])
    service = QueryService(
        test_project.paths,
        test_project.services["search"],
        provider=provider,
    )

    with patch.object(
        service,
        "_sample_candidates",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        with pytest.raises(ProviderExecutionError, match="boom"):
            service.answer_question("traceability", self_consistency=3)


def test_query_service_self_consistency_raises_when_all_samples_fail(
    test_project,
) -> None:
    test_project.write_file("wiki/sources/citations.md", "traceability appears here")
    provider = SequencedProvider(
        [
            RuntimeError("first failure"),
            RuntimeError("second failure"),
            RuntimeError("third failure"),
        ]
    )
    service = QueryService(
        test_project.paths,
        test_project.services["search"],
        provider=provider,
    )

    with pytest.raises(ProviderExecutionError, match="first failure"):
        service.answer_question("traceability", self_consistency=3)


def test_query_service_self_consistency_without_run_store_returns_no_run_id(
    test_project,
) -> None:
    test_project.write_file(
        "wiki/sources/traceability-notes.md",
        _compiled_page(
            "Traceability Notes",
            "Traceability preserves source links in the wiki.",
        ),
    )
    provider = SequencedProvider(
        [
            "Traceability preserves source links. [Traceability Notes]",
            "Traceability preserves source links. [Traceability Notes]",
        ]
    )
    service = QueryService(
        test_project.paths,
        test_project.services["search"],
        provider=provider,
    )

    answer = service.answer_question("traceability", self_consistency=2)

    assert answer.mode == "self-consistency:fake-model:2"
    assert answer.run_id is None


def test_query_service_normalize_claims_skips_citation_only_segments(
    test_project,
) -> None:
    service = QueryService(test_project.paths, test_project.services["search"])
    evidence_bundle = EvidenceBundle(
        question="traceability",
        items=[
            EvidenceItem(
                page_path="wiki/sources/traceability-notes.md",
                title="Traceability Notes",
                snippet="Traceability preserves source links in the wiki.",
                score=3,
            )
        ],
    )

    claims = service._normalize_claims(
        "[Traceability Notes]\nTraceability preserves source links. [Traceability Notes]",
        evidence_bundle,
    )

    assert len(claims) == 1
    assert claims[0].text == "Traceability preserves source links"


def test_query_service_best_evidence_match_returns_none_for_blank_claim(
    test_project,
) -> None:
    service = QueryService(test_project.paths, test_project.services["search"])
    evidence_bundle = EvidenceBundle(question="traceability", items=[])

    item, ratio, overlap = service._best_evidence_match("", evidence_bundle)

    assert item is None
    assert ratio == 0.0
    assert overlap == 0


def test_query_service_merge_skips_blank_claims_and_renders_without_title(
    test_project,
) -> None:
    service = QueryService(test_project.paths, test_project.services["search"])
    evidence_bundle = EvidenceBundle(question="traceability", items=[])
    candidates = [
        CandidateAnswer(
            raw_text="A",
            claims=[
                Claim(text="   ", grounded=False),
                Claim(
                    text="Traceability remains reviewable",
                    source_page="wiki/sources/missing.md",
                    confidence=1.0,
                    grounded=True,
                ),
            ],
            model_name="fake-model",
        )
    ]

    merged = service._merge_candidates(candidates, evidence_bundle)

    assert merged.text == "Traceability remains reviewable."
    assert len(merged.accepted_claims) == 1


def test_query_service_claims_match_rejects_disjoint_claims(test_project) -> None:
    service = QueryService(test_project.paths, test_project.services["search"])

    assert (
        service._claims_match(
            Claim(text="Traceability preserves source links", grounded=True),
            Claim(text="PDF conversion uses Docling", grounded=True),
        )
        is False
    )


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
    query_service = _provider_query_service(
        test_project,
        "Traceability appears here. [Citations]",
    )
    answer = query_service.answer_question("traceability")

    saved_path = query_service.save_answer("What is traceability?", answer)

    assert (test_project.root / saved_path).exists()


def test_save_answer_refreshes_search_index_for_analysis_pages(test_project) -> None:
    from src.services.query_service import QueryAnswer

    answer = QueryAnswer(
        answer="Persistent traceability analysis lives in the wiki.",
        citations=[],
        mode="test",
    )

    saved_path = test_project.services["query"].save_answer(
        "What is persistent traceability?",
        answer,
    )

    assert (test_project.paths.graph_exports_dir / "search_index.sqlite3").exists()
    results = test_project.services["search"].search("persistent traceability")
    paths = {result.path for result in results}
    assert saved_path in paths


def test_indexable_chunks_skip_generated_concepts(test_project) -> None:
    service = test_project.services["search"]
    path = test_project.write_file(
        "wiki/concepts/generated.md",
        "---\ntitle: Generated\ntype: concept\nsummary: S\n"
        "generated_at: 2026-04-19T00:00:00Z\nsource_pages: []\n---\n\n# Generated\n",
    )

    chunks = service._indexable_chunks(path, "wiki/concepts/generated.md")

    assert chunks == []


def test_indexable_chunks_returns_empty_for_unreadable_file(
    monkeypatch, test_project
) -> None:
    service = test_project.services["search"]
    path = test_project.write_file("wiki/sources/unreadable.md", "content")
    original_read_text = Path.read_text

    def fake_read_text(self: Path, *args, **kwargs):
        if self == path:
            raise OSError("boom")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    chunks = service._indexable_chunks(path, "wiki/sources/unreadable.md")

    assert chunks == []


def test_search_index_store_wraps_sqlite_operational_errors(
    monkeypatch, tmp_path
) -> None:
    from src.storage.search_index_store import SearchIndexStore, SearchIndexUnavailable

    class BrokenConnection:
        def execute(self, *_args, **_kwargs):
            return None

        def executescript(self, *_args, **_kwargs):
            raise sqlite3.OperationalError("no such module: fts5")

        def close(self) -> None:
            return None

    monkeypatch.setattr(sqlite3, "connect", lambda _path: BrokenConnection())

    store = SearchIndexStore(tmp_path / "search_index.sqlite3")

    with pytest.raises(SearchIndexUnavailable):
        store.load_indexed_files()


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


# --- P1 FTS5 improvement tests ---


def test_search_does_not_scan_markdown_when_fts_returns_no_hits(
    monkeypatch, test_project
) -> None:
    """Zero-hit FTS result should be returned as-is; markdown scan must not run."""
    test_project.write_file("wiki/sources/page.md", "xyzzy unique term present")
    service = test_project.services["search"]

    scan_called: list[bool] = []
    original_scan = service._scan_markdown_files

    def tracking_scan(*args, **kwargs):
        scan_called.append(True)
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(service, "_scan_markdown_files", tracking_scan)

    # Query for a term that won't be in the FTS stemmed index
    results = service.search("zzznomatchzzzqqqq", limit=5)

    assert results == []
    assert (
        not scan_called
    ), "markdown scan must not run when FTS is available and healthy"


def test_search_index_removes_deleted_files(test_project) -> None:
    """Pages deleted from wiki/ must disappear from search results after refresh."""
    path = test_project.write_file(
        "wiki/sources/deleted-page.md", "ephemeral content here"
    )
    service = test_project.services["search"]

    service.refresh(force=True)
    before = service.search("ephemeral", limit=5)
    assert any("deleted-page" in r.path for r in before)

    path.unlink()
    service.refresh(force=True)
    after = service.search("ephemeral", limit=5)
    assert not any("deleted-page" in r.path for r in after)


def test_search_index_updates_changed_file(test_project) -> None:
    """After a file is modified, a fresh refresh must reflect the new content."""
    path = test_project.write_file("wiki/sources/mutable.md", "original term aplhabet")
    service = test_project.services["search"]

    service.refresh(force=True)
    before = service.search("aplhabet", limit=5)
    assert any("mutable" in r.path for r in before)

    path.write_text("replacement content newword", encoding="utf-8")
    service.refresh(force=True)

    after_old = service.search("aplhabet", limit=5)
    after_new = service.search("newword", limit=5)
    assert not any("mutable" in r.path for r in after_old)
    assert any("mutable" in r.path for r in after_new)


def test_maintenance_pages_excluded_from_search_index(test_project) -> None:
    """wiki/index.md and wiki/log.md must not appear in FTS search results."""
    test_project.write_file("wiki/index.md", "maintenance uniqueindextoken content")
    test_project.write_file("wiki/log.md", "maintenance uniquelogtoken activity")

    service = test_project.services["search"]
    service.refresh(force=True)

    assert service.search("uniqueindextoken", limit=5) == []
    assert service.search("uniquelogtoken", limit=5) == []


def test_selective_frontmatter_excludes_implementation_metadata(test_project) -> None:
    """raw_path, source_hash, and compiled_at must not be indexed for search."""
    test_project.write_file(
        "wiki/sources/impl-meta.md",
        "---\n"
        "title: Implementation Meta\n"
        "raw_path: raw/sources/hidden.md\n"
        "source_hash: abc123hashvalue\n"
        "compiled_at: 2026-04-19T00:00:00Z\n"
        "provider: openai\n"
        "---\n\n"
        "# Implementation Meta\n\n"
        "Body without hidden terms.\n",
    )
    service = test_project.services["search"]
    service.refresh(force=True)

    # These implementation-detail tokens must not match
    assert service.search("abc123hashvalue", limit=5) == []
    assert service.search("openai", limit=5) == []
    # But the title must still be indexed
    results = service.search("implementation meta", limit=5)
    assert any("impl-meta" in r.path for r in results)


def test_search_index_rebuilds_on_version_mismatch(monkeypatch, test_project) -> None:
    """refresh() must trigger a full rebuild if the stored version doesn't match."""
    from src.storage.search_index_store import SearchIndexStore

    test_project.write_file("wiki/sources/versioned.md", "version check content")
    service = test_project.services["search"]
    service.refresh(force=True)

    # Tamper the stored schema version to simulate a stale index
    original_check = service.index_store.check_version
    monkeypatch.setattr(service.index_store, "check_version", lambda: False)

    rebuild_called: list[bool] = []
    original_rebuild = service.index_store.rebuild

    def tracking_rebuild(*args, **kwargs):
        rebuild_called.append(True)
        return original_rebuild(*args, **kwargs)

    monkeypatch.setattr(service.index_store, "rebuild", tracking_rebuild)

    # Even though inventory hasn't changed, stale version must trigger rebuild
    result = service.refresh()
    assert result is True
    assert rebuild_called


def test_refresh_file_upserts_single_file_without_full_rebuild(
    monkeypatch, test_project
) -> None:
    """refresh_file() must call upsert_file, not rebuild."""
    test_project.write_file("wiki/sources/single.md", "upsert candidate content")
    service = test_project.services["search"]

    upsert_called: list[bool] = []
    rebuild_called: list[bool] = []
    original_upsert = service.index_store.upsert_file

    def tracking_upsert(*args, **kwargs):
        upsert_called.append(True)
        return original_upsert(*args, **kwargs)

    def tracking_rebuild(*args, **kwargs):
        rebuild_called.append(True)

    monkeypatch.setattr(service.index_store, "upsert_file", tracking_upsert)
    monkeypatch.setattr(service.index_store, "rebuild", tracking_rebuild)

    path = test_project.paths.root / "wiki/sources/single.md"
    service.refresh_file(path)

    assert upsert_called
    assert not rebuild_called


def test_search_index_store_upsert_replaces_stale_chunks(test_project) -> None:
    """upsert_file must replace old chunks so stale terms are no longer searchable."""
    from src.storage.search_index_store import (
        IndexedChunk,
        IndexedFileState,
        SearchIndexStore,
    )

    store = test_project.services["search"].index_store
    state = IndexedFileState(page_path="wiki/sources/up.md", mtime_ns=1, size_bytes=10)
    old_chunk = IndexedChunk(
        page_path="wiki/sources/up.md",
        page_type="source",
        title="Up",
        section="Up",
        chunk_index=0,
        metadata="",
        body="stale obsolete content",
    )
    store.upsert_file(state, [old_chunk])

    new_chunk = IndexedChunk(
        page_path="wiki/sources/up.md",
        page_type="source",
        title="Up",
        section="Up",
        chunk_index=0,
        metadata="",
        body="fresh updated content",
    )
    store.upsert_file(
        IndexedFileState(page_path="wiki/sources/up.md", mtime_ns=2, size_bytes=20),
        [new_chunk],
    )

    hits_old = store.search('"stale"', limit=5)
    hits_new = store.search('"fresh"', limit=5)
    assert not hits_old
    assert hits_new


def test_search_index_store_delete_missing_files(test_project) -> None:
    """delete_missing_files must remove pages no longer in the given path set."""
    from src.storage.search_index_store import IndexedChunk, IndexedFileState

    store = test_project.services["search"].index_store
    for slug in ("keep", "remove"):
        store.upsert_file(
            IndexedFileState(
                page_path=f"wiki/sources/{slug}.md", mtime_ns=1, size_bytes=5
            ),
            [
                IndexedChunk(
                    page_path=f"wiki/sources/{slug}.md",
                    page_type="source",
                    title=slug.title(),
                    section=slug.title(),
                    chunk_index=0,
                    metadata="",
                    body=f"{slug} unique term",
                )
            ],
        )

    deleted = store.delete_missing_files({"wiki/sources/keep.md"})

    assert deleted == 1
    indexed = store.load_indexed_files()
    assert "wiki/sources/keep.md" in indexed
    assert "wiki/sources/remove.md" not in indexed


def test_page_dedup_returns_enough_unique_pages(test_project) -> None:
    """When one page has many high-ranked chunks, other pages must still appear."""
    # Create one very long page that will produce many chunks
    long_body = "## Section {i}\n\nalpha beta gamma delta epsilon " * 40
    test_project.write_file(
        "wiki/sources/dominant.md",
        f"# Dominant\n\n{long_body}",
    )
    # Create several shorter pages that also match
    for i in range(4):
        test_project.write_file(
            f"wiki/sources/other-{i}.md",
            f"alpha beta result page {i}",
        )

    results = test_project.services["search"].search("alpha beta", limit=4)

    unique_paths = {r.path for r in results}
    # Must return 4 unique pages, not just the dominant page repeated
    assert len(unique_paths) == len(results) == 4


def test_search_index_meta_version_stored_after_rebuild(test_project) -> None:
    """Metadata table must store schema_version and chunker_version after rebuild."""
    test_project.write_file("wiki/sources/meta-check.md", "version metadata test")
    service = test_project.services["search"]
    service.refresh(force=True)

    assert service.index_store.check_version() is True
    assert service.index_store.load_meta("schema_version") == "1"
    assert service.index_store.load_meta("chunker_version") == "1"


def test_search_falls_back_to_scan_when_fts_was_already_disabled(test_project) -> None:
    """search() must use markdown scan when _fts_available is False before the call."""
    test_project.write_file(
        "wiki/sources/fallback-pre.md", "determinism content present"
    )
    service = test_project.services["search"]
    service._fts_available = False

    results = service.search("determinism")

    assert any("fallback-pre" in r.path for r in results)


def test_refresh_file_noop_when_fts_is_disabled(test_project) -> None:
    """refresh_file() must return immediately when _fts_available is False."""
    service = test_project.services["search"]
    service._fts_available = False

    upsert_called: list[bool] = []
    service.index_store.upsert_file = lambda *_a, **_kw: upsert_called.append(True)  # type: ignore[method-assign]

    path = test_project.paths.root / "wiki/sources/noop.md"
    service.refresh_file(path)

    assert not upsert_called


def test_refresh_file_marks_fts_unavailable_on_upsert_error(
    monkeypatch, test_project
) -> None:
    """refresh_file() must disable FTS when upsert_file raises SearchIndexUnavailable."""
    from src.storage.search_index_store import SearchIndexUnavailable

    path = test_project.write_file("wiki/sources/upsert-fail.md", "content")
    service = test_project.services["search"]

    def raise_unavailable(*_a, **_kw):
        raise SearchIndexUnavailable("broken")

    monkeypatch.setattr(service.index_store, "upsert_file", raise_unavailable)

    service.refresh_file(path)

    assert service._fts_available is False


def test_refresh_file_logs_warning_on_os_error(monkeypatch, test_project) -> None:
    """refresh_file() must handle OSError gracefully without disabling FTS."""
    service = test_project.services["search"]
    path = test_project.paths.root / "wiki/sources/nonexistent-file.md"

    # upsert_file won't even be called since stat() will raise OSError first
    service.refresh_file(path)

    # FTS should remain available — only a warning is expected
    assert service._fts_available is True


def test_search_index_returns_empty_snippet_fallback(test_project) -> None:
    """When a FTS hit has no snippet text, the section/title must be used instead."""
    from src.storage.search_index_store import SearchHit

    test_project.write_file("wiki/sources/snippetless.md", "sparse relevant body")
    service = test_project.services["search"]

    original_search = service.index_store.search

    def patched_search(query, *, limit):
        hits = original_search(query, limit=limit)
        # Return hit with empty snippet to exercise the fallback
        return [
            SearchHit(
                page_path=h.page_path,
                title=h.title,
                section=h.section,
                snippet="",
                score=h.score,
            )
            for h in hits
        ]

    service.index_store.search = patched_search  # type: ignore[method-assign]
    results = service.search("sparse relevant", limit=5)

    assert results
    assert results[0].snippet  # must be section or title, not empty


def test_scan_markdown_skips_zero_scoring_pages(test_project) -> None:
    """_scan_markdown_files must omit pages with no term occurrences."""
    test_project.write_file("wiki/sources/nomatch.md", "unrelated content here")
    service = test_project.services["search"]
    service._fts_available = False  # force scan path

    results = service.search("zzznomatch", limit=5)

    assert results == []


def test_scan_markdown_skips_concept_pages(test_project) -> None:
    """_scan_markdown_files must skip generated concept pages."""
    test_project.write_file(
        "wiki/concepts/generated-concept.md",
        "---\ntype: concept\n---\n\n# Concept\n\ngenerated concept body\n",
    )
    service = test_project.services["search"]
    service._fts_available = False  # force scan path

    results = service.search("generated", limit=5)

    assert not any("generated-concept" in r.path for r in results)


def test_indexable_chunks_fallback_for_empty_body_page(test_project) -> None:
    """A page with frontmatter but no body must produce a single title chunk."""
    service = test_project.services["search"]
    path = test_project.write_file(
        "wiki/sources/empty-body.md",
        "---\ntitle: Empty Body Page\nsummary: no body content\n---\n",
    )

    chunks = service._indexable_chunks(path, "wiki/sources/empty-body.md")

    assert len(chunks) == 1
    assert chunks[0].title == "Empty Body Page"


def test_chunk_markdown_body_handles_non_paragraph_section(test_project) -> None:
    """A section whose text has no blank-line groups must be treated as one paragraph."""
    from src.services.search_service import _chunk_markdown_body

    # Section text with no blank lines — _paragraphs returns [] for single-line-group
    # but the text itself is non-empty, triggering the fallback
    text = "## Tight Section\n\nline one\nline two\nline three\n"
    chunks = _chunk_markdown_body(text, "Tight Section")

    assert chunks


def test_chunk_markdown_body_skips_blank_normalized_paragraphs() -> None:
    """Normalized empty paragraph strings must be silently skipped."""
    from src.services.search_service import _chunk_markdown_body

    text = "# Title\n\n   \n\nReal paragraph content here\n"
    chunks = _chunk_markdown_body(text, "Title")

    assert chunks
    assert all(chunk.body.strip() for chunk in chunks)


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

    # Modify the actual normalized file on disk to simulate a source change.
    sources = test_project.services["manifest"].list_sources()
    record = sources[0]
    norm_path = test_project.root / (record.normalized_path or record.raw_path)
    norm_path.write_text("# Diff\n\nEdited body\n", encoding="utf-8")

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


# --- P2 integration tests: cross-service paths ---


def test_ingest_compile_edit_recompile_lint_stale_cycle(test_project) -> None:
    path = test_project.write_file("notes/cycle.md", "# Cycle\n\nOriginal body.\n")
    test_project.services["ingest"].ingest_path(path)
    test_project.services["compile"].compile()

    report = test_project.services["lint"].lint()
    stale = [i for i in report.issues if i.code == "stale-source-page"]
    assert len(stale) == 0

    sources = test_project.services["manifest"].list_sources()
    sources[0].content_hash = "edited-hash"
    test_project.services["manifest"].save_source(sources[0])

    report2 = test_project.services["lint"].lint()
    assert any(i.code == "stale-source-page" for i in report2.issues)

    test_project.services["compile"].compile(force=True)

    report3 = test_project.services["lint"].lint()
    assert not any(i.code == "stale-source-page" for i in report3.issues)


def test_ingest_two_sources_compile_lint_both_orphans(test_project) -> None:
    test_project.write_file(
        "wiki/sources/lonely-a.md",
        _compiled_page("Lonely A", "Body A no links."),
    )
    test_project.write_file(
        "wiki/sources/lonely-b.md",
        _compiled_page("Lonely B", "Body B no links."),
    )

    report = test_project.services["lint"].lint()
    orphans = [i for i in report.issues if i.code == "orphan-page"]

    source_orphans = [o for o in orphans if "sources/" in o.path]
    assert len(source_orphans) >= 2


def test_ingest_compile_query_save_lint_saved_page(test_project) -> None:
    path = test_project.write_file("notes/qa.md", "# QA\n\nTraceability evidence.\n")
    test_project.services["ingest"].ingest_path(path)
    test_project.services["compile"].compile()

    query_service = _provider_query_service(
        test_project,
        "Traceability evidence. [Qa]",
    )
    answer = query_service.answer_question("traceability")
    saved = query_service.save_answer("What is traceability?", answer)
    assert (test_project.root / saved).exists()

    report = test_project.services["lint"].lint()
    saved_issues = [i for i in report.issues if saved.replace("/", "/") in i.path]
    codes = {i.code for i in saved_issues}
    assert "broken-link" not in codes


def test_ingest_compile_export_vault_mirrors_wiki(test_project) -> None:
    path = test_project.write_file("notes/vault.md", "# Vault\n\nVault body.\n")
    test_project.services["ingest"].ingest_path(path)
    test_project.services["compile"].compile()
    test_project.services["export"].export_vault()

    wiki_files = sorted(
        f.relative_to(test_project.paths.wiki_dir).as_posix()
        for f in test_project.paths.wiki_dir.rglob("*.md")
    )
    vault_files = sorted(
        f.relative_to(test_project.paths.vault_obsidian_dir).as_posix()
        for f in test_project.paths.vault_obsidian_dir.rglob("*.md")
    )
    assert wiki_files == vault_files


def test_ingest_duplicate_status_shows_count_one(test_project) -> None:
    path = test_project.write_file("notes/dup.md", "# Dup\n\nDuplicate body.\n")
    first = test_project.services["ingest"].ingest_path(path)
    second = test_project.services["ingest"].ingest_path(path)

    assert first.created is True
    assert second.created is False

    snapshot = test_project.services["status"].snapshot(initialized=True)
    assert snapshot.source_count == 1


def test_ingest_compile_search_returns_correct_paths(test_project) -> None:
    path = test_project.write_file(
        "notes/search-test.md", "# Search Test\n\nKnowledge base findable.\n"
    )
    test_project.services["ingest"].ingest_path(path)
    test_project.services["compile"].compile()

    results = test_project.services["search"].search("knowledge base")

    matching = [r for r in results if "search-test" in r.path]
    assert len(matching) >= 1
    assert matching[0].path.startswith("wiki/sources/")


def test_search_snippet_excludes_frontmatter(test_project) -> None:
    test_project.write_file(
        "wiki/sources/frontmatter-test.md",
        "---\ntitle: Frontmatter Test\nsummary: Meta here\n---\n\n"
        "# Frontmatter Test\n\nThe real body has traceability.\n",
    )

    results = test_project.services["search"].search("traceability")

    assert len(results) >= 1
    snippet = results[0].snippet
    assert "traceability" in snippet
    assert "---" not in snippet
    assert "summary:" not in snippet


def test_search_excludes_generated_concept_pages_but_includes_analysis(
    test_project,
) -> None:
    test_project.write_file(
        "wiki/sources/source-page.md",
        "# Source Page\n\nTraceability evidence.\n",
    )
    test_project.write_file(
        "wiki/concepts/analysis-page.md",
        "---\ntitle: Analysis Page\nsummary: S\ntype: analysis\n---\n\n"
        "# Analysis Page\n\nTraceability reused here.\n",
    )
    test_project.write_file(
        "wiki/concepts/gen-concept.md",
        "---\ntitle: Generated Concept\ntype: concept\nsummary: S\n"
        "generated_at: 2026-04-19T00:00:00Z\nsource_pages: []\n---\n\n"
        "# Generated Concept\n\nTraceability concept.\n",
    )

    results = test_project.services["search"].search("traceability")

    paths = [r.path for r in results]
    assert any("wiki/sources/" in p for p in paths)
    # Analysis pages are now searchable
    assert any("analysis-page" in p for p in paths)
    # Generated concept pages are still excluded
    assert not any("gen-concept" in p for p in paths)


def test_save_answer_includes_summary_in_frontmatter(test_project) -> None:
    query_service = _provider_query_service(
        test_project,
        "Traceability is preserved through compiled source pages.",
    )
    test_project.write_file("wiki/sources/sr.md", "traceability appears here")
    answer = query_service.answer_question("traceability")

    saved_path = query_service.save_answer("How does traceability work?", answer)

    content = (test_project.root / saved_path).read_text(encoding="utf-8")
    assert "summary:" in content
    assert "Traceability" in content.split("---")[1]


def test_save_answer_summary_fallback_for_empty_answer(test_project) -> None:
    from src.services.query_service import QueryAnswer, QueryService

    query_service = test_project.services["query"]
    answer = QueryAnswer(answer="", citations=[], mode="test")

    saved_path = query_service.save_answer("What is traceability?", answer)

    content = (test_project.root / saved_path).read_text(encoding="utf-8")
    assert "summary: 'Analysis page for: What is traceability?'" in content
