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
    assert results[0].path in {"wiki/index.md", "wiki/sources/first.md"}


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
