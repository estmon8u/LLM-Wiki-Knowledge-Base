from __future__ import annotations

import threading
from unittest.mock import AsyncMock, patch

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


def test_query_service_self_consistency_keeps_partial_success_when_one_sample_fails(
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

    answer = service.answer_question("traceability", self_consistency=3)

    assert answer.run_id is not None
    assert "Traceability preserves source links." in answer.answer
    record = run_store.get_run(answer.run_id)
    assert record is not None
    assert len(record.candidates) == 3
    assert sum(1 for candidate in record.candidates if candidate.error) == 1
    assert record.merged_answer is not None
    assert record.merged_answer.candidate_count == 2


def test_query_service_self_consistency_without_provider_reports_fallback_mode(
    test_project,
) -> None:
    test_project.write_file("wiki/sources/citations.md", "traceability appears here")

    answer = test_project.services["query"].answer_question(
        "traceability",
        self_consistency=3,
    )

    assert answer.mode == "heuristic:no-provider"
    assert "traceability appears here" in answer.answer


def test_query_service_self_consistency_falls_back_when_sampling_crashes(
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
        answer = service.answer_question("traceability", self_consistency=3)

    assert answer.mode == "heuristic-fallback"
    assert "traceability appears here" in answer.answer


def test_query_service_self_consistency_falls_back_when_all_samples_fail(
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

    answer = service.answer_question("traceability", self_consistency=3)

    assert answer.mode == "heuristic-fallback"
    assert "traceability appears here" in answer.answer
    assert answer.run_id is None


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

    answer = test_project.services["query"].answer_question("traceability")
    saved = test_project.services["query"].save_answer("What is traceability?", answer)
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
