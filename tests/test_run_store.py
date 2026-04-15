"""Tests for src/storage/run_store.py — SQLite run artifact persistence."""

from __future__ import annotations

import pytest

from src.schemas.claims import (
    CandidateAnswer,
    Claim,
    EvidenceBundle,
    EvidenceItem,
    MergedAnswer,
)
from src.schemas.review import ReviewFinding, Verdict
from src.schemas.runs import RunRecord
from src.storage.run_store import RunStore


@pytest.fixture()
def store(tmp_path):
    """Create a RunStore backed by a temporary SQLite file."""
    s = RunStore(db_path=tmp_path / "test_runs.db")
    yield s
    s.close()


# ── Helpers ──────────────────────────────────────────────────────────


def _make_query_run(**overrides) -> RunRecord:
    defaults = dict(
        command="query",
        model_id="gpt-5.4-mini",
        evidence_bundle=EvidenceBundle(
            question="What is X?",
            items=[
                EvidenceItem(
                    page_path="wiki/sources/a.md", title="A", snippet="info about X"
                )
            ],
        ),
        candidates=[
            CandidateAnswer(
                raw_text="X is a concept.",
                claims=[Claim(text="X is a concept", source_page="wiki/sources/a.md")],
                model_name="gpt-5.4-mini",
                latency_ms=200,
            )
        ],
        merged_answer=MergedAnswer(
            text="X is a concept.",
            accepted_claims=[
                Claim(text="X is a concept", source_page="wiki/sources/a.md")
            ],
            candidate_count=1,
        ),
        final_text="X is a concept.",
        token_cost=150,
        wall_time_ms=450,
    )
    defaults.update(overrides)
    return RunRecord(**defaults)


def _make_review_run(**overrides) -> RunRecord:
    defaults = dict(
        command="review",
        model_id="claude-sonnet-4-6",
        review_findings=[
            ReviewFinding(
                issue_type="contradiction",
                affected_pages=["a.md", "b.md"],
                claim="X contradicts Y",
                verdict=Verdict.CONTRADICTORY,
                confidence=0.85,
            )
        ],
        final_text="Found 1 contradiction.",
        token_cost=300,
        wall_time_ms=900,
    )
    defaults.update(overrides)
    return RunRecord(**defaults)


# ── Positive / Happy Path ───────────────────────────────────────────


class TestRunStorePositive:
    def test_save_and_get_roundtrip(self, store):
        run = _make_query_run()
        returned_id = store.save_run(run)
        assert returned_id == run.run_id

        loaded = store.get_run(run.run_id)
        assert loaded is not None
        assert loaded.run_id == run.run_id
        assert loaded.command == "query"
        assert loaded.model_id == "gpt-5.4-mini"
        assert loaded.final_text == "X is a concept."
        assert loaded.token_cost == 150

    def test_evidence_bundle_preserved(self, store):
        run = _make_query_run()
        store.save_run(run)
        loaded = store.get_run(run.run_id)
        assert loaded.evidence_bundle is not None
        assert loaded.evidence_bundle.question == "What is X?"
        assert len(loaded.evidence_bundle.items) == 1

    def test_candidates_preserved(self, store):
        run = _make_query_run()
        store.save_run(run)
        loaded = store.get_run(run.run_id)
        assert len(loaded.candidates) == 1
        assert loaded.candidates[0].raw_text == "X is a concept."
        assert len(loaded.candidates[0].claims) == 1

    def test_merged_answer_preserved(self, store):
        run = _make_query_run()
        store.save_run(run)
        loaded = store.get_run(run.run_id)
        assert loaded.merged_answer is not None
        assert loaded.merged_answer.text == "X is a concept."
        assert loaded.merged_answer.candidate_count == 1

    def test_review_findings_preserved(self, store):
        run = _make_review_run()
        store.save_run(run)
        loaded = store.get_run(run.run_id)
        assert len(loaded.review_findings) == 1
        assert loaded.review_findings[0].verdict == Verdict.CONTRADICTORY

    def test_list_runs(self, store):
        r1 = _make_query_run()
        r2 = _make_review_run()
        store.save_run(r1)
        store.save_run(r2)
        runs = store.list_runs()
        assert len(runs) == 2

    def test_list_runs_filter_by_command(self, store):
        store.save_run(_make_query_run())
        store.save_run(_make_review_run())
        queries = store.list_runs(command="query")
        assert len(queries) == 1
        assert queries[0].command == "query"

    def test_citation_count(self, store):
        run = _make_query_run()
        store.save_run(run)
        count = store.citation_count(run.run_id)
        # 1 candidate claim + 1 merged accepted claim = 2
        assert count == 2

    def test_runs_citing_page(self, store):
        run = _make_query_run()
        store.save_run(run)
        ids = store.runs_citing_page("wiki/sources/a.md")
        assert run.run_id in ids


# ── Negative / Error Path ───────────────────────────────────────────


class TestRunStoreNegative:
    def test_get_nonexistent_returns_none(self, store):
        assert store.get_run("nonexistent") is None

    def test_runs_citing_nonexistent_page(self, store):
        assert store.runs_citing_page("no/such/page.md") == []

    def test_citation_count_nonexistent(self, store):
        assert store.citation_count("nope") == 0

    def test_list_runs_empty(self, store):
        assert store.list_runs() == []

    def test_list_runs_filter_no_match(self, store):
        store.save_run(_make_query_run())
        assert store.list_runs(command="nonexistent") == []


# ── Boundary / Edge Cases ───────────────────────────────────────────


class TestRunStoreBoundary:
    def test_empty_candidates_and_findings(self, store):
        run = RunRecord(command="query", final_text="No results.")
        store.save_run(run)
        loaded = store.get_run(run.run_id)
        assert loaded.candidates == []
        assert loaded.review_findings == []
        assert loaded.merged_answer is None

    def test_zero_token_cost(self, store):
        run = RunRecord(command="query", token_cost=0)
        store.save_run(run)
        loaded = store.get_run(run.run_id)
        assert loaded.token_cost == 0

    def test_very_long_claim_text(self, store):
        long_text = "A" * 10_000
        run = _make_query_run()
        run.candidates[0].claims[0].text = long_text
        store.save_run(run)
        loaded = store.get_run(run.run_id)
        assert loaded.candidates[0].claims[0].text == long_text

    def test_save_replaces_on_duplicate_run_id(self, store):
        run = _make_query_run()
        store.save_run(run)
        # Mutate and re-save with same run_id.
        run.final_text = "Updated answer."
        store.save_run(run)
        loaded = store.get_run(run.run_id)
        assert loaded.final_text == "Updated answer."

    def test_list_runs_respects_limit(self, store):
        for _ in range(5):
            store.save_run(_make_query_run())
        runs = store.list_runs(limit=3)
        assert len(runs) == 3

    def test_review_citations_stored(self, store):
        run = _make_review_run()
        store.save_run(run)
        count = store.citation_count(run.run_id)
        # 1 review finding → 1 citation row
        assert count == 1

    def test_multiple_runs_citing_same_page(self, store):
        r1 = _make_query_run()
        r2 = _make_query_run()
        store.save_run(r1)
        store.save_run(r2)
        ids = store.runs_citing_page("wiki/sources/a.md")
        assert r1.run_id in ids
        assert r2.run_id in ids


# ── Integration ──────────────────────────────────────────────────────


class TestRunStoreIntegration:
    def test_full_lifecycle(self, store):
        """Save query run + review run → list → filter → cite → close → reopen."""
        qr = _make_query_run()
        rr = _make_review_run()
        store.save_run(qr)
        store.save_run(rr)

        all_runs = store.list_runs()
        assert len(all_runs) == 2

        queries_only = store.list_runs(command="query")
        assert len(queries_only) == 1

        citing = store.runs_citing_page("wiki/sources/a.md")
        assert qr.run_id in citing

        # Close and reopen — data persists.
        db_path = store.db_path
        store.close()
        store2 = RunStore(db_path=db_path)
        assert store2.get_run(qr.run_id) is not None
        assert store2.get_run(rr.run_id) is not None
        store2.close()
