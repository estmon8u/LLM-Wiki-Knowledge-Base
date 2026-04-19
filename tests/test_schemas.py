"""Tests for src/schemas/ — claims, review, and run record models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas.claims import (
    CandidateAnswer,
    Claim,
    EvidenceBundle,
    EvidenceItem,
    MergedAnswer,
)
from src.schemas.review import ReviewFinding, Verdict
from src.schemas.runs import RunRecord


# ── EvidenceItem ─────────────────────────────────────────────────────


class TestEvidenceItem:
    def test_positive_create(self):
        item = EvidenceItem(
            page_path="wiki/sources/a.md",
            title="A",
            snippet="hello",
            score=5,
            section="Intro",
            chunk_index=2,
        )
        assert item.page_path == "wiki/sources/a.md"
        assert item.score == 5
        assert item.section == "Intro"
        assert item.chunk_index == 2
        assert item.citation_ref == "wiki/sources/a.md#chunk-2"

    def test_default_score(self):
        item = EvidenceItem(page_path="p", title="t", snippet="s")
        assert item.score == 0
        assert item.section == ""
        assert item.chunk_index is None
        assert item.citation_ref == "p"

    def test_rejects_wrong_type_strict(self):
        with pytest.raises(ValidationError):
            EvidenceItem(page_path=123, title="t", snippet="s")

    def test_rejects_missing_required(self):
        with pytest.raises(ValidationError):
            EvidenceItem(title="t", snippet="s")


# ── EvidenceBundle ───────────────────────────────────────────────────


class TestEvidenceBundle:
    def test_positive_create(self):
        bundle = EvidenceBundle(
            question="What is X?",
            items=[EvidenceItem(page_path="p", title="t", snippet="s")],
        )
        assert bundle.question == "What is X?"
        assert len(bundle.items) == 1

    def test_empty_items_default(self):
        bundle = EvidenceBundle(question="Q")
        assert bundle.items == []

    def test_context_hash_deterministic(self):
        bundle = EvidenceBundle(
            question="Q",
            items=[EvidenceItem(page_path="p", title="t", snippet="s")],
        )
        assert bundle.context_hash == bundle.context_hash

    def test_context_hash_changes_with_question(self):
        a = EvidenceBundle(question="A")
        b = EvidenceBundle(question="B")
        assert a.context_hash != b.context_hash

    def test_context_hash_changes_with_items(self):
        a = EvidenceBundle(
            question="Q",
            items=[EvidenceItem(page_path="p1", title="t", snippet="s")],
        )
        b = EvidenceBundle(
            question="Q",
            items=[EvidenceItem(page_path="p2", title="t", snippet="s")],
        )
        assert a.context_hash != b.context_hash

    def test_context_hash_changes_with_chunk_metadata(self):
        a = EvidenceBundle(
            question="Q",
            items=[
                EvidenceItem(
                    page_path="p",
                    title="t",
                    snippet="s",
                    section="Intro",
                    chunk_index=0,
                )
            ],
        )
        b = EvidenceBundle(
            question="Q",
            items=[
                EvidenceItem(
                    page_path="p",
                    title="t",
                    snippet="s",
                    section="Intro",
                    chunk_index=1,
                )
            ],
        )
        assert a.context_hash != b.context_hash

    def test_context_hash_length(self):
        bundle = EvidenceBundle(question="Q")
        assert len(bundle.context_hash) == 16


# ── Claim ────────────────────────────────────────────────────────────


class TestClaim:
    def test_positive_create(self):
        claim = Claim(text="X is true", source_page="p.md", section="## S")
        assert claim.text == "X is true"
        assert claim.grounded is True

    def test_defaults(self):
        claim = Claim(text="X")
        assert claim.source_page == ""
        assert claim.section == ""
        assert claim.confidence == 1.0
        assert claim.grounded is True

    def test_rejects_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            Claim(text="X", confidence=1.5)

    def test_rejects_confidence_below_zero(self):
        with pytest.raises(ValidationError):
            Claim(text="X", confidence=-0.1)

    def test_rejects_missing_text(self):
        with pytest.raises(ValidationError):
            Claim(source_page="p")

    def test_boundary_confidence_zero(self):
        claim = Claim(text="X", confidence=0.0)
        assert claim.confidence == 0.0

    def test_boundary_confidence_one(self):
        claim = Claim(text="X", confidence=1.0)
        assert claim.confidence == 1.0


# ── CandidateAnswer ─────────────────────────────────────────────────


class TestCandidateAnswer:
    def test_positive_create(self):
        ca = CandidateAnswer(
            raw_text="Answer text",
            claims=[Claim(text="C1")],
            model_name="gpt-5.4-mini",
            latency_ms=120,
        )
        assert ca.raw_text == "Answer text"
        assert len(ca.claims) == 1

    def test_defaults(self):
        ca = CandidateAnswer(raw_text="A")
        assert ca.claims == []
        assert ca.model_name == ""
        assert ca.latency_ms == 0
        assert ca.token_usage is None
        assert ca.error is None

    def test_with_error(self):
        ca = CandidateAnswer(raw_text="", error="Provider timeout")
        assert ca.error == "Provider timeout"

    def test_rejects_wrong_type(self):
        with pytest.raises(ValidationError):
            CandidateAnswer(raw_text=42)


# ── MergedAnswer ─────────────────────────────────────────────────────


class TestMergedAnswer:
    def test_positive_create(self):
        ma = MergedAnswer(
            text="Final",
            accepted_claims=[Claim(text="C1")],
            dropped_claims=[Claim(text="C2", grounded=False)],
            candidate_count=3,
        )
        assert ma.text == "Final"
        assert len(ma.accepted_claims) == 1
        assert len(ma.dropped_claims) == 1
        assert ma.candidate_count == 3

    def test_defaults(self):
        ma = MergedAnswer(text="F")
        assert ma.accepted_claims == []
        assert ma.dropped_claims == []
        assert ma.candidate_count == 0


# ── Verdict ──────────────────────────────────────────────────────────


class TestVerdict:
    def test_enum_values(self):
        assert Verdict.CONSISTENT.value == "consistent"
        assert Verdict.CONTRADICTORY.value == "contradictory"
        assert Verdict.TERM_DRIFT.value == "term_drift"
        assert Verdict.NEEDS_REVIEW.value == "needs_review"

    def test_string_comparison(self):
        assert Verdict.CONSISTENT == "consistent"


# ── ReviewFinding ────────────────────────────────────────────────────


class TestReviewFinding:
    def test_positive_create(self):
        rf = ReviewFinding(
            issue_type="contradiction",
            affected_pages=["a.md", "b.md"],
            claim="X contradicts Y",
            evidence_for="Evidence A",
            evidence_against="Evidence B",
            verdict=Verdict.CONTRADICTORY,
            confidence=0.9,
            citations=["a.md#s1"],
        )
        assert rf.verdict == Verdict.CONTRADICTORY
        assert rf.confidence == 0.9

    def test_defaults(self):
        rf = ReviewFinding(issue_type="test")
        assert rf.affected_pages == []
        assert rf.verdict == Verdict.NEEDS_REVIEW
        assert rf.confidence == 0.5

    def test_rejects_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            ReviewFinding(issue_type="t", confidence=2.0)


# ── RunRecord ────────────────────────────────────────────────────────


class TestRunRecord:
    def test_positive_create(self):
        rr = RunRecord(command="query")
        assert rr.command == "query"
        assert len(rr.run_id) == 12
        assert rr.timestamp != ""
        assert rr.unresolved_disagreement is False

    def test_auto_generated_fields(self):
        a = RunRecord()
        b = RunRecord()
        assert a.run_id != b.run_id
        assert a.timestamp != "" and b.timestamp != ""

    def test_defaults(self):
        rr = RunRecord()
        assert rr.model_id == ""
        assert rr.candidates == []
        assert rr.merged_answer is None
        assert rr.review_findings == []
        assert rr.token_cost == 0

    def test_full_record_with_evidence(self):
        rr = RunRecord(
            command="query",
            model_id="gpt-5.4-mini",
            evidence_bundle=EvidenceBundle(
                question="Q",
                items=[EvidenceItem(page_path="p", title="t", snippet="s")],
            ),
            candidates=[
                CandidateAnswer(
                    raw_text="A",
                    claims=[Claim(text="C1", source_page="p")],
                    model_name="gpt-5.4-mini",
                )
            ],
            merged_answer=MergedAnswer(
                text="Final",
                accepted_claims=[Claim(text="C1", source_page="p")],
                candidate_count=1,
            ),
            final_text="Final",
            token_cost=500,
            wall_time_ms=1200,
        )
        assert rr.evidence_bundle is not None
        assert rr.evidence_bundle.context_hash != ""
        assert len(rr.candidates) == 1

    def test_json_roundtrip(self):
        rr = RunRecord(
            command="review",
            review_findings=[
                ReviewFinding(
                    issue_type="contradiction",
                    verdict=Verdict.CONTRADICTORY,
                )
            ],
        )
        json_str = rr.model_dump_json()
        restored = RunRecord.model_validate_json(json_str)
        assert restored.run_id == rr.run_id
        assert restored.command == "review"
        assert len(restored.review_findings) == 1
        assert restored.review_findings[0].verdict == Verdict.CONTRADICTORY

    def test_json_roundtrip_full(self):
        rr = RunRecord(
            command="query",
            evidence_bundle=EvidenceBundle(question="Q"),
            candidates=[CandidateAnswer(raw_text="A")],
            merged_answer=MergedAnswer(text="F", candidate_count=1),
        )
        json_str = rr.model_dump_json()
        restored = RunRecord.model_validate_json(json_str)
        assert restored.evidence_bundle is not None
        assert restored.evidence_bundle.question == "Q"
        assert len(restored.candidates) == 1
        assert restored.merged_answer is not None
        assert restored.merged_answer.text == "F"

    def test_rejects_wrong_type_strict(self):
        with pytest.raises(ValidationError):
            RunRecord(token_cost="not_an_int")
