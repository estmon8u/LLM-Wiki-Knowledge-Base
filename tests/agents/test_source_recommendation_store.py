"""Tests for the SourceRecommendationStore JSON store."""

from __future__ import annotations

import pytest

from graphwiki_kb.agents.models import (
    ResearchRunRecord,
    SourceRecommendation,
    WebFinding,
)
from graphwiki_kb.services.source_recommendation_store import (
    SourceRecommendationStore,
    SourceRecommendationStoreError,
)


def _make_record(*, run_id: str, recs: int = 2) -> ResearchRunRecord:
    return ResearchRunRecord(
        run_id=run_id,
        question="benchmark question",
        created_at="2026-05-19T12:34:56+00:00",
        local_answer={"answer": "..."},
        kb_gaps=["gap a"],
        web_findings=[
            WebFinding(
                title="t",
                url="https://example.com",
                summary="s",
                relevance="medium",
            )
        ],
        recommendations=[
            SourceRecommendation(
                id=i,
                title=f"Rec {i}",
                url=f"https://example.com/{i}",
                source_type="paper",
                retrieved_at="2026-05-19T12:34:56+00:00",
                why_add="r",
            )
            for i in range(1, recs + 1)
        ],
    )


def test_store_saves_run_and_writes_latest_pointer(test_project) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.ensure_directory()
    record = _make_record(run_id="research_20260519T123456Z_benchmark_question")
    path = store.save(record)
    assert path.exists()
    assert store.latest_pointer.exists()
    loaded = store.load("latest")
    assert loaded.run_id == record.run_id
    assert len(loaded.recommendations) == 2


def test_store_resolve_recommendations_by_id(test_project) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.ensure_directory()
    record = _make_record(run_id="research_20260519T120000Z_q", recs=3)
    store.save(record)

    loaded, resolved = store.resolve_recommendations([1, 3], run_id="latest")
    assert loaded.run_id == record.run_id
    assert [r.id for r in resolved] == [1, 3]


def test_store_resolve_recommendations_raises_for_missing_id(test_project) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.ensure_directory()
    store.save(_make_record(run_id="research_20260519T120000Z_q", recs=2))

    with pytest.raises(SourceRecommendationStoreError):
        store.resolve_recommendations([99], run_id="latest")


def test_store_list_runs_returns_chronological(test_project) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.ensure_directory()
    store.save(_make_record(run_id="research_20260101T120000Z_q1"))
    store.save(_make_record(run_id="research_20260201T120000Z_q2"))
    runs = store.list_runs()
    assert [r.run_id for r in runs] == [
        "research_20260101T120000Z_q1",
        "research_20260201T120000Z_q2",
    ]


def test_store_load_missing_run_raises(test_project) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.ensure_directory()
    with pytest.raises(SourceRecommendationStoreError):
        store.load("research_does_not_exist")


def test_store_latest_returns_none_when_empty(test_project) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.ensure_directory()
    assert store.latest() is None


def test_store_latest_pointer_with_missing_file_falls_back_to_list(
    test_project,
) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.ensure_directory()
    store.save(_make_record(run_id="research_20260101T120000Z_q"))
    store.latest_pointer.write_text(
        '{"run_id": "x", "path": "missing.json"}',
        encoding="utf-8",
    )
    record = store.latest()
    assert record is not None
    assert record.run_id == "research_20260101T120000Z_q"


def test_store_skips_malformed_files_when_listing(test_project) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.ensure_directory()
    store.save(_make_record(run_id="research_20260101T120000Z_q"))
    malformed = store.directory / "research-bad.json"
    malformed.write_text("not json", encoding="utf-8")
    runs = store.list_runs()
    assert len(runs) == 1


def test_store_resolve_recommendations_without_ids_returns_all(test_project) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.ensure_directory()
    store.save(_make_record(run_id="research_20260101T120000Z_q", recs=3))
    _record, recs = store.resolve_recommendations([])
    assert len(recs) == 3
    assert {r.id for r in recs} == {1, 2, 3}


def test_store_load_unreadable_file_raises(test_project) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.ensure_directory()
    path = store.directory / "research-bad.json"
    path.write_text("{not: valid json", encoding="utf-8")
    with pytest.raises(SourceRecommendationStoreError):
        store._load_path(path)  # type: ignore[attr-defined]


def test_generate_run_id_is_deterministic_and_includes_slug() -> None:
    run_id = SourceRecommendationStore.generate_run_id(
        "Recent RAG benchmarks",
        created_at="2026-05-19T12:34:56+00:00",
    )
    assert run_id.startswith("research_20260519T123456")
    assert "recent-rag-benchmarks" in run_id


def _empty_record(run_id: str) -> ResearchRunRecord:
    return ResearchRunRecord(
        run_id=run_id,
        question="empty",
        created_at="2026-05-19T12:34:56+00:00",
        local_answer={"answer": "..."},
        kb_gaps=[],
        web_findings=[],
        recommendations=[],
    )


def test_empty_run_does_not_overwrite_latest_pointer(test_project) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.ensure_directory()
    first = _make_record(run_id="research_20260101T120000Z_q", recs=2)
    store.save(first)
    assert store.latest_pointer.exists()

    # A later research call with no recommendations must not hide the
    # previous run's pointer.
    store.save(_empty_record("research_20260201T120000Z_empty"))

    latest = store.latest()
    assert latest is not None
    assert latest.run_id == first.run_id


def test_latest_with_recommendations_skips_empty_runs(test_project) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.ensure_directory()
    older = _make_record(run_id="research_20260101T120000Z_q", recs=1)
    store.save(older)
    store.save(_empty_record("research_20260201T120000Z_empty"))

    fallback = store.latest_with_recommendations()
    assert fallback is not None
    assert fallback.run_id == older.run_id


def test_latest_with_recommendations_returns_none_when_only_empty(
    test_project,
) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.ensure_directory()
    store.save(_empty_record("research_20260201T120000Z_empty"))
    assert store.latest_with_recommendations() is None


def test_resolve_recommendations_falls_back_when_latest_is_empty(
    test_project,
) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.ensure_directory()
    older = _make_record(run_id="research_20260101T120000Z_q", recs=2)
    store.save(older)
    # Write an empty research file directly so the timestamp newer-first
    # order matters (save() refuses to update latest.json for empty runs).
    empty_path = store.directory / "research-20260201T120000Z-empty.json"
    empty_path.write_text(
        _empty_record("research_20260201T120000Z_empty").model_dump_json(indent=2),
        encoding="utf-8",
    )

    record, resolved = store.resolve_recommendations([1, 2], run_id="latest")
    assert record.run_id == older.run_id
    assert [r.id for r in resolved] == [1, 2]


def test_resolve_recommendations_explicit_run_id_does_not_fall_back(
    test_project,
) -> None:
    store = SourceRecommendationStore(test_project.paths)
    store.ensure_directory()
    store.save(_make_record(run_id="research_20260101T120000Z_q", recs=2))
    empty_id = "research_20260201T120000Z_empty"
    store.save(_empty_record(empty_id))

    with pytest.raises(SourceRecommendationStoreError):
        # Requesting recommendations from the empty run by id must raise
        # rather than silently jumping to the older run.
        store.resolve_recommendations([1], run_id=empty_id)
