from __future__ import annotations

from pathlib import Path

import pytest

from src.models.source_models import RawSourceRecord
from src.storage.compile_run_store import CompileRunStore
from src.services.project_service import utc_now_iso


def _source(slug: str) -> RawSourceRecord:
    return RawSourceRecord(
        source_id=f"source-{slug}",
        slug=slug,
        title=slug.title(),
        origin=f"notes/{slug}.md",
        source_type="file",
        raw_path=f"raw/sources/{slug}.md",
        normalized_path=f"raw/normalized/{slug}.md",
        content_hash=f"hash-{slug}",
        ingested_at=utc_now_iso(),
    )


def test_compile_run_store_marks_failure_and_exposes_resume_candidate(
    tmp_path: Path,
) -> None:
    store = CompileRunStore(tmp_path / "compile_runs.json")
    alpha = _source("alpha")
    beta = _source("beta")

    run = store.start_run([alpha, beta], force=False)
    store.mark_source_compiled(run.run_id, alpha)
    failed = store.mark_failed(run.run_id, error="boom", failed_source=beta)

    assert failed.completed_source_slugs == ["alpha"]
    assert failed.pending_source_slugs == ["beta"]
    assert failed.failed_source_slug == "beta"
    assert store.resume_candidate() is not None


def test_compile_run_store_completed_run_clears_resume_candidate(
    tmp_path: Path,
) -> None:
    store = CompileRunStore(tmp_path / "compile_runs.json")
    alpha = _source("alpha")

    run = store.start_run([alpha], force=False)
    store.mark_source_compiled(run.run_id, alpha)
    completed = store.mark_completed(run.run_id)

    assert completed.status == "completed"
    assert completed.pending_source_slugs == []
    assert store.resume_candidate() is None


def test_compile_run_store_exposes_running_active_run_as_resume_candidate(
    tmp_path: Path,
) -> None:
    store = CompileRunStore(tmp_path / "compile_runs.json")

    run = store.start_run([_source("alpha")], force=False)
    candidate = store.resume_candidate()

    assert candidate is not None
    assert candidate.run_id == run.run_id
    assert candidate.status == "running"


def test_compile_run_store_active_run_and_clear_resume_candidate(
    tmp_path: Path,
) -> None:
    store = CompileRunStore(tmp_path / "compile_runs.json")

    assert store.active_run() is None

    run = store.start_run([_source("alpha")], force=False)
    active = store.active_run()
    assert active is not None
    assert active.run_id == run.run_id

    store.mark_failed(run.run_id, error="boom")
    assert store.resume_candidate() is not None

    store.clear_resume_candidate()

    assert store.resume_candidate() is None


def test_compile_run_store_raises_for_unknown_active_run(tmp_path: Path) -> None:
    store = CompileRunStore(tmp_path / "compile_runs.json")

    with pytest.raises(ValueError, match="Compile run is not active"):
        store.mark_completed("missing-run")


def test_compile_run_store_archives_interrupted_active_run_on_new_start(
    tmp_path: Path,
) -> None:
    store = CompileRunStore(tmp_path / "compile_runs.json")
    store.start_run([_source("alpha")], force=False)

    new_run = store.start_run([_source("beta")], force=True)
    history = store.load_history()

    assert len(history) == 1
    assert history[0].status == "failed"
    assert history[0].error == "Interrupted before completion."
    assert new_run.status == "running"
