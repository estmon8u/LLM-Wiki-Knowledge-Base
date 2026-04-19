from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_smoke_script_runs_clean_on_temporary_corpus(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_root = tmp_path / "raw"
    project_root = tmp_path / "project"
    script_path = repo_root / "scripts" / "run_real_corpus_smoke.py"

    raw_root.mkdir()
    (raw_root / "sample.md").write_text(
        "# Sample\n\n" "A knowledge base wiki can accumulate information over time.\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--raw-root",
            str(raw_root),
            "--project-root",
            str(project_root),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        errors="replace",
    )

    assert result.returncode == 0
    log_text = (project_root / "command-smoke-log.txt").read_text(encoding="utf-8")
    assert "python -m src.cli --help" in log_text
    assert "python -m src.cli --project-root" in log_text
    assert "export" in log_text
