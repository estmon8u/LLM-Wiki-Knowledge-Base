"""Aggregation with bootstrap confidence intervals + leaderboard rendering.

Per-(backend) metric means are reported with **bootstrap 95% CIs** so that
within-noise differences between backends are not mistaken for real wins. No
single gameable composite is used as the headline — every metric is reported
on its own.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricSummary:
    """Bootstrap summary of one metric for one backend."""

    backend: str
    metric: str
    n: int
    mean: float
    ci_low: float
    ci_high: float


def bootstrap_ci(
    values: list[float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` via percentile bootstrap."""
    if not values:
        return (0.0, 0.0, 0.0)
    mean = sum(values) / len(values)
    if len(values) == 1:
        return (mean, mean, mean)
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (mean, lo, hi)


def summarize(
    rows: list[dict[str, Any]],
    metric_keys: list[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> list[MetricSummary]:
    """Summarize per-(backend, metric) with bootstrap CIs over questions.

    ``None`` metric values are skipped (e.g. retrieval metrics for questions
    without ground truth), so each metric averages only where it applies.
    """
    backends: list[str] = []
    for row in rows:
        if row["backend"] not in backends:
            backends.append(row["backend"])
    summaries: list[MetricSummary] = []
    for backend in backends:
        backend_rows = [r for r in rows if r["backend"] == backend]
        for metric in metric_keys:
            values = [
                float(r[metric]) for r in backend_rows if r.get(metric) is not None
            ]
            if not values:
                continue
            mean, lo, hi = bootstrap_ci(values, n_boot=n_boot, seed=seed)
            summaries.append(
                MetricSummary(
                    backend=backend,
                    metric=metric,
                    n=len(values),
                    mean=mean,
                    ci_low=lo,
                    ci_high=hi,
                )
            )
    return summaries


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the raw per-(backend, question) metric rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_leaderboard_markdown(
    path: Path,
    summaries: list[MetricSummary],
    *,
    metric_order: list[str],
    title: str = "RAG evaluation leaderboard",
    notes: list[str] | None = None,
) -> None:
    """Write a per-backend leaderboard (one column per metric, mean [lo, hi])."""
    path.parent.mkdir(parents=True, exist_ok=True)
    backends: list[str] = []
    for entry in summaries:
        if entry.backend not in backends:
            backends.append(entry.backend)
    by_key = {(s.backend, s.metric): s for s in summaries}
    present_metrics = [
        m for m in metric_order if any((b, m) in by_key for b in backends)
    ]

    lines: list[str] = [f"# {title}", ""]
    for note in notes or []:
        lines.append(f"> {note}")
    if notes:
        lines.append("")
    header = "| Backend | " + " | ".join(present_metrics) + " |"
    sep = "|---|" + "|".join("---" for _ in present_metrics) + "|"
    lines.append(header)
    lines.append(sep)
    for backend in backends:
        cells: list[str] = []
        for metric in present_metrics:
            summary = by_key.get((backend, metric))
            if summary is None:
                cells.append("n/a")
            else:
                cells.append(
                    f"{summary.mean:.3f} "
                    f"[{summary.ci_low:.2f},{summary.ci_high:.2f}] (n={summary.n})"
                )
        lines.append(f"| {backend} | " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
