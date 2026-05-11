from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from src.services.graphrag_query_service import GRAPH_QUERY_METHODS
from src.services.graphrag_status_service import GraphRAGStatusService
from src.services.project_service import build_project_paths, utc_now_iso
from src.services.query_router_service import QueryRouterService


RESULTS_DIR = Path("eval") / "results"
ARTIFACTS_DIR = "artifacts"
RETRIEVAL_COLUMNS = (
    "question_id",
    "question",
    "retriever",
    "method",
    "status",
    "expected_method",
    "routed_method",
    "method_fit",
    "expected_source_count",
    "matched_source_count",
    "recall_at_5",
    "multi_source_coverage",
    "latency_seconds",
    "artifact_path",
    "error",
)
ANSWER_COLUMNS = (
    "question_id",
    "question",
    "retriever",
    "method",
    "status",
    "claim_support",
    "claim_support_rate",
    "insufficient_evidence_expected",
    "insufficient_evidence_observed",
    "insufficient_evidence_behavior",
    "comprehensiveness",
    "diversity",
    "latency_seconds",
    "artifact_path",
    "error",
)


@dataclass(frozen=True)
class BenchmarkQuestion:
    id: str
    question: str
    intent: str
    category: str
    expected_method: str | None
    expected_sources: tuple[str, ...]
    expected_behaviors: tuple[str, ...]
    notes: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkQuestion":
        return cls(
            id=str(payload["id"]),
            question=str(payload["question"]),
            intent=str(payload.get("intent", "")),
            category=str(payload.get("category", "unspecified")),
            expected_method=_optional_str(payload.get("expected_method")),
            expected_sources=tuple(
                str(item) for item in payload.get("expected_sources", [])
            ),
            expected_behaviors=tuple(
                str(item) for item in payload.get("expected_behaviors", [])
            ),
            notes=_optional_str(payload.get("notes")),
        )


@dataclass(frozen=True)
class Benchmark:
    path: Path
    version: int
    name: str
    description: str
    source_project: str | None
    questions: tuple[BenchmarkQuestion, ...]
    notes: tuple[str, ...]

    @property
    def root(self) -> Path:
        return self.path.parent.parent


@dataclass(frozen=True)
class CommandRun:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    latency_seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "command": list(self.command),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "latency_seconds": round(self.latency_seconds, 4),
        }


@dataclass(frozen=True)
class EvaluationConfig:
    benchmark_path: Path
    project_root: Path | None
    results_dir: Path
    limit: int
    allow_provider_calls: bool
    include_legacy_ask: bool
    graph_methods: tuple[str, ...]
    timeout_seconds: int


@dataclass(frozen=True)
class EvaluationResult:
    benchmark: Benchmark
    project_root: Path | None
    generated_at: str
    retrieval_rows: list[dict[str, object]]
    answer_rows: list[dict[str, object]]


def load_benchmark(path: Path) -> Benchmark:
    resolved_path = path.resolve()
    payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark file is not a mapping: {path}")
    question_payloads = payload.get("questions")
    if not isinstance(question_payloads, list) or not question_payloads:
        raise ValueError("Benchmark must include a non-empty questions list.")
    return Benchmark(
        path=resolved_path,
        version=int(payload.get("version", 1)),
        name=str(payload.get("name", path.stem)),
        description=str(payload.get("description", "")),
        source_project=_optional_str(payload.get("source_project")),
        questions=tuple(
            BenchmarkQuestion.from_dict(item)
            for item in question_payloads
            if isinstance(item, dict)
        ),
        notes=tuple(str(item) for item in payload.get("notes", [])),
    )


def resolve_project_root(
    benchmark: Benchmark, explicit_project_root: Path | None
) -> Path | None:
    if explicit_project_root is not None:
        return explicit_project_root.resolve()
    if not benchmark.source_project:
        return None
    source_path = Path(benchmark.source_project)
    if source_path.is_absolute():
        return source_path
    candidates = (
        benchmark.root / source_path,
        benchmark.root.parent / source_path,
        benchmark.path.parent / source_path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (benchmark.root / source_path).resolve()


def run_graph_modes_evaluation(config: EvaluationConfig) -> EvaluationResult:
    benchmark = load_benchmark(config.benchmark_path)
    project_root = resolve_project_root(benchmark, config.project_root)
    command_cwd = benchmark.root
    generated_at = utc_now_iso()
    retrieval_rows: list[dict[str, object]] = []
    answer_rows: list[dict[str, object]] = []
    for question in benchmark.questions:
        retrieval_rows.append(evaluate_auto_route(question, project_root=project_root))
        retrieval_rows.append(
            evaluate_legacy_find(
                question,
                project_root=project_root,
                command_cwd=command_cwd,
                results_dir=config.results_dir,
                limit=config.limit,
                timeout_seconds=config.timeout_seconds,
            )
        )
        if config.include_legacy_ask:
            answer_rows.append(
                evaluate_legacy_ask(
                    question,
                    project_root=project_root,
                    command_cwd=command_cwd,
                    results_dir=config.results_dir,
                    allow_provider_calls=config.allow_provider_calls,
                    timeout_seconds=config.timeout_seconds,
                )
            )
        for method in config.graph_methods:
            answer_rows.append(
                evaluate_graph_method(
                    question,
                    method=method,
                    project_root=project_root,
                    command_cwd=command_cwd,
                    results_dir=config.results_dir,
                    allow_provider_calls=config.allow_provider_calls,
                    timeout_seconds=config.timeout_seconds,
                )
            )
    result = EvaluationResult(
        benchmark=benchmark,
        project_root=project_root,
        generated_at=generated_at,
        retrieval_rows=retrieval_rows,
        answer_rows=answer_rows,
    )
    write_results(
        result, config.results_dir, allow_provider_calls=config.allow_provider_calls
    )
    return result


def evaluate_auto_route(
    question: BenchmarkQuestion, *, project_root: Path | None
) -> dict[str, object]:
    expected_method = question.expected_method or ""
    try:
        status_service = (
            GraphRAGStatusService(build_project_paths(project_root))
            if project_root is not None and project_root.exists()
            else None
        )
        route = QueryRouterService(status_service).route(question.question)
        status = "ok"
        error = ""
        routed_method = route.method
    except Exception as exc:
        status = "failed"
        error = _short_error(exc)
        routed_method = ""
    return _retrieval_row(
        question,
        retriever="graph-auto-router",
        method="auto",
        status=status,
        expected_method=expected_method,
        routed_method=routed_method,
        method_fit=_bool_metric(routed_method == expected_method)
        if expected_method and routed_method
        else "",
        expected_source_count=len(question.expected_sources),
        matched_source_count="",
        recall_at_5="",
        multi_source_coverage="",
        latency_seconds=0,
        artifact_path="",
        error=error,
    )


def evaluate_legacy_find(
    question: BenchmarkQuestion,
    *,
    project_root: Path | None,
    command_cwd: Path,
    results_dir: Path,
    limit: int,
    timeout_seconds: int,
) -> dict[str, object]:
    if project_root is None:
        return _skipped_retrieval_row(
            question,
            retriever="legacy-fts",
            method="find",
            status="skipped_no_project_root",
        )
    command = _kb_command(
        project_root,
        "legacy",
        "find",
        "--limit",
        str(limit),
        "--json",
        question.question,
    )
    run = run_command(command, cwd=command_cwd, timeout_seconds=timeout_seconds)
    artifact_path = write_artifact(
        results_dir,
        question.id,
        "legacy_find.json",
        {
            "question": question.question,
            "run": run.to_dict(),
            "payload": _json_from_stdout(run.stdout),
        },
    )
    if not run.ok:
        return _retrieval_row_for_run(
            question,
            retriever="legacy-fts",
            method="find",
            status="failed",
            run=run,
            artifact_path=artifact_path,
            error=_command_error(run),
        )
    payload = _json_from_stdout(run.stdout)
    if not isinstance(payload, dict):
        return _retrieval_row_for_run(
            question,
            retriever="legacy-fts",
            method="find",
            status="failed",
            run=run,
            artifact_path=artifact_path,
            error="Could not parse legacy find JSON output.",
        )
    results = payload.get("results", [])
    if not isinstance(results, list):
        results = []
    source_score = score_expected_source_coverage(
        question.expected_sources,
        results[:limit],
    )
    return _retrieval_row(
        question,
        retriever="legacy-fts",
        method="find",
        status="ok",
        expected_method=question.expected_method or "",
        routed_method="",
        method_fit="",
        expected_source_count=source_score["expected_count"],
        matched_source_count=source_score["matched_count"],
        recall_at_5=source_score["recall"],
        multi_source_coverage=source_score["multi_source_coverage"],
        latency_seconds=run.latency_seconds,
        artifact_path=artifact_path,
        error="",
    )


def evaluate_legacy_ask(
    question: BenchmarkQuestion,
    *,
    project_root: Path | None,
    command_cwd: Path,
    results_dir: Path,
    allow_provider_calls: bool,
    timeout_seconds: int,
) -> dict[str, object]:
    if not allow_provider_calls:
        return _skipped_answer_row(
            question,
            retriever="legacy-fts",
            method="ask",
            status="skipped_provider_call",
        )
    if project_root is None:
        return _skipped_answer_row(
            question,
            retriever="legacy-fts",
            method="ask",
            status="skipped_no_project_root",
        )
    command = _kb_command(project_root, "legacy", "ask", question.question)
    run = run_command(command, cwd=command_cwd, timeout_seconds=timeout_seconds)
    artifact_path = write_artifact(
        results_dir,
        question.id,
        "legacy_ask.json",
        {"question": question.question, "run": run.to_dict()},
    )
    if not run.ok:
        return _answer_row_for_run(
            question,
            retriever="legacy-fts",
            method="ask",
            status="failed",
            run=run,
            artifact_path=artifact_path,
            error=_command_error(run),
        )
    return score_answer_row(
        question,
        retriever="legacy-fts",
        method="ask",
        status="ok",
        answer_text=run.stdout,
        claim_support="legacy-citation-validated",
        latency_seconds=run.latency_seconds,
        artifact_path=artifact_path,
        error="",
    )


def evaluate_graph_method(
    question: BenchmarkQuestion,
    *,
    method: str,
    project_root: Path | None,
    results_dir: Path,
    allow_provider_calls: bool,
    timeout_seconds: int,
    command_cwd: Path | None = None,
) -> dict[str, object]:
    if method not in GRAPH_QUERY_METHODS:
        raise ValueError(f"Unsupported GraphRAG method: {method}")
    if not allow_provider_calls:
        return _skipped_answer_row(
            question,
            retriever="graphrag",
            method=method,
            status="skipped_provider_call",
        )
    if project_root is None:
        return _skipped_answer_row(
            question,
            retriever="graphrag",
            method=method,
            status="skipped_no_project_root",
        )
    command = _kb_command(
        project_root,
        "graph",
        "ask",
        "--method",
        method,
        "--json",
        question.question,
    )
    run = run_command(
        command,
        cwd=command_cwd or project_root,
        timeout_seconds=timeout_seconds,
    )
    payload = _json_from_stdout(run.stdout)
    artifact_path = write_artifact(
        results_dir,
        question.id,
        f"graphrag_{method}.json",
        {"question": question.question, "run": run.to_dict(), "payload": payload},
    )
    if not run.ok:
        return _answer_row_for_run(
            question,
            retriever="graphrag",
            method=method,
            status="failed",
            run=run,
            artifact_path=artifact_path,
            error=_command_error(run),
        )
    answer_text = ""
    claim_support = "graph-output"
    if isinstance(payload, dict):
        answer_text = str(payload.get("answer") or payload.get("raw_output") or "")
        claim_support = str(payload.get("claim_support") or "graph-output")
    else:
        answer_text = run.stdout
    return score_answer_row(
        question,
        retriever="graphrag",
        method=method,
        status="ok",
        answer_text=answer_text,
        claim_support=claim_support,
        latency_seconds=run.latency_seconds,
        artifact_path=artifact_path,
        error="",
    )


def run_command(
    command: Sequence[str], *, cwd: Path, timeout_seconds: int
) -> CommandRun:
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return CommandRun(
        command=tuple(command),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        latency_seconds=time.perf_counter() - started,
    )


def score_expected_source_coverage(
    expected_sources: Sequence[str], result_payloads: Sequence[object]
) -> dict[str, object]:
    expected = [_normalize_identifier(value) for value in expected_sources if value]
    if not expected:
        return {
            "expected_count": 0,
            "matched_count": "",
            "recall": "",
            "multi_source_coverage": "",
        }
    searchable_results = [_searchable_text(result) for result in result_payloads]
    matched = 0
    for expected_source in expected:
        if any(expected_source in searchable for searchable in searchable_results):
            matched += 1
    recall = matched / len(expected)
    return {
        "expected_count": len(expected),
        "matched_count": matched,
        "recall": round(recall, 4),
        "multi_source_coverage": _bool_metric(matched == len(expected))
        if len(expected) > 1
        else "",
    }


def score_answer_row(
    question: BenchmarkQuestion,
    *,
    retriever: str,
    method: str,
    status: str,
    answer_text: str,
    claim_support: str,
    latency_seconds: float,
    artifact_path: str,
    error: str,
) -> dict[str, object]:
    expected_insufficient = "insufficient_evidence" in question.expected_behaviors
    observed_insufficient = _mentions_insufficient_evidence(answer_text)
    expected_coverage = score_expected_source_coverage(
        question.expected_sources,
        [{"answer": answer_text}],
    )
    return _answer_row(
        question,
        retriever=retriever,
        method=method,
        status=status,
        claim_support=claim_support,
        claim_support_rate=_claim_support_rate(claim_support),
        insufficient_evidence_expected=_bool_metric(expected_insufficient),
        insufficient_evidence_observed=_bool_metric(observed_insufficient),
        insufficient_evidence_behavior=_bool_metric(
            observed_insufficient == expected_insufficient
        )
        if expected_insufficient
        else "",
        comprehensiveness=_comprehensiveness_score(answer_text),
        diversity=expected_coverage["recall"],
        latency_seconds=latency_seconds,
        artifact_path=artifact_path,
        error=error,
    )


def write_results(
    result: EvaluationResult, results_dir: Path, *, allow_provider_calls: bool
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        results_dir / "retrieval_metrics.csv", RETRIEVAL_COLUMNS, result.retrieval_rows
    )
    write_csv(results_dir / "answer_metrics.csv", ANSWER_COLUMNS, result.answer_rows)
    (results_dir / "summary.md").write_text(
        render_summary(result, allow_provider_calls=allow_provider_calls),
        encoding="utf-8",
    )


def write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_artifact(
    results_dir: Path, question_id: str, filename: str, payload: dict[str, object]
) -> str:
    artifact_path = results_dir / ARTIFACTS_DIR / question_id / filename
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        return artifact_path.relative_to(results_dir.parent.parent).as_posix()
    except ValueError:
        return artifact_path.as_posix()


def render_summary(result: EvaluationResult, *, allow_provider_calls: bool) -> str:
    retrieval_status = _status_counts(result.retrieval_rows)
    answer_status = _status_counts(result.answer_rows)
    average_recall = _average_numeric(
        row.get("recall_at_5") for row in result.retrieval_rows
    )
    method_fit = _average_numeric(
        row.get("method_fit") for row in result.retrieval_rows
    )
    claim_support = _average_numeric(
        row.get("claim_support_rate") for row in result.answer_rows
    )
    project = (
        _display_path(result.project_root, result.benchmark.root)
        if result.project_root
        else "not set"
    )
    provider_mode = (
        "enabled; GraphRAG and provider-backed answer commands were allowed"
        if allow_provider_calls
        else "disabled; provider-backed answer commands were skipped"
    )
    return (
        "# GraphRAG Evaluation Summary\n\n"
        f"- Generated at: {result.generated_at}\n"
        f"- Benchmark: {result.benchmark.name} v{result.benchmark.version}\n"
        f"- Project root: `{project}`\n"
        f"- Provider calls: {provider_mode}\n"
        f"- Questions: {len(result.benchmark.questions)}\n"
        f"- Retrieval rows: {len(result.retrieval_rows)} ({retrieval_status})\n"
        f"- Answer rows: {len(result.answer_rows)} ({answer_status})\n"
        f"- Average Recall@5: {_metric_text(average_recall)}\n"
        f"- Auto-router method fit: {_metric_text(method_fit)}\n"
        f"- Claim support rate: {_metric_text(claim_support)}\n\n"
        "## Outputs\n\n"
        "- `eval/results/retrieval_metrics.csv`\n"
        "- `eval/results/answer_metrics.csv`\n"
        "- `eval/results/artifacts/<question-id>/...`\n\n"
        "Run with `--allow-provider-calls` only when the configured GraphRAG "
        "provider/API key is ready and you explicitly want to spend model and "
        "embedding/query budget.\n"
    )


def _kb_command(project_root: Path, *parts: str) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "src.cli",
        "--project-root",
        str(project_root),
        *parts,
    )


def _retrieval_row_for_run(
    question: BenchmarkQuestion,
    *,
    retriever: str,
    method: str,
    status: str,
    run: CommandRun,
    artifact_path: str,
    error: str,
) -> dict[str, object]:
    return _retrieval_row(
        question,
        retriever=retriever,
        method=method,
        status=status,
        expected_method=question.expected_method or "",
        routed_method="",
        method_fit="",
        expected_source_count=len(question.expected_sources),
        matched_source_count="",
        recall_at_5="",
        multi_source_coverage="",
        latency_seconds=run.latency_seconds,
        artifact_path=artifact_path,
        error=error,
    )


def _answer_row_for_run(
    question: BenchmarkQuestion,
    *,
    retriever: str,
    method: str,
    status: str,
    run: CommandRun,
    artifact_path: str,
    error: str,
) -> dict[str, object]:
    return _answer_row(
        question,
        retriever=retriever,
        method=method,
        status=status,
        claim_support="",
        claim_support_rate="",
        insufficient_evidence_expected=_bool_metric(
            "insufficient_evidence" in question.expected_behaviors
        ),
        insufficient_evidence_observed="",
        insufficient_evidence_behavior="",
        comprehensiveness="",
        diversity="",
        latency_seconds=run.latency_seconds,
        artifact_path=artifact_path,
        error=error,
    )


def _skipped_retrieval_row(
    question: BenchmarkQuestion, *, retriever: str, method: str, status: str
) -> dict[str, object]:
    return _retrieval_row(
        question,
        retriever=retriever,
        method=method,
        status=status,
        expected_method=question.expected_method or "",
        routed_method="",
        method_fit="",
        expected_source_count=len(question.expected_sources),
        matched_source_count="",
        recall_at_5="",
        multi_source_coverage="",
        latency_seconds="",
        artifact_path="",
        error="",
    )


def _skipped_answer_row(
    question: BenchmarkQuestion, *, retriever: str, method: str, status: str
) -> dict[str, object]:
    return _answer_row(
        question,
        retriever=retriever,
        method=method,
        status=status,
        claim_support="",
        claim_support_rate="",
        insufficient_evidence_expected=_bool_metric(
            "insufficient_evidence" in question.expected_behaviors
        ),
        insufficient_evidence_observed="",
        insufficient_evidence_behavior="",
        comprehensiveness="",
        diversity="",
        latency_seconds="",
        artifact_path="",
        error="",
    )


def _retrieval_row(
    question: BenchmarkQuestion,
    *,
    retriever: str,
    method: str,
    status: str,
    expected_method: str,
    routed_method: str,
    method_fit: object,
    expected_source_count: object,
    matched_source_count: object,
    recall_at_5: object,
    multi_source_coverage: object,
    latency_seconds: object,
    artifact_path: str,
    error: str,
) -> dict[str, object]:
    return {
        "question_id": question.id,
        "question": question.question,
        "retriever": retriever,
        "method": method,
        "status": status,
        "expected_method": expected_method,
        "routed_method": routed_method,
        "method_fit": method_fit,
        "expected_source_count": expected_source_count,
        "matched_source_count": matched_source_count,
        "recall_at_5": recall_at_5,
        "multi_source_coverage": multi_source_coverage,
        "latency_seconds": _round_latency(latency_seconds),
        "artifact_path": artifact_path,
        "error": error,
    }


def _answer_row(
    question: BenchmarkQuestion,
    *,
    retriever: str,
    method: str,
    status: str,
    claim_support: str,
    claim_support_rate: object,
    insufficient_evidence_expected: object,
    insufficient_evidence_observed: object,
    insufficient_evidence_behavior: object,
    comprehensiveness: object,
    diversity: object,
    latency_seconds: object,
    artifact_path: str,
    error: str,
) -> dict[str, object]:
    return {
        "question_id": question.id,
        "question": question.question,
        "retriever": retriever,
        "method": method,
        "status": status,
        "claim_support": claim_support,
        "claim_support_rate": claim_support_rate,
        "insufficient_evidence_expected": insufficient_evidence_expected,
        "insufficient_evidence_observed": insufficient_evidence_observed,
        "insufficient_evidence_behavior": insufficient_evidence_behavior,
        "comprehensiveness": comprehensiveness,
        "diversity": diversity,
        "latency_seconds": _round_latency(latency_seconds),
        "artifact_path": artifact_path,
        "error": error,
    }


def _json_from_stdout(stdout: str) -> object:
    text = stdout.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _searchable_text(payload: object) -> str:
    values: list[str] = []
    _collect_strings(payload, values)
    return _normalize_identifier(" ".join(values))


def _collect_strings(payload: object, values: list[str]) -> None:
    if isinstance(payload, dict):
        for value in payload.values():
            _collect_strings(value, values)
    elif isinstance(payload, list):
        for value in payload:
            _collect_strings(value, values)
    elif payload is not None:
        values.append(str(payload))


def _normalize_identifier(value: str) -> str:
    lowered = value.casefold()
    normalized = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(normalized.split())


def _mentions_insufficient_evidence(answer_text: str) -> bool:
    normalized = answer_text.casefold()
    markers = (
        "insufficient evidence",
        "not enough evidence",
        "cannot determine",
        "can't determine",
        "not stated",
        "not available",
    )
    return any(marker in normalized for marker in markers)


def _claim_support_rate(claim_support: str) -> object:
    normalized = claim_support.casefold()
    if not normalized:
        return ""
    if normalized in {"graph-grounded", "graph-output", "legacy-citation-validated"}:
        return 1.0
    if normalized == "stale-index":
        return 0.5
    if normalized == "no-answer":
        return 0.0
    return ""


def _comprehensiveness_score(answer_text: str) -> object:
    words = re.findall(r"\w+", answer_text)
    if not words:
        return 0.0
    return round(min(1.0, len(words) / 180), 4)


def _status_counts(rows: Iterable[dict[str, object]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return ", ".join(f"{status}: {count}" for status, count in sorted(counts.items()))


def _average_numeric(values: Iterable[object]) -> float | None:
    numeric_values: list[float] = []
    for value in values:
        if value == "" or value is None:
            continue
        try:
            numeric_values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _metric_text(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _display_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        pass
    try:
        return f"../{resolved.relative_to(repo_root.resolve().parent).as_posix()}"
    except ValueError:
        return resolved.as_posix()


def _bool_metric(value: bool) -> int:
    return 1 if value else 0


def _round_latency(value: object) -> object:
    if value == "" or value is None:
        return ""
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _command_error(run: CommandRun) -> str:
    return (run.stderr.strip() or run.stdout.strip())[:1000]


def _short_error(exc: Exception) -> str:
    return str(exc)[:1000]
