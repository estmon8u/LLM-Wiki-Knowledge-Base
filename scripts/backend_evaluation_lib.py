"""Three-backend evaluation helpers: legacy, GraphRAG, WikiGraphRAG."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from scripts.evaluation_lib import (
    ARTIFACTS_DIR,
    BenchmarkQuestion,
    EvaluationResult,
    evaluate_auto_route,
    evaluate_graph_method,
    evaluate_legacy_ask,
    evaluate_legacy_find,
    load_benchmark,
    resolve_project_root,
    run_command,
    score_answer_row,
    score_expected_source_coverage,
    utc_now_iso,
    write_results,
)

WIKIGRAPH_METHODS = ("basic", "local", "global", "drift-lite")
BACKEND_RETRIEVAL_COLUMNS = (
    "question_id",
    "backend",
    "method",
    "status",
    "expected_method",
    "method_fit",
    "recall_at_5",
    "multi_source_coverage",
    "retrieved_context_count",
    "graph_node_coverage",
    "trace_depth",
    "latency_seconds",
    "artifact_path",
    "error",
)
BACKEND_ANSWER_COLUMNS = (
    "question_id",
    "backend",
    "method",
    "status",
    "claim_support_rate",
    "wikigraph_context_count",
    "wikigraph_source_count",
    "wikigraph_trace_complete",
    "latency_seconds",
    "artifact_path",
    "error",
)


@dataclass(frozen=True)
class BackendEvaluationConfig:
    benchmark_path: Path
    project_root: Path | None
    results_dir: Path
    limit: int
    allow_provider_calls: bool
    include_legacy_ask: bool
    backends: tuple[str, ...]
    graphrag_methods: tuple[str, ...]
    wikigraph_methods: tuple[str, ...]
    retrieval_only: bool
    timeout_seconds: int
    label: str | None = None


def run_backend_evaluation(config: BackendEvaluationConfig) -> EvaluationResult:
    benchmark = load_benchmark(config.benchmark_path)
    project_root = resolve_project_root(benchmark, config.project_root)
    command_cwd = benchmark.root
    retrieval_rows: list[dict[str, object]] = []
    answer_rows: list[dict[str, object]] = []
    for question in benchmark.questions:
        if "graphrag" in config.backends:
            retrieval_rows.append(
                evaluate_auto_route(question, project_root=project_root)
            )
        if "legacy" in config.backends:
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
            if config.include_legacy_ask and not config.retrieval_only:
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
        if "graphrag" in config.backends and not config.retrieval_only:
            for method in config.graphrag_methods:
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
        if "wikigraph" in config.backends:
            for method in config.wikigraph_methods:
                retrieval_rows.append(
                    evaluate_wikigraph_find(
                        question,
                        method=method,
                        project_root=project_root,
                        command_cwd=command_cwd,
                        results_dir=config.results_dir,
                        limit=config.limit,
                        timeout_seconds=config.timeout_seconds,
                    )
                )
                if not config.retrieval_only:
                    answer_rows.append(
                        evaluate_wikigraph_ask(
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
        generated_at=utc_now_iso(),
        retrieval_rows=retrieval_rows,
        answer_rows=answer_rows,
    )
    write_backend_results(result, config)
    return result


def evaluate_wikigraph_find(
    question: BenchmarkQuestion,
    *,
    method: str,
    project_root: Path | None,
    command_cwd: Path,
    results_dir: Path,
    limit: int,
    timeout_seconds: int,
) -> dict[str, object]:
    if project_root is None:
        return _wikigraph_retrieval_row(
            question,
            method=method,
            status="skipped",
            error="project root not configured",
        )
    command = [
        "poetry",
        "run",
        "kb",
        "--project-root",
        str(project_root),
        "find",
        question.question,
        "--engine",
        "wikigraph",
        "--method",
        method,
        "--limit",
        str(limit),
        "--json",
    ]
    run = run_command(command, cwd=command_cwd, timeout_seconds=timeout_seconds)
    artifact_path = _write_artifact(
        results_dir,
        question.id,
        f"wikigraph_{method}_find.json",
        {"question": question.question, "run": run.to_dict()},
    )
    if not run.ok:
        return _wikigraph_retrieval_row(
            question,
            method=method,
            status="failed",
            latency_seconds=run.latency_seconds,
            artifact_path=artifact_path,
            error=run.stderr.strip() or run.stdout.strip(),
        )
    payload = _parse_json_stdout(run.stdout)
    contexts = payload.get("contexts", []) if isinstance(payload, dict) else []
    coverage = score_expected_source_coverage(question.expected_sources, contexts)
    trace_depth = 0
    if contexts:
        trace_depth = max(len(item.get("trace", [])) for item in contexts)
    expected_method = _expected_wikigraph_method(question, method)
    return {
        "question_id": question.id,
        "backend": "wikigraph",
        "method": method,
        "status": "ok",
        "expected_method": expected_method,
        "method_fit": _method_fit(expected_method, method),
        "recall_at_5": coverage.get("recall", ""),
        "multi_source_coverage": coverage.get("multi_source_coverage", ""),
        "retrieved_context_count": len(contexts),
        "graph_node_coverage": len({item.get("node_id") for item in contexts}),
        "trace_depth": trace_depth,
        "latency_seconds": round(run.latency_seconds, 4),
        "artifact_path": artifact_path,
        "error": "",
    }


def evaluate_wikigraph_ask(
    question: BenchmarkQuestion,
    *,
    method: str,
    project_root: Path | None,
    command_cwd: Path,
    results_dir: Path,
    allow_provider_calls: bool,
    timeout_seconds: int,
) -> dict[str, object]:
    if project_root is None:
        return _wikigraph_answer_row(
            question,
            method=method,
            status="skipped",
            error="project root not configured",
        )
    if not allow_provider_calls:
        command = [
            "poetry",
            "run",
            "kb",
            "--project-root",
            str(project_root),
            "ask",
            question.question,
            "--engine",
            "wikigraph",
            "--method",
            method,
            "--json",
        ]
        run = run_command(command, cwd=command_cwd, timeout_seconds=timeout_seconds)
        artifact_path = _write_artifact(
            results_dir,
            question.id,
            f"wikigraph_{method}_ask.json",
            {"question": question.question, "run": run.to_dict()},
        )
        if not run.ok:
            return _wikigraph_answer_row(
                question,
                method=method,
                status="failed",
                latency_seconds=run.latency_seconds,
                artifact_path=artifact_path,
                error=run.stderr.strip() or run.stdout.strip(),
            )
        payload = _parse_json_stdout(run.stdout)
        answer_text = (
            str(payload.get("answer", "")) if isinstance(payload, dict) else run.stdout
        )
        contexts = payload.get("contexts", []) if isinstance(payload, dict) else []
        source_ids = {
            source_id
            for context in contexts
            for source_id in context.get("source_ids", [])
        }
        return {
            "question_id": question.id,
            "backend": "wikigraph",
            "method": method,
            "status": "ok",
            "claim_support_rate": _claim_support_rate(answer_text, contexts),
            "wikigraph_context_count": len(contexts),
            "wikigraph_source_count": len(source_ids),
            "wikigraph_trace_complete": _bool_metric(bool(contexts)),
            "latency_seconds": round(run.latency_seconds, 4),
            "artifact_path": artifact_path,
            "error": "",
        }
    return score_answer_row(
        question,
        retriever="wikigraph",
        method=method,
        status="skipped",
        answer_text="",
        claim_support="provider-required",
        latency_seconds=0,
        artifact_path="",
        error="provider-backed wikigraph ask uses extractive mode unless --allow-provider-calls",
    )


def write_backend_results(
    result: EvaluationResult, config: BackendEvaluationConfig
) -> None:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    prefix = "backend"
    retrieval_path = config.results_dir / f"{prefix}_retrieval_metrics.csv"
    answer_path = config.results_dir / f"{prefix}_answer_metrics.csv"
    summary_path = config.results_dir / f"{prefix}_summary.md"
    _write_csv(retrieval_path, BACKEND_RETRIEVAL_COLUMNS, result.retrieval_rows)
    _write_csv(answer_path, BACKEND_ANSWER_COLUMNS, result.answer_rows)
    summary_path.write_text(
        _render_summary(result, config),
        encoding="utf-8",
    )
    write_results(
        result,
        config.results_dir,
        allow_provider_calls=config.allow_provider_calls,
    )


def _render_summary(result: EvaluationResult, config: BackendEvaluationConfig) -> str:
    label = config.label or result.benchmark.name
    lines = [
        f"# Backend Evaluation — {label}",
        "",
        f"Generated: {result.generated_at}",
        f"Project: {result.project_root or 'n/a'}",
        f"Backends: {', '.join(config.backends)}",
        "",
        f"Retrieval rows: {len(result.retrieval_rows)}",
        f"Answer rows: {len(result.answer_rows)}",
        "",
    ]
    return "\n".join(lines)


def _write_csv(
    path: Path, columns: Sequence[str], rows: Sequence[dict[str, object]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_artifact(
    results_dir: Path, question_id: str, filename: str, payload: dict[str, object]
) -> str:
    artifact_dir = results_dir / ARTIFACTS_DIR / question_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path.relative_to(results_dir).as_posix()


def _parse_json_stdout(stdout: str) -> dict[str, object]:
    text = stdout.strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _expected_wikigraph_method(question: BenchmarkQuestion, actual: str) -> str:
    return question.expected_method or actual


def _method_fit(expected: str, actual: str) -> str:
    if not expected:
        return ""
    normalized_expected = expected.replace("drift", "drift-lite")
    normalized_actual = actual.replace("drift", "drift-lite")
    return "1" if normalized_expected == normalized_actual else "0"


def _claim_support_rate(answer_text: str, contexts: list[object]) -> str:
    if not answer_text.strip():
        return "0"
    if contexts:
        return "1"
    return "0"


def _bool_metric(value: bool) -> str:
    return "1" if value else "0"


def _wikigraph_retrieval_row(
    question: BenchmarkQuestion,
    *,
    method: str,
    status: str,
    latency_seconds: float = 0,
    artifact_path: str = "",
    error: str = "",
) -> dict[str, object]:
    return {
        "question_id": question.id,
        "backend": "wikigraph",
        "method": method,
        "status": status,
        "expected_method": question.expected_method or "",
        "method_fit": "",
        "recall_at_5": "",
        "multi_source_coverage": "",
        "retrieved_context_count": "",
        "graph_node_coverage": "",
        "trace_depth": "",
        "latency_seconds": latency_seconds,
        "artifact_path": artifact_path,
        "error": error,
    }


def _wikigraph_answer_row(
    question: BenchmarkQuestion,
    *,
    method: str,
    status: str,
    latency_seconds: float = 0,
    artifact_path: str = "",
    error: str = "",
) -> dict[str, object]:
    return {
        "question_id": question.id,
        "backend": "wikigraph",
        "method": method,
        "status": status,
        "claim_support_rate": "",
        "wikigraph_context_count": "",
        "wikigraph_source_count": "",
        "wikigraph_trace_complete": "",
        "latency_seconds": latency_seconds,
        "artifact_path": artifact_path,
        "error": error,
    }
