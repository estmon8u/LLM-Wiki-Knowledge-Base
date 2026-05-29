"""Driver for the research-grounded RAG evaluation harness.

Compares all four answering methods (direct / legacy / graphrag / wikigraph)
with deterministic retrieval metrics, RAGAS answer/context metrics, a
bias-mitigated LLM judge, and bootstrap confidence intervals.

Examples::

    # Offline retrieval-only (no provider/LLM calls)
    python scripts/evaluate_rag.py --retrieval-only \
        --methods legacy graphrag wikigraph

    # Full provider-backed run with RAGAS + judge (costs tokens)
    python scripts/evaluate_rag.py --allow-provider-calls \
        --methods direct legacy graphrag wikigraph --ragas --judge
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.rag_eval.aggregate import (
    summarize,
    write_json,
    write_leaderboard_markdown,
    write_rows_csv,
)
from scripts.rag_eval.backends import build_backends, build_command_context
from scripts.rag_eval.dataset import EvalQuestion, load_benchmark
from scripts.rag_eval.generation_metrics import score_generation
from scripts.rag_eval.judge import LLMJudge
from scripts.rag_eval.ragas_metrics import RagasConfig, RagasItem, RagasScorer
from scripts.rag_eval.retrieval_metrics import score_retrieval
from scripts.rag_eval.types import RagSample

_PROVIDER_BACKED = {"direct", "legacy", "graphrag"}

_RETRIEVAL_METRICS = ["recall_at_k", "precision_at_k", "ndcg_at_k", "mrr", "hit_at_k"]
_RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]
_GENERATION_METRICS = [
    "citation_validity",
    "grounded",
    "refusal_correct",
    "token_f1",
    "rouge_l",
    "entity_coverage",
]
_JUDGE_METRICS = [
    "judge_correctness",
    "judge_groundedness",
    "judge_relevance",
    "judge_overall",
]
_AUX_METRICS = ["latency_seconds", "answer_token_length"]

_CONTAMINATION_NOTE = (
    "Contamination caveat: the corpus papers are public, so the `direct` "
    "(no-retrieval) baseline may benefit from pretraining memorization; weigh "
    "grounded/faithfulness/citation metrics over raw correctness."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Research-grounded RAG eval harness.")
    parser.add_argument(
        "--benchmark", type=Path, default=Path("eval") / "benchmark.yaml"
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("eval") / "rag_eval")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["legacy", "graphrag", "wikigraph-classic", "wikigraph-lightrag"],
        choices=[
            "direct",
            "legacy",
            "graphrag",
            "wikigraph",
            "wikigraph-classic",
            "wikigraph-lightrag",
        ],
    )
    parser.add_argument("--wikigraph-method", default="auto")
    parser.add_argument("--graphrag-method", default="auto")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--allow-provider-calls", action="store_true")
    parser.add_argument("--ragas", action="store_true", help="Run RAGAS metrics.")
    parser.add_argument("--judge", action="store_true", help="Run the LLM judge.")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--ragas-model", default="gpt-5.4-nano")
    parser.add_argument("--label", default=None)
    return parser


def _retrieval_row(question: EvalQuestion, sample: RagSample, k: int) -> dict:
    scores = score_retrieval(
        question.expected_source_ids, sample.retrieved_contexts, k=k
    )
    return {
        "question_id": question.id,
        "category": question.category,
        "backend": sample.backend,
        "method": sample.method,
        "recall_at_k": scores.recall_at_k,
        "precision_at_k": scores.precision_at_k,
        "ndcg_at_k": scores.ndcg_at_k,
        "mrr": scores.mrr,
        "hit_at_k": scores.hit_at_k,
        "retrieved_count": scores.retrieved_count,
        "latency_seconds": sample.latency_seconds,
        "error": sample.error,
    }


def _add_generation(row: dict, question: EvalQuestion, sample: RagSample) -> None:
    g = score_generation(question, sample)
    row.update(
        {
            "citation_validity": g.citation_validity,
            "citation_count": g.citation_count,
            "grounded": g.grounded,
            "refusal_correct": g.refusal_correct,
            "token_f1": g.token_f1,
            "rouge_l": g.rouge_l,
            "entity_coverage": g.entity_coverage,
            "answer_token_length": g.answer_token_length,
            "insufficient_evidence": sample.insufficient_evidence,
            "provider_mode": sample.provider_mode,
        }
    )


def run_eval(args: argparse.Namespace) -> int:
    """Execute the evaluation and write results. Returns an exit code."""
    project_root = (args.project_root or Path.cwd()).resolve()
    context = build_command_context(project_root)
    questions = load_benchmark(args.benchmark)
    if not questions:
        print(f"No benchmark questions in {args.benchmark}.")
        return 1

    methods = list(args.methods)
    if not args.retrieval_only and not args.allow_provider_calls:
        skipped = [m for m in methods if m in _PROVIDER_BACKED]
        if skipped:
            print(
                f"Skipping provider-backed backends {skipped} "
                "(pass --allow-provider-calls to include them)."
            )
        methods = [m for m in methods if m not in _PROVIDER_BACKED]
    if not methods:
        print("No runnable backends after provider gating.")
        return 1

    backends = build_backends(
        context,
        methods,
        wikigraph_method=args.wikigraph_method,
        graphrag_method=args.graphrag_method,
    )

    rows: list[dict] = []
    samples_by_backend: dict[str, list[tuple[EvalQuestion, RagSample]]] = {}
    for question in questions:
        for backend in backends:
            if args.retrieval_only:
                try:
                    contexts = backend.retrieve(question)
                    sample = RagSample(
                        question_id=question.id,
                        question=question.question,
                        backend=backend.name,
                        method=getattr(backend, "method", ""),
                        retrieved_contexts=contexts,
                    )
                except Exception as exc:
                    sample = RagSample(
                        question_id=question.id,
                        question=question.question,
                        backend=backend.name,
                        method=getattr(backend, "method", ""),
                        error=str(exc),
                    )
                rows.append(_retrieval_row(question, sample, args.k))
                continue
            sample = backend.run(question)
            row = _retrieval_row(question, sample, args.k)
            _add_generation(row, question, sample)
            rows.append(row)
            samples_by_backend.setdefault(backend.name, []).append((question, sample))

    if not args.retrieval_only and args.allow_provider_calls and args.ragas:
        _apply_ragas(rows, samples_by_backend, model=args.ragas_model)
    if not args.retrieval_only and args.allow_provider_calls and args.judge:
        _apply_judge(rows, samples_by_backend, context, judge_model=args.judge_model)

    metric_order = (
        _RETRIEVAL_METRICS
        + _RAGAS_METRICS
        + _GENERATION_METRICS
        + _JUDGE_METRICS
        + _AUX_METRICS
    )
    summaries = summarize(rows, metric_order, n_boot=args.bootstrap, seed=args.seed)
    _write_outputs(args, rows, summaries, metric_order)
    return 0


def _apply_ragas(rows, samples_by_backend, *, model: str) -> None:
    scorer = RagasScorer(config=RagasConfig(model=model))
    row_index = {(r["backend"], r["question_id"]): r for r in rows}
    for backend, pairs in samples_by_backend.items():
        items = [
            RagasItem(
                question_id=q.id,
                question=q.question,
                answer=s.answer,
                contexts=s.context_texts,
                reference=q.reference_answer,
            )
            for q, s in pairs
        ]
        scores = scorer.score(items)
        for qid, metric_scores in scores.items():
            row = row_index.get((backend, qid))
            if row is not None:
                row.update(metric_scores)


def _apply_judge(rows, samples_by_backend, context, *, judge_model) -> None:
    from graphwiki_kb.providers import build_provider

    config = dict(context.config)
    if judge_model:
        config = {**config}
        config.setdefault("providers", {})
    provider = build_provider(context.config)
    if provider is None:
        return
    judge = LLMJudge(provider=provider)
    row_index = {(r["backend"], r["question_id"]): r for r in rows}
    for backend, pairs in samples_by_backend.items():
        for question, sample in pairs:
            scores = judge.score_answer(
                question=question.question,
                answer=sample.answer,
                reference=question.reference_answer,
                contexts=sample.context_texts,
            )
            row = row_index.get((backend, question.id))
            if row is not None:
                row.update(
                    {
                        "judge_correctness": scores.correctness,
                        "judge_groundedness": scores.groundedness,
                        "judge_relevance": scores.relevance,
                        "judge_overall": scores.overall,
                    }
                )


def _write_outputs(args, rows, summaries, metric_order) -> None:
    suffix = f"_{args.label}" if args.label else ""
    results_dir: Path = args.results_dir
    write_rows_csv(results_dir / f"rag_eval_rows{suffix}.csv", rows)
    write_json(
        results_dir / f"rag_eval_summary{suffix}.json",
        {
            "benchmark": str(args.benchmark),
            "methods": list(args.methods),
            "retrieval_only": bool(args.retrieval_only),
            "k": args.k,
            "summaries": [s.__dict__ for s in summaries],
        },
    )
    write_leaderboard_markdown(
        results_dir / f"rag_eval_leaderboard{suffix}.md",
        summaries,
        metric_order=metric_order,
        notes=[_CONTAMINATION_NOTE],
    )
    print(f"Wrote leaderboard to {results_dir / f'rag_eval_leaderboard{suffix}.md'}")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = build_parser().parse_args(argv)
    return run_eval(args)
