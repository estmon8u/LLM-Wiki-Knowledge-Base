# RAG evaluation harness (`scripts/rag_eval`)

A from-scratch, research-grounded evaluation harness that compares **all four
answering methods** fairly and rigorously, and is hard to game.

```bash
# Offline, free: rank-aware retrieval metrics across backends
python scripts/evaluate_rag.py --retrieval-only \
    --methods legacy graphrag wikigraph

# Full provider-backed run with RAGAS + bias-mitigated judge (costs tokens)
python scripts/evaluate_rag.py --allow-provider-calls --ragas --judge \
    --methods direct legacy graphrag wikigraph
```

## The four methods

| Method | What it is |
|---|---|
| `direct` | LLM-only baseline, **no retrieval** (parametric knowledge) |
| `legacy` | deprecated SQLite FTS retrieval + grounded answer |
| `graphrag` | Microsoft GraphRAG retrieval + ask controller |
| `wikigraph` | custom WikiGraphRAG (classic/lightrag, method-aware) |

Every backend emits the same `RagSample` (question, retrieved contexts with
source ids + refs, answer, citations, latency), so metrics are computed
identically and fairly.

## Metrics

**Retrieval (deterministic, no LLM)** — scored against ground-truth
`expected_source_ids` with binary relevance and rank awareness:
`recall@k`, `precision@k`, `hit@k`, `MRR`, `nDCG@k`. Questions without ground
truth are excluded from these averages (not scored 0).

**RAGAS (real `ragas` library, provider-backed)** — `faithfulness`,
`answer_relevancy`, `context_precision`, `context_recall`. Metric selection is
automatic and fair: context-dependent metrics are skipped for the no-retrieval
`direct` backend, and reference-dependent metrics are skipped when a question
has no reference answer.

**Generation (deterministic, anti-gaming)** — `citation_validity` (against the
*returned* contexts), `grounded`, `refusal_correct` (rewards correct refusal on
insufficient-evidence questions), `token_f1`/`rouge_l` vs the reference, and
`answer_token_length` (a verbosity signal that is **never** rewarded).

**LLM judge (bias-mitigated)** — an anchored rubric scoring `correctness`,
`groundedness`, `relevance` (1-5), plus an order-swapped pairwise comparator.

## Fairness, rigor, and anti-gaming (research-grounded)

- **Retrieval and generation are evaluated separately** (the RAG-triad split).
- **Rank-aware IR metrics** (nDCG/MRR/recall@k) rather than presence-only counts.
- **Bias-mitigated judging** (from the LLM-as-judge literature): temperature 0,
  strict JSON schema, an anchored rubric, **blinded** system identity, and
  **order-swap** for pairwise comparisons (a position-biased judge that always
  prefers the first answer yields a *tie*, not a spurious win). Use a
  **cross-family** judge (different model family than the generator) when
  possible.
- **No single gameable composite** as the headline — every metric is reported
  on its own, with **bootstrap 95% confidence intervals** so within-noise
  differences are not reported as wins. Verbosity/citation count are never
  rewarded; correct refusal is.
- **Contamination caveat**: the corpus papers are public, so the `direct`
  baseline may benefit from pretraining memorization — weigh grounded/
  faithfulness/citation metrics over raw correctness.

## RAGAS dependency note

`ragas` is a normal project dependency (`^0.2.0`). It pulls the LangChain stack
and required relaxing two pins (`rich` 15→14, `tenacity` 9→8; `openai` 2 kept).
A tiny import-time shim (`scripts/rag_eval/_compat.py`) stubs the removed
`langchain_community.chat_models.vertexai.ChatVertexAI` (an unused VertexAI
wrapper) so `import ragas` succeeds in this environment.

## Outputs

Written under `--results-dir` (default `eval/rag_eval/`):
`rag_eval_rows.csv` (raw per-question rows), `rag_eval_summary.json`
(per-backend metric summaries with CIs), and `rag_eval_leaderboard.md`.

## Reproducibility / gating

Provider/LLM-judge/RAGAS calls are gated behind `--allow-provider-calls` so the
deterministic retrieval layer stays offline and reproducible. `--seed` and a
fixed bootstrap make aggregation deterministic. The benchmark
(`eval/benchmark.yaml`, schema v4) carries `reference_answer` and
`expected_source_ids`; `eval/benchmark_synthesis.yaml` adds cross-corpus
(3+ source) synthesis questions.
```
