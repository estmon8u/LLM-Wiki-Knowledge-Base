# Evaluation Results

Phase 8 evaluation scripts write their report outputs here:

- `summary.md` — human-readable run summary.
- `retrieval_metrics.csv` — legacy FTS retrieval and GraphRAG auto-router metrics.
- `answer_metrics.csv` — GraphRAG/legacy answer-mode metrics.
- `backend_summary*.md`, `backend_retrieval_metrics*.csv`, and
  `backend_answer_metrics*.csv` — the newer cross-backend comparison for
  WikiGraphRAG, Microsoft GraphRAG, and legacy FTS.
- `artifacts/<question-id>/...` — per-command JSON captures, ignored by Git because they can include source snippets or provider output.

Local-safe runs skip provider-backed answer calls or mark WikiGraphRAG
provider-free rows separately:

```bash
poetry run python scripts/evaluate_graph_modes.py
poetry run python scripts/evaluate_backends.py --retrieval-only \
  --backends wikigraph graphrag legacy
```

Use `--allow-provider-calls` only when the configured GraphRAG provider/API key
is ready and you explicitly want to run cost-bearing answer comparisons. Do not
treat provider-free WikiGraphRAG rows as like-for-like wins over skipped
GraphRAG rows; headline answer comparisons require successful rows for both
engines.
