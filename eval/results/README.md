# Evaluation Results

Phase 8 evaluation scripts write their report outputs here:

- `summary.md` — human-readable run summary.
- `retrieval_metrics.csv` — legacy FTS retrieval and GraphRAG auto-router metrics.
- `answer_metrics.csv` — GraphRAG/legacy answer-mode metrics.
- `artifacts/<question-id>/...` — per-command JSON captures, ignored by Git because they can include source snippets or provider output.

Default runs are local-safe and skip provider-backed answer calls:

```bash
poetry run python scripts/evaluate_graph_modes.py
```

Use `--allow-provider-calls` only when the configured GraphRAG provider/API key is ready and you explicitly want to run cost-bearing answer comparisons.
