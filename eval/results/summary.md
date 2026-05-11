# GraphRAG Evaluation Summary

- Generated at: 2026-05-11T22:27:19+00:00
- Benchmark: graphrag-pivot-evaluation v2
- Project root: `../kb-cli-realworld-pdf-20260415/project-codex-20260424-userchanges-realrun`
- Provider calls: disabled; provider-backed answer commands were skipped
- Questions: 12
- Retrieval rows: 24 (ok: 24)
- Answer rows: 48 (skipped_provider_call: 48)
- Average Recall@5: 0.530
- Auto-router method fit: 0.667
- Claim support rate: n/a

## Outputs

- `eval/results/retrieval_metrics.csv`
- `eval/results/answer_metrics.csv`
- `eval/results/artifacts/<question-id>/...`

Run with `--allow-provider-calls` only when the configured GraphRAG provider/API key is ready and you explicitly want to spend model and embedding/query budget.
