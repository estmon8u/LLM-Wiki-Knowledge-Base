# WikiGraphRAG vs Microsoft GraphRAG verification report

Generated on 2026-05-21 from a disposable KB project built from repository documentation (`README.md`, `docs/graphrag-pivot.md`, `docs/start-guide.md`, and `docs/architecture/high-level.md`). The project was initialized with `kb init`, configured for OpenAI (`gpt-4.1-mini`, `text-embedding-3-small`), ingested with `kb add`, and built with `kb update --force --allow-partial`. GraphRAG indexing completed successfully and exported 993 graph wiki pages.

## Harness fixes before scoring

- Fixed the Phase 8 evaluator so legacy retrieval calls the current `kb find --engine legacy` surface instead of the removed `kb legacy find` command group.
- Fixed GraphRAG answer evaluation to pass `--engine graphrag`, preventing accidental use of the default WikiGraphRAG engine.
- Added stricter expected entities / expected answer terms / insufficient-evidence expectations to the benchmark schema used by the backend evaluator.
- Added provider-mode and citation-reference validity metrics so provider-free rows cannot be presented as like-for-like wins over skipped GraphRAG rows.

## Real retrieval scores

| Backend | Effective Recall@5 | Ground-truth questions | Avg latency | Errors |
|---|---:|---:|---:|---:|
| WikiGraphRAG auto | **0.767** | 5/6 | **0.029s** | 0 |
| Microsoft GraphRAG auto artifact search | 0.350 | 5/6 | 0.538s | 0 |
| Legacy FTS comparator | 0.467 | 5/6 | 0.057s | 0 |

## Real provider-backed answer scores

| Backend | Provider mode | Quality score | Grounded entity rate | Grounded term rate | Avg citations | Insufficient-evidence match | Avg latency |
|---|---|---:|---:|---:|---:|---:|---:|
| WikiGraphRAG auto | provider-backed via KB provider | **0.907** | 0.833 | 0.833 | 4.00 | **1.00** | **16.69s** |
| Microsoft GraphRAG auto | provider-backed GraphRAG | 0.830 | **1.000** | 0.833 | 2.67 | 0.83 | 41.50s |

## Interpretation

On this small documentation corpus, WikiGraphRAG was better on retrieval recall, answer quality composite, insufficient-evidence behavior, citation count, and latency. Microsoft GraphRAG had a higher grounded-entity rate on the answer benchmark and produced longer answers. The unsupported-question row is the clearest quality gap: WikiGraphRAG correctly marked insufficient evidence, while Microsoft GraphRAG produced a grounded-looking answer and failed the insufficient-evidence expectation.

These scores should not be generalized as a universal claim that WikiGraphRAG is always better than GraphRAG. They are real scores from a successful end-to-end run on a small docs corpus after fixing harness issues that could otherwise game the comparison.

## Artifacts

- `eval/results/backend_summary_docs_retrieval.md`
- `eval/results/backend_retrieval_metrics_docs_retrieval.csv`
- `eval/results/backend_summary_docs_answers.md`
- `eval/results/backend_answer_metrics_docs_answers.csv`
