# Evaluation Results

The current `verification_report.md` documents the WikiGraphRAG vs
Microsoft GraphRAG comparison on a 10-paper arXiv corpus after
fixing eight gameability issues in the harness. **Start there.**

## Headline artifacts

| Path | Content |
|---|---|
| `verification_report.md` | Full write-up: what was wrong with the previous evaluator, what we fixed, what we added, the real scores, and the honest conclusion. |
| `backend_summary_real_pdf_answers_v2.md` | Auto-generated headline tables (retrieval + provider-backed answers). |
| `backend_answer_metrics_real_pdf_answers_v2.csv` | Per-question answer metrics (all 14 questions × 2 backends). |
| `backend_retrieval_metrics_real_pdf_answers_v2.csv` | Per-question retrieval metrics. |
| `artifacts/backend_runs_real_pdf_answers_v2.json` | Raw JSON: full answer text, citation refs, retrieved titles / paths / snippets. |
| `per_question_review.md` | Side-by-side WGR vs GraphRAG answers + retrieved contexts per question. |
| `per_pdf_review.md` | Per-paper inspection: wiki page, TextUnit count, entity-mention counts. |

## Phase 4 ablation artifacts

| Path | Content |
|---|---|
| `backend_summary_baseline_v2.md` | WGR with `retrieval_improvements_enabled=false`. |
| `backend_summary_improved_v5.md` | WGR with `retrieval_improvements_enabled=true` (default) — neutral on this corpus. |
| `backend_summary_improved_v6.md` | Same as v5 with the colon-prefix entity alias fix (small recall regression on `realm_vs_rag`, big answer-side win on REALM-specific local questions). |

## Legacy artifacts

| Path | Content |
|---|---|
| `summary.md`, `retrieval_metrics.csv`, `answer_metrics.csv` | Pre-pivot Phase 8 evaluator output, kept for git history; superseded by the `backend_*` files above. |
| `command_surface_e2e.md` | Independent end-to-end CLI smoke test summary. |

## Reproduction

```bash
poetry run python scripts/fetch_eval_corpus.py
poetry run kb --project-root ~/wgr-eval-project init
poetry run kb --project-root ~/wgr-eval-project config provider set openai
poetry run kb --project-root ~/wgr-eval-project add ~/eval-pdfs/*.pdf
poetry run kb --project-root ~/wgr-eval-project update --force --allow-partial
poetry run python scripts/evaluate_backends.py \
  --project-root ~/wgr-eval-project --backends wikigraph graphrag \
  --allow-provider-calls --graphrag-retrieve-mode text_units \
  --label real_pdf_answers_v2
poetry run python scripts/render_eval_reviews.py
```

Provider-backed runs use the centrally configured KB provider
(OpenAI by default — set `OPENAI_API_KEY`). Mistral OCR is required
for PDF normalization; set `MISTRAL_API_KEY`.
