# Evaluation Results

The current `verification_report.md` documents the WikiGraphRAG vs
Microsoft GraphRAG comparison on a 10-paper arXiv corpus after
fixing nine gameability issues in the harness. **Start there.**
Earlier verification artifacts used an optimistic docs-corpus headline. The
current reports supersede that with the 10-PDF evaluation after metric and
harness hardening.

Current metric semantics are intentionally stricter. Effective recall averages
only include questions with ground-truth sources, body-snippet source matching
handles TextUnit-backed retrieval, and raw entity mentions are separated from
`grounded_entity_rate` plus the composite `answer_quality_score`. Answer-side
scoring accepts well-known acronym expansions such as `FiD` /
`Fusion-in-Decoder` and handles WikiGraphRAG citations that resolve to a
neighboring TextUnit on the same retrieved source path.

Evaluator subprocesses use explicit `--engine graphrag` and `--engine legacy`
selectors instead of relying on the default `kb ask` engine. Benchmark metadata
names the unified `kb ask/find --engine ...` commands, records
backend-specific `expected_methods`, tracks `expected_answer_terms` and
`forbidden_answer_terms`, and reports grounded answer-term rate alongside
grounded entity rate so generic name-dropping answers do not score well.

The de-game pass made GraphRAG retrieval contribute snippets symmetrically,
uses word-boundary source matching for single-token slugs, scores answer quality
with binary supported citations instead of raw citation volume, publishes loose
and strict WikiGraphRAG citation validity, and reports `Effective Recall@8`
because both backends retrieve up to eight contexts. It also replaced the old
GraphRAG entity/relationship scan with `GraphRAGArtifactRetriever`, a
parquet-backed retriever over GraphRAG `text_units`, `community_reports`,
entities, and relationships. The evaluator default is
`--graphrag-retrieve-mode text_units`; `artifact` mode is retained only to show
the old baseline.

The Phase 4 WikiGraphRAG retrieval-improvement switches remain behind
`wikigraph.retrieval_improvements_enabled`. Aggressive unweighted RRF plus
alias-expanded BM25 regressed entity-centric questions, so public retrieval kept
only the section-title overlap boost and a gentler drift-lite cross-bundle
bonus. The symmetric source-matching snippet cap is 4000 characters, and
`scripts/fetch_eval_corpus.py` downloads the fixed 10-paper arXiv corpus.
Colon-prefix title aliases are also part of the entity catalog so canonical
short names such as `REALM`, `FiD`, and `Self-RAG` can resolve to the actual
page entities before answer synthesis.

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
| `backend_summary_improved_v5.md` | WGR with `retrieval_improvements_enabled=true` (default) after keeping only the conservative Phase 4 subset; neutral on this corpus. |
| `backend_summary_improved_v6.md` | Same as v5 with the colon-prefix entity alias fix (small recall regression on `realm_vs_rag`, big answer-side win on REALM-specific local questions). |

## Legacy artifacts

| Path | Content |
|---|---|
| `summary.md`, `retrieval_metrics.csv`, `answer_metrics.csv` | Pre-pivot Phase 8 evaluator output, kept for git history; superseded by the `backend_*` files above. |
| `command_surface_e2e.md` | Independent end-to-end CLI smoke test summary for the unified command surface. |

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
