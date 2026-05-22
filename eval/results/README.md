# Evaluation Results

The current `verification_report.md` documents the WikiGraphRAG vs
Microsoft GraphRAG comparison on a 10-paper arXiv corpus after
fixing nine gameability issues in the harness. **Start there.**
Commit `1031eb8` was the first branch commit to save real verification
artifacts: a docs-corpus backend comparison, command-surface smoke output, and
an initial `verification_report.md`. Later evaluation-fairness commits replaced
that optimistic docs-corpus headline with the 10-PDF report now tracked here.

Metric semantics changed during the branch. Commit `c28a62e` introduced
`effective_recall_at_5` so retrieval averages only include questions with
ground-truth sources, added body-snippet source matching for TextUnit-backed
retrieval, and split raw entity mentions from `grounded_entity_rate` plus the
composite `answer_quality_score`.
Commit `e70d9cf` then made answer-side scoring less brittle by accepting
well-known acronym expansions such as `FiD` / `Fusion-in-Decoder` and fixed
WikiGraphRAG citation validation when a provider cited a neighboring TextUnit on
the same retrieved source path.
Commit `b77b1cc` fixed evaluator engine selection after CLI unification:
subprocess answer runs now pass explicit `--engine graphrag` or
`--engine legacy` instead of relying on whichever default `kb ask` happens to
use. The benchmark model also gained `expected_answer_terms` and
`forbidden_answer_terms`, and answer rows now carry provider mode and richer
claim/source-trace scoring inputs.
Commit `a4ab162` hardened the benchmark itself: question metadata now names the
unified `kb ask/find --engine ...` commands, records backend-specific
`expected_methods`, and reports grounded answer-term rate alongside grounded
entity rate so a generic name-dropping answer cannot score well.
Commit `d6812ed` was a quality-gate formatting pass over the evaluator scripts
and WikiGraph context builder. It did not change the documented metric
definitions or command surface.
Commit `564b20d` started the Phase 2 de-game pass: GraphRAG retrieval now
contributes snippets symmetrically, source matching uses word boundaries for
single-token slugs, answer quality uses binary supported citations instead of
raw citation volume, WikiGraphRAG reports loose and strict citation validity,
and retrieval summaries now say `Effective Recall@8` because both backends
retrieve up to eight contexts.
Commit `cc67fd5` fixed G2 by replacing the old GraphRAG evaluator scan over
entity/relationship artifacts with `GraphRAGArtifactRetriever`, a parquet-backed
retriever over GraphRAG `text_units`, `community_reports`, entities, and
relationships. The evaluator default is now `--graphrag-retrieve-mode
text_units`; `artifact` mode is retained only to show the old baseline.
Commit `f57e4b0` introduced the Phase 4 WikiGraphRAG retrieval-improvement
switches behind `wikigraph.retrieval_improvements_enabled`: reciprocal-rank
fusion, alias-expanded BM25, section-title overlap boosts, and RRF across
drift-lite sub-question bundles. The later ablation summaries record which of
those changes survived the real-corpus check.
Commit `1db294e` is the honest Phase 4 correction: unweighted RRF plus
alias-expanded BM25 regressed entity-centric questions, so public retrieval kept
only the section-title overlap boost and a gentler drift-lite cross-bundle
bonus. The same commit raised the symmetric source-matching snippet cap from
600 to 4000 characters and added `scripts/fetch_eval_corpus.py` for the
10-paper arXiv corpus.
Commit `c1e5f46` added the colon-prefix alias fix. It can lower a retrieval
recall row when entity-hop expansion narrows around one paper, but it is the
right entity-catalog behavior because provider-backed answers need canonical
short names such as `REALM`, `FiD`, and `Self-RAG` to resolve to the actual page
entities.

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
| `backend_summary_improved_v5.md` | WGR with `retrieval_improvements_enabled=true` (default) after commit `1db294e` kept only the conservative Phase 4 subset; neutral on this corpus. |
| `backend_summary_improved_v6.md` | Same as v5 with the commit `c1e5f46` colon-prefix entity alias fix (small recall regression on `realm_vs_rag`, big answer-side win on REALM-specific local questions). |

## Legacy artifacts

| Path | Content |
|---|---|
| `summary.md`, `retrieval_metrics.csv`, `answer_metrics.csv` | Pre-pivot Phase 8 evaluator output, kept for git history; superseded by the `backend_*` files above. |
| `command_surface_e2e.md` | Independent end-to-end CLI smoke test summary first added with commit `1031eb8`. |

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
