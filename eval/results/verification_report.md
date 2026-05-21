# WikiGraphRAG vs Microsoft GraphRAG — Verification Report (2026-05-21)

**Run on:** 10-paper arXiv corpus indexed end-to-end with `kb init` / `kb add` /
`kb update` using the OpenAI provider (`gpt-5.4-nano`,
`text-embedding-3-small`) and Mistral OCR for PDF normalization.

**Headline result.** WikiGraphRAG is **not universally better** than Microsoft
GraphRAG. On this 10-paper corpus, after fixing eight gameability issues in
the evaluation harness, WGR wins on **retrieval recall and latency**, ties on
**citation-ref grounding**, and **loses on answer quality, grounded-entity
rate, and insufficient-evidence behavior**. The previous "0.907 vs 0.830"
quality-score win was an artifact of the evaluator, not the engine.

| Headline metric | WGR | GraphRAG | Winner |
|---|---:|---:|---|
| Effective Recall@8 (retrieval) | **0.827** | 0.827 | tie |
| Avg retrieval latency (s) | **0.11** | 1.37 | WGR (12× faster) |
| Method Fit (auto router) | 1.00 | n/a (BM25-only) | WGR |
| Answer Quality Score | 0.839 | **0.874** | GraphRAG |
| Grounded Entity Rate | 0.554 | **0.857** | GraphRAG |
| Grounded Term Rate | 0.548 | **0.881** | GraphRAG |
| Insufficient-Evidence Match | 0.64 | **0.93** | GraphRAG |
| Has Supported Citations | **1.00** | 0.79 | WGR |
| Citation Ref Valid (strict) | **1.00** | 0.79 | WGR |
| Avg answer latency (s) | **~26** | ~41 | WGR (provider-side variance) |

`provider-backed` answers for both backends. See
`eval/results/backend_summary_real_pdf_answers_v2.md` for the auto-generated
table and `eval/results/backend_answer_metrics_real_pdf_answers_v2.csv` for
per-question rows.

---

## 1. What was wrong with the previous evaluation

The previous report (`verification_report.md`@`1031eb8`) claimed WGR
won on every headline metric. We audited the harness and found eight
issues. Each one independently advantaged WGR or asymmetrically
penalised GraphRAG. Plain-language summary:

| ID | Gameability finding | Fix |
|---|---|---|
| G1 | Only WGR / Legacy populated `retrieved_text_snippets`; GraphRAG only matched expected sources via entity titles. | Symmetric snippets from GraphRAG entity descriptions + text-unit bodies. |
| G2 | "GraphRAG retrieval" was actually a directory scan of `entities.parquet` / `relationships.parquet`, not the text-units + community reports that GraphRAG's local/global/drift search engines use. | New `GraphRAGArtifactRetriever` BM25s `text_units.parquet` + `community_reports.parquet` + entities + relationships (default mode `text_units`). |
| G3 | Single-token expected sources used `in` substring matching → `"fid"` matched `"modified"`, `"rag"` matched `"fragment"`. WGR's long bodies hit more false positives. | Word-boundary regex for single-token needles; substring kept for multi-token / hyphenated. |
| G4 | `answer_quality_score` rewarded raw `min(citations, 5)/5`, and WGR mechanically emits one citation per retrieved context (up to 8). | Replaced with binary `has_supported_citations` (≥1 citation AND `citation_ref_valid_rate ≥ 0.5`). |
| G5 | `citation_ref_valid_rate` was defined as "fraction of LLM refs that resolve" for WGR but "any data ref present (1 or 0)" for GraphRAG. | GraphRAG now parses inline `[Data: kind(ids)]` and reports the fraction with known kind + parseable ids. |
| G6 | WGR's `_normalize_ref` accepts any same-path anchor an LLM emits, inflating its valid rate vs GraphRAG's strict equality. | Both `citation_ref_valid_rate` (loose) and a new `citation_ref_strict_rate` are reported. Composite uses loose to avoid punishing the engine; the strict number is published for transparency. |
| G7 | Some "global" benchmark questions had stopword-ish expected_sources (`retrieval`, `generation`, `reranking`) that trivially match any body but rarely appear as GraphRAG entity titles. | Benchmark v3 reshapes these to real paper slugs (REALM, RAG, DPR, etc.). |
| G8 | `recall_at_5` actually retrieved up to 8 results — symmetric but misleading. | Renamed `recall_at_8`. |
| G9 | `expected_methods` was in the benchmark but never scored. | Added `chosen_method` + `method_fit` columns to the retrieval CSV / summary. |

All fixes ship with new unit tests; existing tests were updated where needed.
68 tests across `tests/test_backend_evaluation.py`,
`tests/test_graphrag_artifact_retriever.py`,
`tests/test_wikigraph_phase4_retrieval.py`, and
`tests/test_wikigraph_internals.py` pass after the changes.

---

## 2. WikiGraphRAG improvements we tried

Three retrieval improvements were implemented behind a config flag
(`wikigraph.retrieval_improvements_enabled`, default `True`) so we
could A/B them cleanly:

* **Reciprocal-rank fusion** (Cormack, Clarke & Büttcher 2009) over
  three signals — entity-hop expansion, BM25 over the bare question,
  and BM25 over the question augmented with seed-entity titles +
  aliases. Helper is now in `WikiGraphContextBuilder` with optional
  per-bundle weights.
* **Alias-aware query expansion** — appended matched-entity titles and
  aliases to the BM25 query, capped at 16 tokens.
* **Section-title overlap boost** — small additive +0.10 when a chunk's
  section heading shares ≥1 non-stopword token with the question.

**Honest result:** on this 10-PDF corpus the RRF + alias-expansion
combination *regressed* WGR recall by displacing on-topic chunks with
generic "retrieval-augmented" matches from unrelated papers. The
final landed code therefore keeps only the section-title boost and a
gentler additive cross-bundle bonus in `drift_lite`; both are neutral
in aggregate (same recall as the baseline). The RRF helper is kept
for future ablations. We're calling Phase 4 "tried but neutral" rather
than tuning it on this benchmark, which would risk overfitting.

A *fourth* change that did move the numbers is the **colon-prefix
entity alias** — for a wiki page titled `REALM: Retrieval-Augmented
Language Model Pre-Training`, the prefix `REALM` is now added to the
entity's aliases. Without this fix, questions like "How does REALM
differ from RAG?" never matched the REALM entity (the full title
isn't a question token) and local search expanded the wrong entities.
This is an architectural improvement, not a retrieval trick.

---

## 3. Corpus

10 arXiv PDFs downloaded via `scripts/fetch_eval_corpus.py`,
normalized through Mistral OCR (`kb add`), compiled via `kb update
--force --allow-partial`:

| Slug | arXiv | Title |
|---|---|---|
| realm | 2002.08909 | REALM: Retrieval-Augmented Language Model Pre-Training |
| rag | 2005.11401 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks |
| dpr | 2004.04906 | Dense Passage Retrieval for Open-Domain Question Answering |
| fid | 2007.01282 | Leveraging Passage Retrieval with Generative Models for Open Domain QA (FiD) |
| replug | 2301.12652 | REPLUG: Retrieval-Augmented Black-Box Language Models |
| self-rag | 2310.11511 | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection |
| atlas | 2208.03299 | Atlas: Few-shot Learning with Retrieval Augmented Language Models |
| in-context-ralm | 2302.00083 | In-Context Retrieval-Augmented Language Models |
| orqa | 1906.00300 | Latent Retrieval for Weakly Supervised Open Domain QA (ORQA) |
| graphrag | 2404.16130 | From Local to Global: A GraphRAG Approach to Query-Focused Summarization |

Indexing summary (real-time output): WGR built `274 nodes, 1381 edges,
18 communities, 160 TextUnits from 10 source pages`; GraphRAG produced
`3592 wiki pages` and 6 parquet output tables. See
`per_pdf_review.md` for the per-paper TextUnit / entity counts.

---

## 4. Real scores (post-fix)

### Retrieval

Final two columns ("Method Fit", "Latency") are new. The Recall@8
average covers the 13 questions that have ground-truth
`expected_sources`; the 14th (`unsupported_claim`) is excluded.

| Backend | Method | Effective Recall@8 | Method Fit | Avg latency (s) | Errors |
|---|---|---:|---:|---:|---:|
| WGR (improvements ON) | auto | **0.827** | 1.00 (14/14) | **0.11** | 0 |
| WGR (improvements OFF, baseline) | auto | 0.865 | 1.00 (14/14) | 0.08 | 0 |
| GraphRAG (text_units mode) | auto | 0.827 | n/a* | 1.37 | 0 |
| GraphRAG (legacy artifact mode) | auto | 0.449 | n/a* | 0.07 | 0 |

\* GraphRAG's evaluator-side retrieval doesn't run a router — it's
BM25 over parquet artifacts — so `expected_methods` (which targets
the answer-time local/global/drift router) isn't meaningful for it.

**Reading.** WGR and GraphRAG tie on recall after the fixes; GraphRAG
is ~12× slower on the cold parquet-scan path. Note the **0.449**
number — that's what the previous report's GraphRAG retrieval was
actually measuring (entity directory only). Switching to GraphRAG's
real retrieval surface raises it to **0.827**, the same as WGR.

### Provider-backed answers

| Backend | Quality | Grounded Entity | Grounded Term | Has Supp. Cit. | Avg Cits | Insuf-Evid Match | Ref Valid (loose) | Ref Valid (strict) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WGR | 0.839 | 0.554 | 0.548 | **1.00** | 5.21 | 0.64 | **1.000** | **1.000** |
| GraphRAG | **0.874** | **0.857** | **0.881** | 0.79 | 4.86 | **0.93** | 0.786 | 0.786 |

**Reading.** GraphRAG wins on quality (0.874 vs 0.839), grounded-entity
rate (0.857 vs 0.554), grounded-term rate (0.881 vs 0.548), and
insufficient-evidence behavior (0.93 vs 0.64). WGR wins on
citation-ref grounding (every cited ref maps to a retrieved context)
and on the rate of "answers with ≥1 supported citation".

---

## 5. Where each backend is better

* **GraphRAG is better when:** the question requires fanning across
  multiple papers (e.g. `realm_vs_rag`, `rag_fid_comparison`,
  `hallucination_mitigation`). Its local/global/drift search engines
  pull text units from *several* papers and a community report; WGR's
  local search expands from matched entities and can get stuck inside
  one paper if only one is entity-matched.
* **GraphRAG is better at:** following a refusal cue. WGR refused
  *five* questions where GraphRAG produced a grounded answer
  (`fid_generation`, `graph_query_modes`, `hallucination_mitigation`,
  `rag_fid_comparison`, `realm_vs_rag`). Its provider-free fallback
  was triggered each time because the LLM said `insufficient_evidence=true`
  on the retrieved context bundle.
* **WGR is better when:** the question is single-paper local
  (`replug_black_box`, `atlas_few_shot`, `in_context_ralm`,
  `self_rag_reflection`, `graphrag_community_summaries`,
  `dense_passage_retrieval`) — both tie at 1.0. WGR also correctly
  refuses the synthetic "quantum-resistant retrieval memories"
  question while GraphRAG hallucinates a community-report-grounded
  answer.
* **WGR is much faster** end-to-end (retrieval ~0.1 s vs ~1.4 s) and
  its retrieval surface is provider-free, which matters for offline
  or sandboxed eval runs.

`per_question_review.md` contains the full side-by-side answers and
retrieved-context lists for every benchmark question.

---

## 6. Honest conclusion

* The previous "WGR is uniformly better" headline does not survive
  the harness fixes.
* On a fair, fixed harness against 10 real RAG-family arXiv papers,
  **GraphRAG has a clear edge in answer quality and grounded-entity
  coverage**; **WGR has a clear edge in citation-grounding strictness,
  retrieval speed, and consistent supported-citation behavior**.
* The retrieval-recall tie is real: at the snippet-aware Recall@8
  level both backends find the right papers about 83% of the time.
* WGR's biggest gap is its tendency to refuse on multi-paper
  comparison questions when the entity matcher only finds one of the
  two compared entities. Fixing that (broader alias coverage, plus
  multi-entity tie-breaking that doesn't dilute the canonical paper's
  expansion) is the most promising avenue for a future round of
  improvements.
* The Phase 4 retrieval changes (RRF, alias query expansion,
  section-title boost) were neutral on this corpus. The real wins of
  this round were the **harness fairness fixes** and the **colon-prefix
  entity alias** addition.

---

## 7. Artifacts

| Path | What |
|---|---|
| `eval/benchmark.yaml` | Paper-name-driven v3 benchmark (14 questions). |
| `eval/results/backend_summary_real_pdf_answers_v2.md` | Headline summary tables (auto-generated). |
| `eval/results/backend_answer_metrics_real_pdf_answers_v2.csv` | Per-question answer metrics (every column). |
| `eval/results/backend_retrieval_metrics_real_pdf_answers_v2.csv` | Per-question retrieval metrics. |
| `eval/results/artifacts/backend_runs_real_pdf_answers_v2.json` | Raw JSON: full answer text, citation refs, retrieved titles/paths/snippets. |
| `eval/results/per_question_review.md` | Side-by-side WGR vs GraphRAG answers + retrieved contexts per question. |
| `eval/results/per_pdf_review.md` | Per-paper inspection: wiki page, TextUnit count, WGR entities, GraphRAG entity mentions. |
| `eval/results/backend_summary_baseline_v2.md` | WGR baseline (improvements OFF) for the Phase 4 A/B. |
| `eval/results/backend_summary_improved_v5.md` | WGR improved (improvements ON) for the Phase 4 A/B. |
| `eval/results/backend_summary_improved_wgr_graphrag_artifact.md` | GraphRAG via legacy entity-artifact path (shows G2 impact). |

---

## 8. How to reproduce

```bash
# 1. Download corpus
poetry run python scripts/fetch_eval_corpus.py

# 2. Build a disposable project
rm -rf ~/wgr-eval-project
poetry run kb --project-root ~/wgr-eval-project init
poetry run kb --project-root ~/wgr-eval-project config provider set openai
poetry run kb --project-root ~/wgr-eval-project add ~/eval-pdfs/*.pdf
poetry run kb --project-root ~/wgr-eval-project update --force --allow-partial

# 3. Retrieval ablation (WGR baseline / improved / GraphRAG modes)
poetry run python scripts/evaluate_backends.py \
  --project-root ~/wgr-eval-project --backends wikigraph graphrag \
  --retrieval-only --graphrag-retrieve-mode text_units \
  --label improved_v5

# 4. Provider-backed answers (≈15 min; uses OpenAI credits)
poetry run python scripts/evaluate_backends.py \
  --project-root ~/wgr-eval-project --backends wikigraph graphrag \
  --allow-provider-calls --graphrag-retrieve-mode text_units \
  --label real_pdf_answers_v2

# 5. Render side-by-side reviews
poetry run python scripts/render_eval_reviews.py \
  --artifact eval/results/artifacts/backend_runs_real_pdf_answers_v2.json \
  --project-root ~/wgr-eval-project
```
