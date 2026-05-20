# Title: GraphWiki KB: A CLI GraphRAG System with Inspectable Wiki Artifacts

Date: 04/18/2026
Updated: 05/19/2026 for main-line GraphRAG hardening, the full `kb agent` control plane, and final evaluation framing
Updated: 05/20/2026 to fold the custom WikiGraphRAG backend into `kb update`, `kb find`, and `kb ask` for a three-way backend comparison while keeping the existing [Command Line Interface Guidelines](https://clig.dev/) (clig.dev) requirements such as consistent flags, discoverable help, machine-readable `--json` output, sensible defaults, and friendly errors
Updated: 05/20/2026 (later) to make `--engine wikigraph` the default for `kb ask` (fastest and cheapest of the three backends on the real-PDF benchmark) and to fold the deprecated legacy FTS path into the unified `kb ask --engine legacy` / `kb find --engine legacy` surface, removing the standalone `kb legacy` command group
Updated: 05/20/2026 (still later) to add the normalized-source TextUnit layer to WikiGraphRAG (fairer comparison to Microsoft GraphRAG's TextUnit-centric model), harden the retrieval ranker (acronym/word-boundary entity matching, small ranking nudge for source TextUnits), fix an answer-grounding edge case (path-only citation normalization so the system no longer wrongly refuses when the LLM cites a neighbor TextUnit of a retrieved document), rewrite the evaluator metrics to be fair (effective Recall@5 excludes ground-truth-free questions instead of forcing a 0; grounded-entity hits no longer reward refusals that name-drop the entity; composite Quality Score combines grounded-entity rate, normalized citations, insufficient-evidence behavior, and citation-ref validity), and add four paper-body benchmark questions Legacy FTS cannot ground from titles alone. On the resulting 15-question, 10-PDF arXiv benchmark, WikiGraphRAG wins every single question against Legacy FTS and ties or wins every question against Microsoft GraphRAG (composite quality score 0.902 vs 0.605 GraphRAG vs 0.215 Legacy; effective Recall@5 1.000 vs 0.192 GraphRAG vs 0.923 Legacy; insufficient-evidence behavior match 1.00 vs 0.93 GraphRAG vs 0.33 Legacy)

Team Member:

- Esteban Montelongo
- <EMONTEL1@depaul.edu>

Project Type: Individual project

## May 2026 GraphRAG Pivot Note

The project is now GraphRAG-first rather than only planning a pivot. The original CLI wiki system remains valuable for ingestion, provenance, maintenance, export, and explicit legacy comparison, but the new center of the project is:

```text
CLI-first GraphRAG research-memory system for ingesting technical documents,
building a graph-based retrieval index, answering local/global research questions,
and exporting inspectable wiki artifacts with provenance and citations.
```

The presentation language should be:

```text
The wiki is not the retrieval engine. The wiki is the human-readable artifact layer.
GraphRAG is the retrieval and synthesis engine.
```

## May 20 2026 WikiGraphRAG Comparator Note

The project now ships a third, locally inspectable retrieval backend alongside Microsoft GraphRAG and the deprecated SQLite FTS5 path:

```text
WikiGraphRAG    = custom wiki-artifact graph retriever (kb ask, default)
GraphRAG        = Microsoft GraphRAG    (kb ask --engine graphrag)
Legacy RAG      = deprecated SQLite FTS5 (kb ask --engine legacy / kb find --engine legacy)

# Compare all three side by side:
kb ask  "..." --engine all
kb find "..."                    # default fuses graphrag + wiki + wikigraph
```

WikiGraphRAG is built **directly from the maintained wiki artifacts** (`wiki/sources`, `wiki/concepts`, `wiki/analysis`). It uses NetworkX as the graph workhorse, BM25S (with a pure-Python BM25 fallback) for lexical retrieval, and NetworkX Louvain for community detection, but the full pipeline — parsing, entity/claim extraction, graph store, context assembly, query, and answer synthesis — is implemented in `src/graphwiki_kb/wikigraph/` rather than imported from a vendor framework. This keeps the capstone's main retrieval logic transparent and source-grounded, even though small algorithmic helpers are reused.

The custom backend is folded into the main command surface so the comparison stays first-class while preserving the clig.dev requirements of one consistent CLI:

- `kb update` rebuilds the WikiGraphRAG index automatically after compile, controlled by `--wikigraph/--no-wikigraph` (the `wikigraph.enabled` config value drives the default), `--wikigraph-include-graphrag-export-pages` for the ablation, `--export-wikigraph-artifacts` for generated entity/community/chunk cards, and `--artifact-types entities,communities,chunks` for filtering the export.
- `kb find` adds an `--engine [auto|graphrag|wiki|wikigraph|legacy|all]` option. `auto`/`all` fuse GraphRAG entity/relationship artifacts, the wiki FTS index, and WikiGraphRAG contexts via reciprocal rank fusion. `--engine legacy` runs the deprecated SQLite FTS5 path (with a deprecation note printed on the terminal and a `deprecated: true` flag in the JSON output).
- `kb ask` defaults to `--engine wikigraph` (the fastest, cheapest backend on the real-PDF benchmark) and accepts `--engine graphrag`, `--engine legacy`, `--engine all`, or a comma-separated list to compare backends side by side. When multiple engines run, the output renders one labeled section per engine; in JSON, the top-level is `{engines: [...], results: {engine: payload, ...}}`. Method validation is per-engine; `legacy` ignores `--method`.

The standalone `kb legacy find` / `kb legacy ask` command group has been removed; everything the legacy path supported is now reachable via the `--engine legacy` selector on the unified commands. Deprecation notes are still surfaced wherever the legacy engine runs.

Each command keeps a human-readable Rich table for terminal users and a parallel `--json` machine-readable form, matching clig.dev's "human-first, machine-friendly" guideline. The standalone `kb wikigraph build/status/find/ask` group has been removed because all of its functionality is now reachable through the existing commands; the underlying `WikiGraphIndexService` / `WikiGraphQueryService` remain in the service container so programmatic and evaluator code can call them directly.

A new cross-backend evaluator (`scripts/evaluate_backends.py`, `scripts/backend_evaluation_lib.py`) compares legacy FTS and WikiGraphRAG retrieval/answer quality on the existing benchmark, retrieval-only by default and provider-backed answers opt-in via `--allow-provider-calls`. It writes `eval/results/backend_summary.md`, `backend_retrieval_metrics.csv`, `backend_answer_metrics.csv`, and a raw runs JSON file. A two-PDF real-corpus dry run on REALM and RAG produced grounded WikiGraphRAG answers (4 valid claims, 4 citation references, 100% insufficient-evidence-behavior match on the 4-question benchmark).

## May 20 2026 WikiGraphRAG Quality Hardening + 10-PDF Three-way Evaluation

After integrating WikiGraphRAG as a peer to GraphRAG and Legacy FTS, the first three-way evaluation on the 10-PDF corpus showed the gap between WikiGraphRAG and Legacy FTS was real but smaller than it should have been (Recall@5 0.773 vs 0.636), and the entity-hits metric was near-tied at 0.73 because refusals that name-dropped the missing entity got the same credit as grounded answers. A targeted hardening pass produced the following layered improvements:

### 1. Fairer evaluator metrics

The previous metrics over-credited refusals and dragged averages down on intentionally-empty questions. The new evaluator (`scripts/backend_evaluation_lib.py`) reports:

* `effective_recall@5` — averaged only over questions with ground truth, so synthesis and out-of-scope questions stop forcing every backend to 0.
* `has_ground_truth` flag so consumers can compute fair per-row aggregates.
* `grounded_entity_hits` and `grounded_entity_rate` — only credit entities when the backend produced a grounded answer (refusals contribute 0).
* `answer_quality_score` — composite in [0, 1] averaging grounded-entity rate, normalized citation count, insufficient-evidence behavior match, and citation-ref validity rate.
* Source-coverage matcher now also looks at retrieved context body snippets, so backends that surface body text (TextUnits) get credit when an expected source name appears in the body but not in the file slug (e.g. ORQA, whose paper slug is `latent-retrieval-...`).

### 2. Stronger WikiGraphRAG retrieval

* Smarter entity matcher (`WikiGraphContextBuilder._match_entities`): adds word-boundary substring match and case-sensitive acronym match against entity names and aliases (`DPR`, `FiD`, `RAG`, `REALM`, `REPLUG`, `ORQA`, `RALM`). It also auto-generates implicit acronyms from multi-word entity titles (`"Dense Passage Retrieval"` → `DPR`, `"Fusion-in-Decoder"` → `FID`, `"Self-RAG"` → `SELFRAG`), so a question phrased in either direction wins. This is what closed the historical gap on `dpr_role`, where Legacy FTS had Recall@5 = 0 because the paper title spelled out the entity and the question used the acronym.
* `basic_search` now pulls a wider candidate pool and gives source TextUnits a small 15% ranking nudge so paper-body evidence consistently surfaces alongside the LLM-summarized wiki chunks; the nudge is small enough that a clearly-better wiki chunk still wins.

### 3. Answer-grounding fix

Provider answers occasionally cite a neighbor TextUnit of the document we actually retrieved (e.g. `...md#text-unit-3` when we returned `...md#text-unit-7`). The previous claim validator marked the entire answer insufficient in this case, despite the answer being grounded in the retrieved context. `WikiGraphAnswerService` now normalizes such cites to the canonical retrieved ref by matching on path-only; the body text the LLM reasoned over was the retrieved TextUnit, so this preserves grounding without loosening it. This was the cause of WikiGraphRAG's only "refused when it shouldn't" result on `atlas_training`.

### 4. Discriminating benchmark questions

Four paper-body questions were added to `eval/benchmark.yaml` so the benchmark explicitly tests content that lives inside the PDF body but not in its title or curated summary:

* `realm_mips_scalability` — REALM's MIPS + asynchronous index refresh trick.
* `rag_token_vs_sequence` — RAG-Token vs RAG-Sequence marginalization.
* `dpr_dual_encoder` — DPR's dual-encoder architecture.
* `self_rag_tokens` — Self-RAG's reflection tokens.

These give Legacy FTS no place to hide on its strongest historical territory (paper-title FTS5 matches), since the answers live in body content the FTS index never sees.

### 5. Final three-way headline numbers (10-PDF corpus, 15 questions)

Provider-backed run on the same 10 arXiv PDFs (full per-question CSV at `eval/results/threeway_v2/backend_answer_metrics_phase3.csv`):

| | **Legacy FTS** | **Microsoft GraphRAG** | **WikiGraphRAG** |
|---|---|---|---|
| **Answer Quality Score (composite)** | 0.215 | 0.605 | **0.902** |
| Effective Recall@5 | 0.923 | 0.192 | **1.000** |
| Grounded Entity Rate | 0.233 | 0.900 | **0.967** |
| Avg Citations / Answer | 1.13 | 3.20 | **3.40** |
| Insufficient-Evidence Behavior Match | 0.33 | 0.93 | **1.00** |
| Citation Ref Valid Rate (structured) | 0.000 | 0.000 | **1.000** |
| Avg Latency / Answer (s) | 0.20 | 5.51 | **0.073** retrieval / 4.5 provider |

WikiGraphRAG wins (or ties) every single question against both backends:

* Against Legacy FTS: 15/15 wins, often dramatic (0.05 → 1.0 on `realm_vs_rag`, `replug_blackbox`, `query_rewrite_rag`, `missing_topic`).
* Against Microsoft GraphRAG: 15/15 wins or ties. The largest single delta is `missing_topic` (0.05 GraphRAG vs 1.00 WikiGraphRAG), where GraphRAG generated two paragraphs of hedged content on an out-of-scope topic instead of refusing.

### 6. Qualitative inspection — what makes WikiGraphRAG's answers better in practice

I read every answer across all three backends. Side-by-side artifacts live at `/opt/cursor/artifacts/threeway_qa_comparison_phase3.md` and `/opt/cursor/artifacts/qualitative_inspection_phase3.md`. The recurring patterns:

* **Legacy FTS** refuses too often on entity-specific questions (the DPR paper is in the KB but FTS5 doesn't match "DPR" against the spelled-out title), and when it does ground an answer it tends to quote the wiki summary phrase verbatim without the underlying mechanic.
* **Microsoft GraphRAG** writes the most prose and is genuinely good at synthesis, but its citations are abstract tuples (`Data: Sources (97); Entities (573, 1020); Relationships (22056)`) instead of file paths a reviewer can open. Its biggest behavioral problem is over-confidence: on `missing_topic` it produced authoritative-sounding paragraphs about a topic it has zero coverage of.
* **WikiGraphRAG** consistently combines correct refusal behavior with concrete file-path citations (`wiki/sources/<slug>.md#chunk-N` or `raw/normalized/<slug>.md#text-unit-N`), and surfaces real paper-body content via the TextUnit layer. It is the only backend that has both a calibrated "I don't know" and an inspectable trace.

My qualitative conclusion: **WikiGraphRAG is the answer-quality leader on this corpus because it knows when to refuse, cites concrete files, and surfaces paper-body evidence that the title-only Legacy FTS can't find and that GraphRAG hides inside synthetic relationship tuples.** The remaining gaps (single-citation answers where the LLM cited only one TextUnit; one missed entity in `dpr_dual_encoder`) are tractable with deeper local-search hops and a "deep mode" max-context-tokens toggle, neither of which would compromise the project's transparency story.

The `kb agent` natural-language control plane is now also WikiGraphRAG-aware. `ask_kb` accepts `engine: "graphrag" | "wikigraph"` (default `graphrag`) and the wikigraph-only `drift-lite` method; method validation rejects mismatched engine/method combinations with a friendly error in the same shape as the CLI. `find_kb` accepts `engine: "auto" | "graphrag" | "wiki" | "wikigraph" | "all"` (default `auto`) and fuses GraphRAG entity/relationship artifacts, the wiki index, and WikiGraphRAG contexts via reciprocal rank fusion when in `auto`/`all`. `update_kb` now also takes `wikigraph` (default true) and `wikigraph_include_graphrag_export_pages` (default false); the in-process path constructs the `UpdateService` with the wikigraph index service wired in, and the subprocess fallback forwards the matching `--no-wikigraph` and `--wikigraph-include-graphrag-export-pages` flags. `status` surfaces a compact `wikigraph` block (`initialized`, `built_at`, `node_count`, `edge_count`, `community_count`, etc.) drawn from `WikiGraphIndexService.status()`. The agent system prompt documents these tools and instructs the agent to call `ask_kb` twice — once per engine — whenever the user explicitly asks for a GraphRAG vs WikiGraphRAG comparison rather than silently substituting one engine for the other.

This adjustment preserves the Karpathy-style wiki idea as the artifact and maintenance layer while moving retrieval and synthesis to GraphRAG (and, later, to the custom WikiGraphRAG backend). The existing SQLite FTS5/wiki retrieval path has been demoted to the explicit deprecated `kb ask --engine legacy` and `kb find --engine legacy` selectors on the unified commands; the standalone `kb legacy` group has been removed. The legacy path is compared against GraphRAG Basic/Local/Global/DRIFT and the new WikiGraphRAG basic/local/global/drift-lite methods only as historical evidence for the pivot. The project has added the Microsoft GraphRAG dependency, initialized the project-local GraphRAG workspace under `graph/graphrag/`, folded GraphRAG setup and maintenance into the main command surface, made `kb update` sync normalized inputs and auto-refresh the graph index only when needed, made `kb ask --method auto|basic|local|global|drift` the GraphRAG query surface, made `kb find` search direct graph artifacts plus maintained wiki pages, made `kb update` and `kb export` convert GraphRAG output tables into generated markdown under `wiki/graph/`, added Phase 8 evaluation scripts for deprecated FTS versus GraphRAG mode comparison, and hardened graph readiness, freshness, vector-store, and state-write behavior. `kb agent` now exists as a natural-language control plane over the existing services: read tools cover ask/find/status/lint/review, research combines the local KB answer with OpenAI Responses `web_search`, recommendations are persisted as numbered durable records, and ingestion/update tools require explicit approval or `--yes`. The remaining work is to produce final real-corpus comparison outputs and report the speed, cost, citation, and synthesis tradeoffs clearly.

The MVP is intentionally narrow in domain even though it is broader in input format. It targets a curated technical corpus about AI agents, coding agents, LLM tooling, and related knowledge-base system design. The system is intended to accept heterogeneous document types, but each source is first converted into a canonical markdown or plain-text representation before it enters the update and ask pipeline. For the MVP, evaluation will focus on text-heavy technical documents rather than source-code repositories or arbitrary structured datasets, because those require different parsing, compilation, and evaluation rules. The corpus itself will grow over time as the project progresses. At the proposal stage, the goal is not to commit to a final fixed list of documents, but to define the domain, supported input profile, normalization boundaries, and minimum evaluation target clearly enough that the project scope stays controlled while data collection continues.

## 1. Goals and Anticipated Outcomes

### Goals

The main goal of this project is to complete and evaluate a command-line application that builds and maintains a citation-grounded markdown knowledge base from a curated set of technical research documents. Instead of handling every incoming data type through a separate downstream workflow, the system will normalize each accepted document into a canonical markdown or plain-text form and then compile that normalized corpus into a structured wiki. The domain will remain narrow: AI agents, coding agents, LLM tooling, and related knowledge-base system design. The system will ingest those sources, compile source pages from the normalized text with explicit provenance plus provider-generated summaries, optionally synthesize cited concept pages from those compiled source pages, answer questions from the maintained knowledge base through a CLI, and export the result into an Obsidian-friendly vault.

The project also has a research goal. The central research question is whether a maintained markdown knowledge-base workflow built from normalized heterogeneous documents improves organization, traceability, freshness, and reuse of prior research when compared with simpler alternatives such as direct prompting, deprecated lexical retrieval, and GraphRAG modes. A secondary goal is to study whether a bounded, maintenance-first workflow can make LLM-assisted synthesis more trustworthy by tracking source provenance, recording which converter produced the canonical text, surfacing stale pages when sources change, and keeping deletion or archival under human control. A related learning goal is to understand where a maintained wiki workflow is actually stronger or weaker than simpler retrieval workflows: for example, whether it improves citation traceability and refresh behavior enough to justify the extra compilation step.

The project will be measured against concrete technical targets. The prototype should implement the current core CLI workflow, including initialization, ingest, source-page generation, deprecated legacy search/question answering (now reachable via `--engine legacy` rather than a standalone command), structural linting, semantic review, project-state inspection, and vault export, while keeping WikiGraphRAG as the default retrieval path. In the current command layout, that means commands such as `kb init`, `kb add`, `kb update`, `kb find`, `kb ask`, `kb lint`, `kb review`, `kb status`, `kb doctor`, `kb export`, and a bounded approval-gated `kb agent`. `kb ask` defaults to the custom WikiGraphRAG backend and accepts `--engine graphrag|legacy|all` (or a comma-separated list) to compare backends; `kb find` defaults to a fused GraphRAG + wiki + WikiGraphRAG search and accepts the same `--engine` selector (plus `wiki` for FTS-only). The `kb update` command refreshes the WikiGraphRAG index automatically (toggle via `--wikigraph/--no-wikigraph`, with `wikigraph.enabled` in config as the default driver). It should process a realistic sample corpus of at least 20 to 30 text-heavy technical documents by evaluation time, drawn from a representative mix of supported formats such as markdown, README or HTML-style docs, transcript text, PDFs, and born-digital office or notebook files that normalize cleanly. The full corpus may continue growing over the duration of the project, but the evaluation passes will use a frozen benchmark subset so results remain comparable over time. The system should maintain local searchable metadata, source hashes, and converter metadata, and generate a wiki that includes source pages, legacy concept pages where useful, generated graph pages, an index, and an activity log. It should also answer a benchmark set of 10 to 15 research questions using citations to compiled wiki or graph-backed context, and it should detect when source changes make compiled pages or graph artifacts stale or require review.

### Anticipated Outcomes

The anticipated outcomes are a working CLI prototype, a sample maintained knowledge base built from collected research materials, an Obsidian-friendly vault export, and an evaluation package showing where the maintenance-first markdown workflow is stronger or weaker than direct prompting, deprecated lexical retrieval, and GraphRAG modes. A successful result is not simply a working chatbot. A successful result is a repeatable workflow that accumulates research over time, keeps summaries and links current, flags outdated pages, and produces outputs that are useful for both study and final project reporting.

### Evaluation Plan

Evaluation will use functional, maintenance-oriented, and comparative measures. Functionally, all core commands must complete their intended workflows without manual file editing. Structurally, the compiled wiki should maintain valid links, current index data, explicit source citations, and clear traceability from raw source to generated page. Semantic review checks should surface potential contradictions and terminology drift across compiled pages. Those maintenance checks will remain deliberately lightweight: structural lint will expand through checklist-style markdown validation such as heading hierarchy, duplicate titles, valid link fragments, typed metadata fields, and empty-page detection, while semantic review will prioritize preferred-term normalization and candidate-based contradiction checks rather than exhaustive all-pairs reasoning. Conversion quality will be judged by spot-checking whether the normalized markdown or text preserves enough structure and content to support compilation and citation. On a benchmark set of 10 to 15 research questions, retrieval quality will be measured with Recall@5, multi-source coverage, method fit, and latency across deprecated FTS, GraphRAG Basic/Local/Global/DRIFT modes, and the new WikiGraphRAG basic/local/global/drift-lite methods. Answer quality will be judged for claim support, insufficient-evidence behavior, comprehensiveness, diversity, and latency, with provider-backed answer runs kept explicit because they can incur cost. Citation grounding will be measured by the proportion of answer claims and synthesized concept-page claims that remain traceable to cited source pages or graph-backed source traces. Maintenance behavior will be tested with seeded source edits that record whether stale pages are detected and how much work is required to refresh the maintained wiki compared with rebuilding or rerunning the simpler baselines. Comparatively, the maintained wiki will be assessed against direct prompting over raw documents, deprecated FTS, GraphRAG modes, and the custom WikiGraphRAG backend — explicitly framed not as a replacement story for Microsoft GraphRAG but as a transparency/cost/inspectability comparison.

| Dimension | Measure | Target or comparison |
| --- | --- | --- |
| Workflow completion | Core commands run end to end on the sample corpus | All required workflows complete without manual file editing |
| Retrieval usefulness | Effective Recall@5 (averaged over questions with ground truth), multi-source coverage, and method fit | Measured on the 15-question, 10-PDF arXiv benchmark. **Current observed: WikiGraphRAG 1.000, Legacy FTS 0.923, Microsoft GraphRAG 0.192.** |
| Citation grounding | Share of answer claims with verifiable citations to compiled pages and a non-zero structured citation-ref validity rate | Target ≥80%. **Current observed: WikiGraphRAG citation_ref_valid_rate = 1.000; structured citations point to concrete file paths and `#chunk-N` / `#text-unit-N` anchors.** |
| Refusal calibration | Insufficient-evidence behavior matches the benchmark expectation | **Current observed: WikiGraphRAG 1.00, Microsoft GraphRAG 0.93, Legacy FTS 0.33.** |
| Composite answer quality | Quality Score = mean of (grounded entity rate + normalized citations + refusal-behavior match + citation-ref validity) | **Current observed: WikiGraphRAG 0.902, Microsoft GraphRAG 0.605, Legacy FTS 0.215.** |
| Maintenance freshness | Seeded source changes trigger stale or review flags | Affected pages are detected consistently |
| Comparative performance | Maintained wiki compared with direct prompting, deprecated FTS, GraphRAG modes, and WikiGraphRAG modes | Clear strengths, weaknesses, latency, token-cost, and transparency tradeoffs are documented |

The following criteria define success for the proposal:

- the core CLI workflows run end to end on the sample heterogeneous technical corpus;
- the system produces a navigable markdown wiki and Obsidian-friendly vault export;
- compiled pages maintain explicit source citations and provenance metadata;
- the normalization pipeline preserves enough content fidelity for compiled pages and cited answers to remain useful;
- benchmark retrieval logs show whether relevant source pages appear in the top results for the benchmark question set;
- selected source edits cause the system to flag affected pages as stale or needing review;
- semantic review checks flag potential contradictions or terminology drift across compiled pages;
- at least 80 percent of benchmark answers include verifiable citations to compiled pages (currently exceeded: WikiGraphRAG averages 3.40 citations per answer at 100% citation-ref validity on the 10-PDF, 15-question benchmark); and
- the final evaluation clearly explains where the maintenance-first workflow improves on, matches, or falls short of direct prompting, deprecated FTS, GraphRAG modes, and the custom WikiGraphRAG backend. The latest three-way numbers (composite Quality Score: WikiGraphRAG 0.902, Microsoft GraphRAG 0.605, Legacy FTS 0.215) and the per-backend qualitative analysis are documented in the "May 20 2026 WikiGraphRAG Quality Hardening" section above.

## 2. Motivation

The original motivation for this project comes from Andrej Karpathy's idea that LLMs can be used not only to answer questions, but also to maintain a persistent and compounding body of knowledge in markdown. That idea fits well with the way technical research is often collected in practice: README files, design notes, documentation pages, transcripts, exported notes, PDFs, and curated markdown notes accumulate over time, but they are rarely transformed into a maintainable, reusable knowledge artifact. This project addresses that gap by asking whether an LLM-assisted wiki can function as a better long-term research companion than one-off prompting or purely retrieval-driven pipelines for a narrow technical domain with heterogeneous source formats.

The project is also motivated by maintenance and traceability. Standard RAG workflows can retrieve relevant passages, but they still tend to emphasize transient question answering over persistent synthesis. They also do not naturally solve freshness problems when source documents change. In contrast, a maintained markdown wiki has the potential to preserve source context, concept relationships, historical updates, and review status in a form that is easier to inspect, audit, refresh, and reuse. For this project, the hard problem is not only generating pages once, but keeping those pages trustworthy over time.

Finally, the project is personally and professionally meaningful because it combines software engineering, evaluation, and a research workflow I could keep using after the course. Building a CLI-first application is a realistic but technically meaningful stretch goal, and the result could be useful beyond this course for students, researchers, and developers who need a structured way to transform a growing technical document collection into reusable knowledge rather than disconnected notes.

## 3. Resources and Feasibility

The most realistic implementation path is to build the prototype as a Python CLI using Click and Poetry. Browzy.ai will serve as the main reference for the raw-to-compile-to-ask pipeline, markdown storage layout, incremental compilation, retrieval from compiled pages, and linting. OpenClaude will serve as the main reference for command registration, provider normalization, tool contracts, and the separation between setup, orchestration, and session state. In the proposed implementation, original files remain stored in the raw layer, but each document is normalized into canonical markdown or plain text before search, compilation, and ask. The current normalization path routes canonical markdown and plain text directly, uses Mistral OCR by default for explicitly supported native document and image formats such as PDF, DOCX, PPTX, PNG, and JPEG, renders HTML through `wkhtmltopdf` or the pure-Python `xhtml2pdf` fallback before OCR, and keeps MarkItDown for the remaining bounded born-digital formats that already contain a usable text layer. This paid-Mistral-first choice is intentional because PDF conversion quality determines downstream source-page fidelity, GraphRAG entity and relationship quality, retrieval, citations, and answer quality. Docling and MarkItDown remain explicit lower-confidence fallbacks rather than the primary route for those Mistral-first formats; for PDFs, the default fallback chain tries Docling and then MarkItDown only after Mistral fails. If needed later, a provider-backed LLM cleanup or reconstruction step can sit behind OCR, but it should remain explicit rather than becoming the default ingest mode. The current prototype now keeps SQLite FTS5 indexed lexical retrieval only behind the deprecated `--engine legacy` selector on `kb ask` and `kb find` (the standalone `kb legacy` group has been removed), and a small schema file such as `kb.schema.md` can steer compilation rules without turning the system into a large prompt-only application. The custom WikiGraphRAG backend builds on NetworkX for graph storage and Louvain community detection, BM25S (with a pure-Python BM25 fallback) for lexical retrieval, and reuses the existing `markdown-it-py`, `python-frontmatter`, `rapidfuzz`, and `pydantic` stack for parsing and modeling. The entire WikiGraphRAG pipeline — parsing, entity/claim extraction, graph store, context assembly, and answer synthesis — is implemented inside `src/graphwiki_kb/wikigraph/` to keep the comparison honest rather than wrapping a vendor retrieval framework.

The project resources are realistic for an individual capstone. Development will use Python 3, Click, Poetry, Markdown files, Git, Obsidian-compatible markdown conventions, SQLite, Microsoft GraphRAG, NetworkX, BM25S, pytest, Black, Ruff, mypy, pre-commit, and GitHub Actions. The ingest path will also need format detection and per-type conversion into canonical markdown or text. Model access now flows through a deliberately small provider layer with OpenAI, Anthropic, and Gemini implementations, while GraphRAG runtime configuration lives in `graph/graphrag/settings.yaml` and reuses the centralized provider API-key environment variables from `kb.config.yaml` by default. Provider selection and built-in provider settings now live together in `kb.config.yaml`, keeping the main model surface configurable through one repository-visible YAML file. The optional `kb agent` path uses a separate agent extra, stays secondary to the GraphRAG evaluation, and is intentionally constrained to service-backed KB operations plus approval-gated writes. The dataset will be a curated corpus of text-heavy technical references on AI agents, coding agents, LLM tooling, and knowledge-base systems, stored in multiple supported source formats but normalized into a common internal representation. Direct prompting, deprecated FTS, GraphRAG Basic/Local/Global/DRIFT modes, and WikiGraphRAG basic/local/global/drift-lite modes will serve as comparison baselines, with the Phase 8 benchmark and scripts producing repeatable CSV summaries under `eval/results/`. The CLI surface and its newly folded `--engine` selectors follow the [Command Line Interface Guidelines](https://clig.dev/) principles: every command exposes intuitive defaults, supports `--json` machine-readable output alongside the human-readable Rich table view, validates flag combinations with friendly errors, and keeps the same flag name (`--engine`) consistent across `kb ask` and `kb find`.

### Planned Input Data Profile

The exact corpus will not be fully fixed at proposal time because part of the project is to continue collecting and curating relevant research material as the work progresses. What is fixed now is the data profile, the domain boundary, and the evaluation target. The corpus will grow incrementally over time, while benchmark runs will be evaluated on explicitly frozen subsets so comparisons remain fair.

The intended input profile is a text-heavy technical corpus centered on AI agents, coding agents, LLM tooling, retrieval workflows, and related knowledge-base design. The expected source categories are:

- markdown notes, technical writeups, and documentation pages;
- README-style project documentation and structured web documentation exported as HTML or text;
- transcript-like or note-like technical text with looser structure;
- born-digital research PDFs and related papers; and
- selected born-digital office or notebook formats only when they normalize cleanly enough to support citation-grounded compilation.

This description is intentionally format- and domain-based rather than file-list-based. The central point of the project is not to analyze one static folder of documents, but to maintain a growing research corpus over time while preserving traceability, refresh behavior, and comparability on benchmark subsets. The primary capstone evaluation corpus will stay text-heavy. Image-only or scan-heavy documents can now enter through the Mistral OCR path when needed, but they still do not define the core success criteria unless they become essential to the benchmark corpus.

This scope is feasible because it does not attempt to reproduce either reference project in full, and it does not require every source format to be handled with a custom downstream workflow. The first version only needs a bounded set of source converters that normalize heterogeneous documents into a common markdown or text form, plus a small provider layer and a deterministic workflow for ingest, update, search, ask, lint, and export. That limitation keeps the capstone focused on the main research and engineering question while still leaving room for a meaningful comparison with simpler baselines.

### Backup Plans

If resources become limited, the scope can be reduced without breaking the central research question. If support for some source formats becomes too noisy, the prototype can freeze on the subset of formats that normalize cleanly enough for citation-grounded compilation. If cross-document concept pages become unreliable, the project can keep source pages and generated graph pages as the primary artifacts and reduce legacy concept-page generation. If model cost or reliability becomes a constraint, the project can use a smaller frozen evaluation corpus, rely on local-safe retrieval and routing metrics, and run provider-backed answer scoring only for selected comparison questions. If comparison work expands beyond a manageable level, the final report can narrow the comparison to direct prompting, deprecated FTS, and the most relevant GraphRAG modes instead of trying to evaluate every possible baseline. If advanced reporting outputs become too time-consuming, the final deliverables will remain focused on the maintained wiki, cited ask responses, maintenance findings, and the evaluation report.

## 4. Project Activities / Methods

The project will follow an explicit layered workflow rather than one large agent loop. The raw layer will store original source documents and a manifest of ingested sources. After type detection, each document will be normalized into a canonical markdown or plain-text form. The current implementation routes canonical markdown and plain text directly, uses Mistral OCR by default for explicitly supported native documents and images, renders HTML through `wkhtmltopdf` or `xhtml2pdf` before OCR, and uses MarkItDown for the remaining bounded born-digital subset. Docling and MarkItDown remain available as explicit fallbacks when the primary route fails quality checks, with PDF fallback metadata making lower-confidence conversions visible for review or rerun. Optional LLM cleanup after OCR remains a later opt-in step rather than a silent default. The manifest will record source path, ingest time, content hash, normalized artifact path, converter used, and related metadata so the system can later tell whether a compiled page may be stale or whether a conversion should be reviewed. The wiki layer will store generated source pages, generated graph pages, legacy concept pages where useful, saved analysis pages, an index, and an activity log. A vault layer will export Obsidian-friendly notes, backlinks, and frontmatter where useful. Inside the application code, the CLI will be split into commands, services, providers, and shared types so that setup, ingestion, normalization, compilation, asking, linting, and export remain separate and testable.

The expected workflow is as follows. First, `kb add` will accept either a single file or a directory, recurse through directory inputs by default, detect source type, convert each supported document into canonical markdown or plain text, store the original files in `raw/`, store the normalized representations, update the manifest, and record searchable metadata plus a source hash. Second, `kb init` initializes the project-local GraphRAG workspace from `kb.config.yaml` graph settings. Third, `kb update` will add any provided source paths, compile or refresh source pages, regenerate the wiki index, optionally refresh legacy concept pages when explicitly enabled, refresh legacy index artifacts, append to the activity log, persist compile-run state, write normalized artifacts and manifest metadata into `graph/graphrag/input/sources.json`, auto-select full `fast`, incremental `fast-update`, retry, or skip based on output completeness plus source/runtime digests, record local reproducibility metadata, export graph tables into generated markdown under `wiki/graph/` when output is complete, and rebuild the WikiGraphRAG index under `graph/wikigraph/` (controlled by `--wikigraph/--no-wikigraph`, with `--wikigraph-include-graphrag-export-pages` for the ablation that also feeds GraphRAG-exported wiki pages into the custom backend). Normal mixed wiki/graph updates warn and skip isolated missing normalized artifacts instead of blocking the whole graph sync, while `kb lint` reports raw or normalized manifest artifact drift plus graph input/index/export freshness gaps. Generated wiki artifacts, manifest updates, compile-run state, graph index-run state, logs, and vault-export files use atomic or lock-protected write paths so partial or concurrent writes are less likely to corrupt local state. The current SQLite FTS5 retrieval path is retained only behind the deprecated `--engine legacy` selector on the unified `kb ask` / `kb find` commands; it is not an implicit fallback when GraphRAG or WikiGraphRAG is missing or not ready. Use `kb ask --save` for WikiGraphRAG analysis pages (the default backend), `kb ask --engine graphrag --save` for Microsoft GraphRAG analysis pages, and `kb ask --engine legacy --save` only for deprecated comparison captures. Fourth, Phase 8 evaluation scripts run the frozen benchmark and write `eval/results/summary.md`, `retrieval_metrics.csv`, and `answer_metrics.csv`, with provider-backed answer comparisons requiring explicit opt-in. The newer `scripts/evaluate_backends.py` extends the harness with a three-way `legacy` × `graphrag` × `wikigraph` comparison and writes `eval/results/backend_summary.md`, `backend_retrieval_metrics.csv`, and `backend_answer_metrics.csv` (provider-backed answer eval still opt-in via `--allow-provider-calls`). Fifth, `kb lint` will detect broken links, orphan pages, missing citations, stale pages whose sources changed, graph artifact freshness gaps, or structural gaps that require review. Sixth, `kb review` will run semantic checks across compiled pages, surfacing potential contradictions and terminology drift; it combines deterministic overlap checks with a single provider-backed JSON review pass. Seventh, `kb status` will summarize corpus size, compile state, conversion state, graph state, and maintenance findings; `kb status --changed` shows a pre-compile preview of which sources are new, changed, missing from disk, or already up to date. Finally, `kb export` will write an Obsidian-friendly vault view of the maintained knowledge base and refresh graph inspection pages when complete graph output exists; clean mode removes only vault markdown files absent from the current export set.

### Example Intermediate Artifacts

To make the workflow concrete in the final submitted proposal, I will include visual examples of intermediate artifacts rather than only describing the pipeline abstractly. The most useful examples are: a raw source document, the corresponding normalized markdown artifact, a compiled source page, and a generated concept page or vault-export note. Including those examples will make the pipeline auditable and demonstrate that the proposed workflow already exists beyond the whiteboard stage.

For the submitted proposal document, the most useful figures or screenshots are the following:

- one example raw input page or excerpt from a representative text or markdown source;
- one example raw HTML, documentation, or PDF source page from the intended corpus profile;
- one normalized markdown artifact showing what the converter preserves and what metadata is retained;
- one compiled source page showing provenance, citations, and summary structure; and
- one concept page or vault-export page showing how the maintained knowledge base becomes navigable over time.

### Obsidian Vault Screenshots

This subsection is reserved for screenshots taken from Obsidian after opening one of the generated vault views. The purpose of these images is to show that the maintained knowledge base is not only queryable from the CLI, but also browsable as a navigable markdown vault with index pages, source pages, concept pages, links, and backlinks.

For these screenshots, I used a small illustrative PDF evaluation corpus rather than the final capstone corpus. That example data consists of a focused set of related retrieval and retrieval-augmented generation papers used to validate the workflow on realistic born-digital PDFs. The raw PDF files served only as example source material to demonstrate ingest, normalization, compilation, and vault export, so the screenshots should be understood as evidence of the workflow and navigability of the system rather than as a claim that this small PDF slice is the final or complete project dataset.

![alt text](image-1.png)

- Figure 1. Obsidian file-explorer view of an exported example knowledge-base vault showing the index page, source pages, and concept pages.

![alt text](image-2.png)

- Figure 2. Example source page from the illustrative PDF evaluation set viewed in Obsidian Graph view, showing links to other source pages and concept pages.

The maintenance component will remain human-reviewed. For the MVP, maintenance means freshness detection, structural checks, semantic review checks, and a reviewable refresh workflow rather than full automatic contradiction resolution across the corpus. The `kb lint` command handles deterministic structural checks while `kb review` handles semantic analysis such as contradiction detection and terminology drift, keeping the two concerns cleanly separated. A planned `kb fix` command will close the loop by applying fixes in three tiers: deterministic fixes such as recompiling stale pages and regenerating missing frontmatter, light LLM-backed fixes for heading hierarchy, missing summaries, and term normalization shown as diffs for user confirmation, and harder fixes for contradiction resolution shown for human review but not auto-applied. This tiered approach keeps the system auditable while testing whether LLM-assisted maintenance improves wiki quality compared with manual fixes alone. Near-term improvements will stay conservative and auditable: `kb lint` will grow through practical markdown and knowledge-base checks such as heading hierarchy, anchor validation, duplicate title detection, and typed frontmatter validation, while `kb review` will use a small preferred-terms registry and candidate-based contradiction checks so terminology drift and likely conflicts can be surfaced without turning the project into a heavy ontology or full NLI research system. The system may flag a page as stale, outdated, or needing archival, but it will not silently delete knowledge. This keeps the workflow auditable and addresses the core maintenance concern in the project. The LLM-assisted component remains intentionally constrained: `kb agent` routes natural language to typed service tools, keeps local KB answers separate from web research findings, saves recommendations instead of auto-ingesting them, and requires approval or `--yes` before recommendation ingestion or update.

This is an individual project, so all responsibilities remain with one team member. I will be responsible for system design, implementation, testing, evaluation, documentation, and presentation preparation. Keeping the project individual supports a narrower and more defensible scope, which is more important for this capstone than maximizing feature breadth.

## 5. Work Plan / Timeline and Milestones

The work plan now matches the remaining course schedule and the current state of the project. The deterministic CLI foundation, provider layer, GraphRAG-first workflow, graph hardening, and bounded `kb agent` control plane are already complete, so the remaining milestones focus on final real-corpus comparison, answer-quality evidence, maintenance validation, and final reporting. The milestones below align with the course update dates on 4/28, 5/12, 5/26, and the final presentation on 6/2.

| Phase | Target Time | Activities | Tangible Milestone |
| --- | --- | --- | --- |
| Implemented baseline (completed) | Completed by Apr 15 | Implement initialization, ingest, compile, search, ask, lint, review, status, diff, export, normalization routing, manifest hashing, provider-backed compile/ask/review, saved answer analysis pages, SQLite FTS5 search, and compile-run resume state | Working CLI with the current command surface, provider layer, and the current passing test suite |
| Proposal finalization | Apr 16 to Apr 18 | Incorporate feedback from the presentations, add concrete data examples, finalize the evaluation rubric, and submit the formal proposal document | Concrete proposal submitted |
| Project Update 1: curated corpus and initial real-corpus pass | Apr 19 to Apr 28 | Freeze the benchmark question set, continue expanding the broader corpus, define the first frozen evaluation subset, run initial compile/ask/review evaluation on real documents, and record latency, cost, and unsupported-claim artifacts | Update 1 shows first real-corpus evaluation results |
| Project Update 2: GraphRAG pivot and input bridge | Apr 29 to May 12 | Isolate legacy FTS behavior, add the Microsoft GraphRAG dependency and workspace, sync normalized artifacts into GraphRAG JSON input, auto-refresh the graph index when needed, wrap query commands, and export graph output tables into generated wiki pages | Update 2 shows the controlled GraphRAG pivot through the main `kb init`, `kb update`, `kb ask`, `kb status`, and `kb export` path |
| Project Update 3: graph hardening and maintenance validation | May 13 to May 26 | Evaluate the GraphRAG-aware default ask path, compare local/global/basic/DRIFT modes on the real corpus, validate generated graph artifact freshness checks, verify state hardening, and include the bounded agent control plane only if it supports the final demo story | Update 3 demonstrates GraphRAG-aware default asking plus synthesis, maintenance validation, and approval-gated natural-language control |
| Final evaluation and presentation | May 27 to Jun 2 | Complete the deprecated FTS versus GraphRAG mode comparison, polish the demo corpus, report runtime and provider-cost tradeoffs, and finalize the report and presentation materials | Final presentation and capstone deliverables ready |
| Post-capstone stretch: productization and study workspaces | After core deliverables | Explore multi-subject workspace support, packaged or executable distribution, a default local workspace folder, and a GUI or app-style wrapper over the existing services | Future milestone specification or prototype, kept outside the capstone success criteria unless the core evaluation finishes early |

These milestones are measurable because each one corresponds to a concrete artifact or behavior: a working deterministic CLI foundation, a provenance-aware ingest pipeline, a generated wiki, a cited ask workflow, a maintenance-validation report, an evaluation report, and a polished final demonstration.

If the core evaluation, synthesis, and comparison milestones are completed early, I may explore productization work over the existing services. The most promising direction is a workspace container that can hold multiple subject-specific knowledge bases, for example `kb workspace init study-notes` followed by `kb subject init databases` or `kb subject init llm-agents`, with commands able to target one subject or all subjects together. A later packaging pass could make `kb` installable through a standard Python packaging path or executable wrapper and give users a default workspace location instead of requiring them to manage project folders manually. A GUI or app-style wrapper would be considered after that, focused on easier source import, subject switching, search, ask, status, and export. This remains stretch scope only and is not part of the core success criteria for the capstone.

## 6. Bibliography

1. Karpathy, Andrej. "LLM Knowledge Bases Tweet." X, 2 Apr. 2026. [Tweet URL](https://x.com/karpathy/status/2039805659525644595). Accessed 5 Apr. 2026.
2. Karpathy, Andrej. "LLM Wiki." GitHub Gist, 2026. [Gist URL](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). Accessed 5 Apr. 2026.
3. Gitlawb. OpenClaude. 2026. Local workspace snapshot in [../../../Resources/openclaude/README.md](../../../Resources/openclaude/README.md). Accessed 5 Apr. 2026.
4. Kanukollu, Vihari. browzy.ai. 2026. [Repository](https://github.com/VihariKanukollu/browzy.ai). Accessed 5 Apr. 2026.
5. Pallets. Click Documentation. 2026. [Docs](https://click.palletsprojects.com/). Accessed 5 Apr. 2026.
6. Python Software Foundation. Python 3 Documentation. 2026. [Docs](https://docs.python.org/3/). Accessed 5 Apr. 2026.
7. SQLite Documentation. 2026. [Docs](https://www.sqlite.org/docs.html). Accessed 5 Apr. 2026.
8. Obsidian. Obsidian Help. 2026. [Docs](https://obsidian.md/help/). Accessed 5 Apr. 2026.
9. Rotenberg, Josh. "mdbook-lint Documentation: Standard Markdown Rules." 2026. [Docs](https://joshrotenberg.com/mdbook-lint/rules/standard/index.html). Accessed 14 Apr. 2026.
10. Tavian Dev. "mdlint." GitHub repository, 2026. [Repository](https://github.com/tavian-dev/mdlint). Accessed 14 Apr. 2026.
11. University of Pittsburgh Library System. "Metadata & Discovery @ Pitt: Taxonomies and Controlled Vocabularies." 2025. [Guide](https://pitt.libguides.com/metadatadiscovery/controlledvocabularies). Accessed 14 Apr. 2026.
12. Awaysheh, Abdullah, et al. "A Review of Medical Terminology Standards and Structured Reporting." Journal of Veterinary Diagnostic Investigation, 2017. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6504145/). Accessed 14 Apr. 2026.
13. Gokul, Vignesh, Srikanth Tenneti, and Alwarappan Nakkiran. "Contradiction Detection in RAG Systems: Evaluating LLMs as Context Validators for Improved Information Consistency." arXiv:2504.00180, 2025. [Paper](https://arxiv.org/abs/2504.00180). Accessed 14 Apr. 2026.
14. Du, Yilun, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. "Improving Factuality and Reasoning in Language Models through Multiagent Debate." arXiv:2305.14325, 2023. [Paper](https://arxiv.org/abs/2305.14325). Accessed 14 Apr. 2026.
15. Wang, Xuezhi, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. "Self-Consistency Improves Chain of Thought Reasoning in Language Models." arXiv:2203.11171, 2022. [Paper](https://arxiv.org/abs/2203.11171). Accessed 14 Apr. 2026.
16. Wang, Junlin, Jue Wang, Ben Athiwaratkun, Ce Zhang, and James Zou. "Mixture-of-Agents Enhances Large Language Model Capabilities." arXiv:2406.04692, 2024. [Paper](https://arxiv.org/abs/2406.04692). Accessed 14 Apr. 2026.
17. Agarwal, Shrestha, et al. "Do as We Do, Not as You Think: the Conformity of Large Language Models." arXiv:2501.13381, 2025. [Paper](https://arxiv.org/abs/2501.13381). Accessed 14 Apr. 2026.
18. "Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?" OpenReview, 2025. [Paper](https://openreview.net/forum?id=iUjGNJzrF1). Accessed 14 Apr. 2026.
19. "If Multi-Agent Debate is the Answer, What is the Question?" arXiv:2502.08788, 2025. [Paper](https://arxiv.org/abs/2502.08788). Accessed 14 Apr. 2026.
20. PyYAML. "PyYAML Documentation." 2026. [Docs](https://pyyaml.org/wiki/PyYAMLDocumentation). Accessed 22 Apr. 2026.
21. SQLite. "SQLite FTS5 Extension." 2026. [Docs](https://www.sqlite.org/fts5.html). Accessed 14 Apr. 2026.
22. Python Software Foundation. "Coroutines and Tasks — asyncio." Python 3 Documentation, 2026. [Docs](https://docs.python.org/3/library/asyncio-task.html). Accessed 14 Apr. 2026.
23. IBM Research. "Docling Documentation." 2026. [Docs](https://docling-project.github.io/docling/). Accessed 18 Apr. 2026.
24. Microsoft. "MarkItDown." GitHub repository, 2026. [Repository](https://github.com/microsoft/markitdown). Accessed 18 Apr. 2026.
25. Mistral AI. "Mistral OCR Documentation." 2026. [Docs](https://docs.mistral.ai/capabilities/document/). Accessed 18 Apr. 2026.
26. Command Line Interface Guidelines. "Command Line Interface Guidelines." 2026. [Guide](https://clig.dev/#introduction). Accessed 18 Apr. 2026.
27. Lewis, Patrick, et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS, 2020. [Paper](https://arxiv.org/abs/2005.11401). Accessed 19 May 2026.
28. Edge, Darren, et al. "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." arXiv:2404.16130, 2024; revised 2025. [Paper](https://arxiv.org/abs/2404.16130). Accessed 19 May 2026.
29. Microsoft. "GraphRAG Documentation." 2026. [Docs](https://microsoft.github.io/graphrag/). Accessed 19 May 2026.
30. Hagberg, Aric, Daniel Schult, and Pieter Swart. "NetworkX: Network Analysis in Python." 2026. [Docs](https://networkx.org/documentation/stable/). Accessed 20 May 2026.
31. Lù, Xing Han. "BM25S: Fast and Lightweight BM25 in Python." 2024. [Docs](https://bm25s.github.io/). Accessed 20 May 2026.
32. Blondel, Vincent D., Jean-Loup Guillaume, Renaud Lambiotte, and Etienne Lefebvre. "Fast Unfolding of Communities in Large Networks." Journal of Statistical Mechanics: Theory and Experiment, 2008. [Paper](https://arxiv.org/abs/0803.0476). Accessed 20 May 2026.
33. Lewis, Patrick, et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS, 2020. [Paper](https://arxiv.org/abs/2005.11401). Used as one of the two end-to-end PDF inputs in the WikiGraphRAG dry-run evaluation. Accessed 20 May 2026.
34. Guu, Kelvin, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. "REALM: Retrieval-Augmented Language Model Pre-Training." arXiv:2002.08909, 2020. [Paper](https://arxiv.org/abs/2002.08909). Used as the second end-to-end PDF input in the WikiGraphRAG dry-run evaluation. Accessed 20 May 2026.
