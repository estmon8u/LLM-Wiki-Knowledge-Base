# Capstone Project Status Overview

Date: 2026-04-24
Updated: 2026-05-22 during the sequential WikiGraphRAG branch documentation pass. Commit `0e6eb47` originally reframed this status note around main-line GraphRAG hardening, the read-only `kb agent` MVP, real-data validation, and lint/type/test cleanup.

## Current Position

The project has moved from scaffold+implementation through the GraphRAG pivot into a mostly stable graph-backed system. Commit `6966831` adds the final honest WikiGraphRAG vs Microsoft GraphRAG verification report, so the remaining work is presentation polish and any explicitly scoped follow-up rather than another architecture rebuild.

1. Proposal, architecture, and the original command surface are complete.
2. Simplification and deterministic baseline are complete.
3. The SQLite FTS5/wiki retrieval path has been demoted to explicit deprecated legacy behavior.
4. Microsoft GraphRAG is installed, and the dedicated `graph/graphrag/` workspace is initialized.
5. `kb init` creates and syncs the GraphRAG workspace through signature-aware Python entrypoint adapters with documented CLI fallback.
6. `kb update` writes GraphRAG JSON input from normalized artifacts and manifest metadata, then auto-selects full index, incremental update, retry, or skip based on source/runtime digests and output completeness.
7. `kb ask --engine graphrag --method auto|basic|local|global|drift` exposes Microsoft GraphRAG query modes explicitly, while default `kb ask` uses WikiGraphRAG.
8. `kb find` searches direct GraphRAG entity/relationship artifacts plus the maintained wiki index, then deduplicates and ranks the combined results.
9. `kb update` and `kb export` export GraphRAG output tables into generated markdown pages under `wiki/graph/`.
10. `kb status`, `kb doctor`, and `kb lint` include GraphRAG readiness, vector-store health, freshness, stale input/index/export checks, and strict readiness modes.
11. Phase 8 evaluation scripts compare deprecated FTS against GraphRAG Basic, Local, Global, and DRIFT with local-safe defaults and opt-in provider-backed answer runs.
12. `kb agent` is implemented as a bounded natural-language control plane over existing services, including local KB ask/find/status/lint/review, web-backed research, durable source recommendations, approved recommendation ingestion, and approved `kb update`; commit `4a96bc6` documented that boundary across the README, start guide, and architecture docs.
13. State hardening now protects config migrations, manifest writes, compile-run state, graph index-run state, wiki logs, and GraphRAG workspace operations.
14. Wiki artifacts remain the inspectable provenance, maintenance, and export layer.

In short, the core workflow works end-to-end, and the pivot keeps that work instead of discarding it. The project is now graph-backed while preserving the original wiki system as the artifact layer. Commit `04a1b9b` made WikiGraphRAG the default `kb ask` engine, kept Microsoft GraphRAG explicit through `--engine graphrag`, and exposed old FTS retrieval only through `--engine legacy` on the unified `find` and `ask` commands.

## GraphRAG Pivot Framing

New project framing:

```text
CLI-first GraphRAG research-memory system for ingesting technical documents,
building a graph-based retrieval index, answering local/global research questions,
and exporting inspectable wiki artifacts with provenance and citations.
```

Key sentence for updates and presentation:

```text
The wiki is not the retrieval engine. The wiki is the human-readable artifact layer.
Microsoft GraphRAG and WikiGraphRAG are the retrieval/synthesis engines.
```

## Implemented

- CLI foundation with the current shipped commands: `init`, `add`, `agent`, `update`, `find`, `ask`, `legacy`, `lint`, `review`, `status`, `export`, `doctor`, `config`, and `sources`. The old `graph` command group has been removed.
- Project layout, config loading, schema loading, and manifest tracking.
- Add pipeline that stores raw sources as source of truth and writes canonical normalized artifacts.
- Conversion routing:
  - direct markdown/text
  - Mistral OCR-first for `.pdf`, `.docx`, `.pptx`, `.png`, `.jpg`, `.jpeg`, `.avif`
  - `wkhtmltopdf` or `xhtml2pdf` + Mistral OCR for `.html`/`.htm`
  - Docling, then MarkItDown, as lower-confidence PDF fallbacks after Mistral OCR
  - MarkItDown for bounded born-digital subset with quality-gated fallbacks.
- Update pipeline that builds source pages, updates the wiki index, optionally refreshes legacy concept pages, refreshes the deprecated FTS5 comparator index with markdown-scan fallback warnings, writes lock-protected compile-run state for resumable runs, and runs GraphRAG sync/index/export unless `--no-graph` is set.
- Deterministic lint for links, fragments, headings, frontmatter, duplicates, empties, manifest artifact drift, stale pages, GraphRAG input/index/export staleness, and maintenance signals.
- Deprecated source-page-only search via SQLite FTS5 chunk index through `kb find --engine legacy`, now treated as temporary legacy infrastructure rather than the future default.
- Microsoft GraphRAG dependency and initialized workspace under `graph/graphrag/`, with tracked settings/prompts/input scaffolding and ignored local `.env`, generated input, output, cache, and logs.
- Main-command GraphRAG input/index sync through `kb update`, which writes `graph/graphrag/input/sources.json` from `raw/_manifest.json` and `raw/normalized/`, configures JSON input and metadata prepending in `settings.yaml`, warns and skips isolated missing normalized artifacts during normal updates, and auto-selects full `fast`, incremental `fast-update`, retry, or skip based on output completeness plus source/runtime digests.
- Main-command GraphRAG workspace/status/export through `kb init`, `kb status`, `kb doctor`, `kb lint`, `kb update`, and `kb export`.
- GraphRAG query wrapper through `kb ask --engine graphrag --method auto|basic|local|global|drift`, including `--save` for GraphRAG-backed analysis pages with retriever/method/index-hash metadata.
- GraphRAG wiki export through `kb update` and `kb export`, which convert GraphRAG output tables into `wiki/graph/index.md`, documents, text units, entities, relationships, and communities while preserving legacy `wiki/concepts/` pages.
- Explicit GraphRAG ask controller through `kb ask --engine graphrag --method auto|basic|local|global|drift`, with readiness checks, deterministic routing, explicit provider credential validation, and graph-backed saved analysis metadata.
- GraphRAG hardening for active-output selection, vector-store state checks, method-specific query readiness, runtime/settings/prompt digest freshness, unique saved-analysis filenames, blank-answer refusal, and raw stdout/stderr preservation.
- Graph-aware top-level `kb find`, which searches direct graph entities/relationships plus maintained wiki pages with bounded streamed Parquet scans, graph artifact diagnostics, stable deduplication, and global ranking of merged candidates.
- Phase 8 benchmark and runners: `eval/benchmark.yaml`, `scripts/evaluate_graph_modes.py`, `scripts/evaluate_retrieval.py`, and `scripts/evaluate_answers.py` write `eval/results/summary.md`, `retrieval_metrics.csv`, and `answer_metrics.csv`.
- Provider-backed `kb ask --engine legacy` and `kb review` with explicit failure semantics.
- Structured provider review output: `kb review` requires JSON/schema-backed findings and rejects malformed legacy pipe-style output instead of treating it as a compatibility fallback.
- Answer persistence with `kb ask --engine legacy --save` and `--save-as`; saved analysis pages are indexed immediately.
- Source-grounded ask behavior: `kb ask --engine legacy` uses source-page chunks as primary evidence, excludes generated concept pages and saved analysis pages from evidence, validates structured claims/citations, and strips raw citation-ref markers from answer prose before display or save.
- Obsidian vault export via `kb export`.
- Library-backed simplification pass: shared Markdown/frontmatter parsing, Pydantic config validation, NLTK collocation-based concept topic extraction, RapidFuzz terminology variants, and Unicode-aware slugging.
- `kb agent` control plane using the optional `graphwiki-kb[agent]` extra, OpenAI Agents SDK runtime boundary, service-backed tools for ask/find/status/lint/review/research/list-recommendations, sessionless one-shot runs, optional SQLite sessions, durable run traces, and ignored local state under `graph/runs/agent/`.
- Config schema version 8 with active `agent` and `research` sections. Research calls the OpenAI Responses `web_search` tool directly, persists numbered source recommendations, and keeps ingestion separate until an approved `ingest_recommendation` tool call. `update_kb` drives the same `UpdateService` path as `kb update` and requires approval or `--yes`.
- Lock-protected state writes for config migration, manifest updates, compile-run state, graph index-run state, wiki log appends, and GraphRAG workspace operations.
- Real-data validation of the agent path on a fresh two-PDF project using local PDF conversion plus OpenAI-backed fast GraphRAG indexing; final status was current, lint and review were clean, and one-shot session bleed was fixed.
- Latest main-line cleanup resolved pre-existing lint, type-check, and test failures and tightened CI/workflow metadata.
- CI and dependency baseline with the current enforced coverage floor from `pyproject.toml`.

## In Progress Now

- Turning the final verification evidence into the presentation narrative: Effective Recall@8 ties at 0.827, WikiGraphRAG is faster with strict citation refs, and Microsoft GraphRAG leads on answer quality plus refusal calibration.
- Keeping provider-backed reruns explicit because they consume model budget and the tracked `eval/results/verification_report.md` is now the branch's source-of-truth comparison.
- Keeping the agent demo bounded around service-backed KB operations, research recommendations, and explicit write approvals.

## Next Regular Work

- Use Phase 8 outputs and real-data runs to support the Update 3 and final comparison narrative.
- Freeze or clearly define the final benchmark subset so retrieval, answer-quality, speed, and cost numbers remain comparable.
- Report the GraphRAG tradeoff honestly: stronger synthesis and inspection, but slower indexing and provider budget.

## Planned Later

- `kb fix --propose` with approval-gated diff workflow.
- Additional structured lint checks (for stale summaries and duplicate concepts).
- Optional provider-backed cleanup/reconstruction path for OCR edge cases.
- Concept synthesis improvements and quality gates.
- Consider whether export or future fix tools belong behind the same agent approval pattern; destructive actions remain out of scope.
- Optional visualization/reporting and non-obligatory UI layer after core outcomes are complete.
- Post-capstone productization: multi-subject workspace mode, installable or executable distribution with a default local workspace folder, and a GUI or app-style wrapper for easier source import, subject switching, search, ask, status, and export.

## Overall Assessment

The current implementation is a strong baseline for the final capstone push:
ingest/update/status/default ask/graph-aware find/agent control plane/graph wiki export/legacy search/legacy ask/review/lint/doctor/export/evaluation are operational and covered by tests. The remaining effort is now producing defensible comparison results and final reporting without losing provenance, maintainability, or the ability to compare against the original FTS path as deprecated legacy behavior.
