# Capstone Project Status Overview

Date: 2026-04-24
Updated: 2026-05-12 for Phase 9 main-command GraphRAG UX

## Current Position

The project has moved from scaffold+implementation into a controlled GraphRAG pivot.

1. Proposal, architecture, and the original command surface are complete.
2. Simplification and deterministic baseline are complete.
3. The SQLite FTS5/wiki retrieval path has been demoted to explicit deprecated legacy behavior.
4. Microsoft GraphRAG is installed, and the dedicated `graph/graphrag/` workspace is initialized.
5. `kb init` creates and syncs the GraphRAG workspace.
6. `kb update` writes GraphRAG JSON input from normalized artifacts and manifest metadata, then auto-selects full index, incremental update, or skip.
7. `kb ask --method auto|basic|local|global|drift` exposes GraphRAG query modes through the default answer controller.
8. `kb update` and `kb export` export GraphRAG output tables into generated markdown pages under `wiki/graph/`.
9. Top-level `kb ask` is now the GraphRAG-aware default controller with deterministic auto-routing across Basic, Local, Global, and DRIFT modes.
10. Phase 8 evaluation scripts compare deprecated FTS against GraphRAG modes with local-safe defaults and opt-in provider-backed answer runs.
11. `kb status`, `kb doctor`, and `kb lint` now include GraphRAG health and freshness checks.
12. Wiki artifacts remain the inspectable provenance, maintenance, and export layer.

In short, the core workflow works end-to-end, and the pivot keeps that work instead of discarding it. Current work is making the project GraphRAG-first while preserving the original wiki system as the artifact layer and exposing old FTS retrieval only through explicit legacy commands.

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
GraphRAG is the retrieval and synthesis engine.
```

## Implemented

- CLI foundation with the current shipped commands: `init`, `add`, `update`, `find`, `ask`, `legacy`, `lint`, `review`, `status`, `export`, `doctor`, `config`, and `sources`. The old `graph` command group has been removed.
- Project layout, config loading, schema loading, and manifest tracking.
- Add pipeline that stores raw sources as source of truth and writes canonical normalized artifacts.
- Conversion routing:
  - direct markdown/text
  - Mistral OCR-first for `.pdf`, `.docx`, `.pptx`, `.png`, `.jpg`, `.jpeg`, `.avif`
  - `wkhtmltopdf` + Mistral OCR for `.html`/`.htm`
  - MarkItDown for bounded born-digital subset with quality-gated fallbacks.
- Update pipeline that builds source pages, updates the wiki index, generates concepts, refreshes FTS5 chunk index, writes compile-run state for resumable runs, and runs GraphRAG sync/index/export unless `--no-graph` is set.
- Deterministic lint for links, fragments, headings, frontmatter, duplicates, empties, stale pages, GraphRAG input/index/export staleness, and maintenance signals.
- Deprecated search over compiled wiki artifacts via SQLite FTS5 chunk index through `kb legacy find`, now treated as temporary legacy infrastructure rather than the future default.
- Microsoft GraphRAG dependency and initialized workspace under `graph/graphrag/`, with tracked settings/prompts/input scaffolding and ignored local `.env`, generated input, output, cache, and logs.
- Main-command GraphRAG input/index sync through `kb update`, which writes `graph/graphrag/input/sources.json` from `raw/_manifest.json` and `raw/normalized/`, configures JSON input and metadata prepending in `settings.yaml`, and auto-selects full `fast`, incremental `fast-update`, or skip based on output completeness plus source/runtime digests.
- Main-command GraphRAG workspace/status/export through `kb init`, `kb status`, `kb doctor`, `kb lint`, `kb update`, and `kb export`.
- GraphRAG query wrapper through `kb ask --method auto|basic|local|global|drift`, including `--save` for GraphRAG-backed analysis pages with retriever/method/index-hash metadata.
- GraphRAG wiki export through `kb update` and `kb export`, which convert GraphRAG output tables into `wiki/graph/index.md`, documents, text units, entities, relationships, and communities while preserving legacy `wiki/concepts/` pages.
- Default GraphRAG ask controller through `kb ask --method auto|basic|local|global|drift`, with readiness checks, deterministic routing, explicit provider credential validation, and graph-backed saved analysis metadata.
- Phase 8 benchmark and runners: `eval/benchmark.yaml`, `scripts/evaluate_graph_modes.py`, `scripts/evaluate_retrieval.py`, and `scripts/evaluate_answers.py` write `eval/results/summary.md`, `retrieval_metrics.csv`, and `answer_metrics.csv`.
- Provider-backed `kb legacy ask` and `kb review` with explicit failure semantics.
- Structured provider review output: `kb review` requires JSON/schema-backed findings and rejects malformed legacy pipe-style output instead of treating it as a compatibility fallback.
- Answer persistence with `kb legacy ask --save` and `--save-as`; saved analysis pages are indexed immediately.
- Source-grounded ask behavior: `kb legacy ask` uses source-page chunks as primary evidence, excludes generated concept pages and saved analysis pages from evidence, validates structured claims/citations, and strips raw citation-ref markers from answer prose before display or save.
- Obsidian vault export via `kb export`.
- Library-backed simplification pass: shared Markdown/frontmatter parsing, Pydantic config validation, NLTK collocation-based concept topic extraction, RapidFuzz terminology variants, and Unicode-aware slugging.
- CI and dependency baseline with enforced 97%+ coverage floor from `pyproject.toml`.

## In Progress Now

- Running Phase 8 evaluation on the frozen benchmark and deciding when to allow provider-backed GraphRAG/legacy answer captures.
- Using Phase 9 health checks to validate real-corpus GraphRAG maintenance through the main command surface.

## Next Regular Work

- Use Phase 8 outputs to support the Update 3 and final comparison narrative.

## Planned Later

- `kb fix --propose` with approval-gated diff workflow.
- Additional structured lint checks (for stale summaries and duplicate concepts).
- Optional provider-backed cleanup/reconstruction path for OCR edge cases.
- Concept synthesis improvements and quality gates.
- Optional visualization/reporting and non-obligatory UI layer after core outcomes are complete.
- Post-capstone productization: multi-subject workspace mode, installable or executable distribution with a default local workspace folder, and a GUI or app-style wrapper for easier source import, subject switching, search, ask, status, and export.

## Overall Assessment

The current implementation is a strong baseline for the pivot:
ingest/update/status/default ask/graph wiki export/legacy search/legacy ask/review/lint/doctor/export/evaluation are operational and covered by tests, and the remaining effort is now producing real-corpus comparison results without losing provenance, maintainability, or the ability to compare against the original FTS path as deprecated legacy behavior.
