# Capstone Project Status Overview

Date: 2026-04-24
Updated: 2026-05-11 for GraphRAG pivot Phase 6

## Current Position

The project has moved from scaffold+implementation into a controlled GraphRAG pivot.

1. Proposal, architecture, and the original command surface are complete.
2. Simplification and deterministic baseline are complete.
3. The SQLite FTS5/wiki retrieval path has been demoted to explicit deprecated legacy behavior.
4. Microsoft GraphRAG is installed, and the dedicated `graph/graphrag/` workspace is initialized.
5. `kb graph sync` writes GraphRAG JSON input from normalized artifacts and manifest metadata.
6. `kb graph init`, `kb graph index`, and `kb graph status` expose the GraphRAG workspace lifecycle through the project CLI.
7. `kb graph ask --method local|global|drift|basic` exposes explicit GraphRAG query modes before the default `kb ask` controller is enabled.
8. `kb graph export-wiki` exports GraphRAG output tables into generated markdown pages under `wiki/graph/`.
9. GraphRAG is becoming the default retrieval and synthesis engine for comparison, synthesis, and corpus-level research questions.
10. Wiki artifacts remain the inspectable provenance, maintenance, and export layer.

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

- CLI foundation with the current shipped commands: `init`, `add`, `update`, `find`, `ask`, `graph`, `legacy`, `lint`, `review`, `status`, `export`, `doctor`, `config`, and `sources`.
- Project layout, config loading, schema loading, and manifest tracking.
- Add pipeline that stores raw sources as source of truth and writes canonical normalized artifacts.
- Conversion routing:
  - direct markdown/text
  - Mistral OCR-first for `.pdf`, `.docx`, `.pptx`, `.png`, `.jpg`, `.jpeg`, `.avif`
  - `wkhtmltopdf` + Mistral OCR for `.html`/`.htm`
  - MarkItDown for bounded born-digital subset with quality-gated fallbacks.
- Update pipeline that builds source pages, updates the wiki index, generates concepts, refreshes FTS5 chunk index, and writes compile-run state for resumable runs.
- Deterministic lint for links, fragments, headings, frontmatter, duplicates, empties, stale pages, and maintenance signals.
- Deprecated search over compiled wiki artifacts via SQLite FTS5 chunk index through `kb legacy find`, now treated as temporary legacy infrastructure rather than the future default.
- Microsoft GraphRAG dependency and initialized workspace under `graph/graphrag/`, with tracked settings/prompts/input scaffolding and ignored local `.env`, generated input, output, cache, and logs.
- GraphRAG input sync through `kb graph sync`, which writes `graph/graphrag/input/sources.json` from `raw/_manifest.json` and `raw/normalized/` while configuring JSON input and metadata prepending in `settings.yaml`.
- GraphRAG workspace/index wrappers through `kb graph init`, `kb graph index --method fast --dry-run`, and `kb graph status`.
- GraphRAG query wrapper through `kb graph ask --method local|global|drift|basic`, including `--save` for GraphRAG-backed analysis pages with retriever/method/index-hash metadata.
- GraphRAG wiki export through `kb graph export-wiki`, which converts GraphRAG output tables into `wiki/graph/index.md`, documents, text units, entities, relationships, and communities while preserving legacy `wiki/concepts/` pages.
- Provider-backed `kb legacy ask` and `kb review` with explicit failure semantics.
- Structured provider review output: `kb review` requires JSON/schema-backed findings and rejects malformed legacy pipe-style output instead of treating it as a compatibility fallback.
- Answer persistence with `kb legacy ask --save` and `--save-as`; saved analysis pages are indexed immediately.
- Source-grounded ask behavior: `kb legacy ask` uses source-page chunks as primary evidence, excludes generated concept pages and saved analysis pages from evidence, validates structured claims/citations, and strips raw citation-ref markers from answer prose before display or save.
- Obsidian vault export via `kb export`.
- Library-backed simplification pass: shared Markdown/frontmatter parsing, Pydantic config validation, NLTK collocation-based concept topic extraction, RapidFuzz terminology variants, and Unicode-aware slugging.
- CI and dependency baseline with enforced 97%+ coverage floor from `pyproject.toml`.

## In Progress Now

- Captured the legacy FTS `find` output for the REALM-vs-RAG benchmark question; provider-backed legacy ask captures remain pending explicit approval because they send retrieved local snippets to the configured external model.
- Preparing default `kb ask` routing and broader freshness/status checks over graph inputs, outputs, and generated graph wiki pages.
- Updating evaluation to compare deprecated FTS against GraphRAG Basic, Local, Global, and DRIFT search without making FTS part of the normal UX.

## Next Regular Work

- Wire the default `kb ask` controller to choose graph query modes.
- Add graph freshness/status checks around synced input, GraphRAG output, and generated `wiki/graph/` pages.

## Planned Later

- `kb fix --propose` with approval-gated diff workflow.
- Additional structured lint checks (for stale summaries and duplicate concepts).
- Optional provider-backed cleanup/reconstruction path for OCR edge cases.
- Concept synthesis improvements and quality gates.
- Optional visualization/reporting and non-obligatory UI layer after core outcomes are complete.
- Post-capstone productization: multi-subject workspace mode, installable or executable distribution with a default local workspace folder, and a GUI or app-style wrapper for easier source import, subject switching, search, ask, status, and export.

## Overall Assessment

The current implementation is a strong baseline for the pivot:
ingest/update/graph init/graph sync/graph index/status/graph ask/graph wiki export/legacy search/legacy ask/review/lint/export are operational and covered by tests, and the remaining effort is now routing default ask behavior and expanding graph freshness checks without losing provenance, maintainability, or the ability to compare against the original FTS path as deprecated legacy behavior.
