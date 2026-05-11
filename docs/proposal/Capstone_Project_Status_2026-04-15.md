# Capstone Project Status Overview

Date: 2026-04-24
Updated: 2026-05-11 for GraphRAG pivot Phase 0

## Current Position

The project has moved from scaffold+implementation into a controlled GraphRAG pivot.

1. Proposal, architecture, and the original command surface are complete.
2. Simplification and deterministic baseline are complete.
3. The SQLite FTS5/wiki retrieval path is being demoted to explicit deprecated legacy behavior.
4. GraphRAG is becoming the default retrieval and synthesis engine for comparison, synthesis, and corpus-level research questions.
5. Wiki artifacts remain the inspectable provenance, maintenance, and export layer.

In short, the core workflow works end-to-end, and the pivot keeps that work instead of discarding it. Current work is reframing the project as GraphRAG-first while preserving the original wiki system as the artifact layer and demoting old FTS retrieval to explicit legacy commands if it remains at all.

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

- CLI foundation with 12 shipped commands: `init`, `add`, `update`, `find`, `ask`, `lint`, `review`, `status`, `export`, `doctor`, `config`, and `sources`.
- Project layout, config loading, schema loading, and manifest tracking.
- Add pipeline that stores raw sources as source of truth and writes canonical normalized artifacts.
- Conversion routing:
  - direct markdown/text
  - Mistral OCR-first for `.pdf`, `.docx`, `.pptx`, `.png`, `.jpg`, `.jpeg`, `.avif`
  - `wkhtmltopdf` + Mistral OCR for `.html`/`.htm`
  - MarkItDown for bounded born-digital subset with quality-gated fallbacks.
- Update pipeline that builds source pages, updates the wiki index, generates concepts, refreshes FTS5 chunk index, and writes compile-run state for resumable runs.
- Deterministic lint for links, fragments, headings, frontmatter, duplicates, empties, stale pages, and maintenance signals.
- Search over compiled wiki artifacts via SQLite FTS5 chunk index, now treated as temporary legacy infrastructure rather than the future default.
- Provider-backed `kb ask` and `kb review` with explicit failure semantics.
- Structured provider review output: `kb review` requires JSON/schema-backed findings and rejects malformed legacy pipe-style output instead of treating it as a compatibility fallback.
- Answer persistence with `kb ask --save` and `--save-as`; saved analysis pages are indexed immediately.
- Source-grounded ask behavior: `kb ask` uses source-page chunks as primary evidence, excludes generated concept pages and saved analysis pages from evidence, validates structured claims/citations, and strips raw citation-ref markers from answer prose before display or save.
- Obsidian vault export via `kb export`.
- Library-backed simplification pass: shared Markdown/frontmatter parsing, Pydantic config validation, NLTK collocation-based concept topic extraction, RapidFuzz terminology variants, and Unicode-aware slugging.
- CI and dependency baseline with enforced 97%+ coverage floor from `pyproject.toml`.

## In Progress Now

- Completing the Phase 0 GraphRAG pivot documentation and shared language.
- Moving the current FTS/wiki retrieval path behind explicit deprecated legacy commands if it remains.
- Preparing the GraphRAG workspace, input sync, indexing, query modes, and graph-derived wiki artifact plan.
- Updating evaluation to compare deprecated FTS against GraphRAG Basic, Local, Global, and DRIFT search without making FTS part of the normal UX.

## Next Regular Work

- Capture deprecated FTS outputs for the questions that exposed retrieval gaps.
- Add the GraphRAG dependency and workspace structure.
- Sync normalized corpus artifacts into GraphRAG JSON input while preserving manifest metadata.
- Wrap GraphRAG init/index/status/query behavior in the CLI incrementally.

## Planned Later

- `kb fix --propose` with approval-gated diff workflow.
- Additional structured lint checks (for stale summaries and duplicate concepts).
- Optional provider-backed cleanup/reconstruction path for OCR edge cases.
- Concept synthesis improvements and quality gates.
- Optional visualization/reporting and non-obligatory UI layer after core outcomes are complete.
- Post-capstone productization: multi-subject workspace mode, installable or executable distribution with a default local workspace folder, and a GUI or app-style wrapper for easier source import, subject switching, search, ask, status, and export.

## Overall Assessment

The current implementation is a strong baseline for the pivot:
ingest/update/search/ask/review/lint/export are operational and covered by tests, and the remaining effort is now introducing GraphRAG without losing provenance, maintainability, or the ability to compare against the original FTS path as deprecated legacy behavior.
