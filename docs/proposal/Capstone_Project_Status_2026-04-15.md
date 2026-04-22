# Capstone Project Status Overview

Date: 2026-04-22

## Current Position

The project has moved from scaffold+implementation into the final validation phase.

1. Proposal, architecture, and command surface are complete.
2. Simplification and deterministic baseline are complete.
3. Real-corpus evaluation, concept quality, and retrieval tuning are the current active focus.
4. Comparative evaluation and final polish are in progress.

In short, the core workflow works end-to-end. Current work is proving quality and producing capstone results on realistic corpus slices.

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
- Search over compiled wiki artifacts via SQLite FTS5 chunk index.
- Provider-backed `kb ask` and `kb review` with explicit failure semantics.
- Answer persistence with `kb ask --save` and `--save-as`; saved analysis pages are indexed immediately.
- Obsidian vault export via `kb export`.
- CI and dependency baseline with enforced 97%+ coverage floor from `pyproject.toml`.

## In Progress Now

- Running a fuller real-corpus evaluation pass on realistic source collections.
- Concept-page synthesis and backlink maintenance.
- Measuring retrieval quality and whether richer ranking is justified.
- Capturing quality, latency, and cost signals from `ask`/`review` runs for the final deliverable.
- Tightening workflow ergonomics for evaluator scripts and reporting.

## Next Regular Work

- Complete simplification cleanups and finish Phase 9 (export/concept simplification).
- Improve concept generation quality and reliability before final milestone reporting.
- Add richer comparison/benchmark reporting if manual SQLite run inspection becomes too slow.
- Add optional hybrid retrieval only if a clear evaluation gain is measured.

## Planned Later

- `kb fix --propose` with approval-gated diff workflow.
- Additional structured lint checks (for stale summaries and duplicate concepts).
- Optional provider-backed cleanup/reconstruction path for OCR edge cases.
- Concept synthesis improvements and quality gates.
- Optional visualization/reporting and non-obligatory UI layer after core outcomes are complete.

## Overall Assessment

The current implementation is in solid shape for final capstone execution:
ingest/update/search/ask/review/lint/export are operational and covered by tests, and the remaining effort is now primarily evidence quality and comparative evaluation rather than core infrastructure.
