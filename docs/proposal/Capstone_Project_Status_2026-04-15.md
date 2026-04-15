# Capstone Project Status Overview

Date: 2026-04-15

## Current Position

The project is past the scaffolding stage and past the initial provider-integration stage. The current milestone position is:

1. Proposal and design: essentially complete.
2. CLI and provider foundation: complete.
3. Corpus curation and real-corpus evaluation: current active phase.
4. Concept synthesis and maintenance validation: next major build phase.
5. Final comparative evaluation and polish: later.

In short, the core product exists and works. The remaining capstone work is mainly evaluation, synthesis, and evidence gathering rather than basic system construction.

## Implemented

- CLI foundation with 10 shipped commands: `init`, `ingest`, `compile`, `search`, `query`, `lint`, `review`, `status`, `diff`, and `export-vault`.
- Project layout, config loading, schema loading, and manifest tracking.
- Ingest pipeline that keeps raw sources as the source of truth and writes canonical normalized artifacts.
- Normalization routes for direct markdown/text, Docling-backed PDF ingest, and a bounded MarkItDown-backed born-digital subset.
- Compile pipeline that generates source pages, wiki index files, and compile logs.
- Deterministic lint for links, fragments, heading structure, duplicate titles, typed frontmatter, empty pages, stale pages, and related maintenance issues.
- Search over compiled wiki artifacts.
- Provider-backed compile, query, and review using OpenAI, Anthropic, and Gemini.
- Saved analysis pages under `wiki/concepts/` after query runs.
- Bounded deliberation layer v1.
- `kb query --self-consistency N` with frozen evidence, parallel provider sampling, deterministic claim merge, and SQLite run persistence.
- `kb review --adversarial` with candidate-pair generation, extractor/skeptic/arbiter prompts, typed findings, and SQLite run persistence.
- Shared Pydantic schemas and SQLite `RunStore` for deliberation artifacts.
- Obsidian-friendly vault export.
- Strong automated validation: 352 passing tests and 98.03% coverage as of 2026-04-15.

## In Progress Now

- Curating and exercising the real research corpus rather than only synthetic or isolated test fixtures.
- Evaluating the bounded deliberation features on realistic documents.
- Capturing quality, latency, token cost, unsupported-claim behavior, and contradiction or terminology-drift behavior from persisted run artifacts.
- Tightening the end-to-end workflow based on real-corpus findings.

This means the current question is no longer "can the system do the workflow at all?" The current question is "how well does it perform on the actual capstone corpus, and where does it still need targeted improvement?"

## Next Regular Work

- Run a fuller real-corpus evaluation pass for `kb query --self-consistency N` and `kb review --adversarial`.
- Add lightweight reporting helpers if manual SQLite run inspection becomes too slow.
- Implement concept-page generation from multiple related source pages.
- Implement backlink maintenance so concept and source pages remain navigable.
- Improve search quality if evaluation shows retrieval is the main bottleneck.
- Add SQLite FTS5-backed retrieval if needed.
- Add provider-backed OCR fallback only if the current deterministic converter set proves insufficient for key sources.

## Planned Later

- `kb fix --propose` with proposer, auditor, and gated diff output.
- Richer lint and maintenance checks such as stale-summary detection and duplicate-concept detection.
- Proposer or critic concept synthesis workflows after initial concept-page generation exists.
- More formal evaluation and reporting helpers for benchmark comparisons.

## Deferred Or Stretch Scope

- Approval-gated external source expansion.
- Graph export and graph-based comparison workflows.
- Optional LLM-backed compile beyond the current provider-generated summary path.
- A Textual terminal UI.
- Multi-model role heterogeneity for proposer, critic, answerer, or judge roles.

## Overall Assessment

The project is in a strong position for the capstone timeline. The foundation that usually consumes the most implementation risk is already in place: ingestion, normalization, compilation, search, query, review, persistence, export, and automated tests are all working. The main remaining work is to prove quality on the real corpus, add concept synthesis and backlink maintenance, and produce a strong comparative evaluation story for the final capstone deliverables.
