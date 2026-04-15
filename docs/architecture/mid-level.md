# Mid-Level Architecture

## Package Map

| Package | Responsibility |
| --- | --- |
| `src/cli.py` | CLI entrypoint and application bootstrap |
| `src/commands/` | Thin user-facing command wrappers |
| `src/services/` | Deterministic normalization, ingest, compile, concept, diff, lint, review, search, query, export, status, config, and manifest services |
| `src/models/` | Shared command, source, tool, and wiki dataclasses |
| `src/engine/` | Command and tool registry boundaries |
| `src/providers/` | Provider abstraction layer with OpenAI, Anthropic, and Gemini implementations |
| `src/schemas/` | Pydantic models for claims, evidence bundles, review findings, and run artifacts |
| `src/prompts/` (planned) | Versioned prompt assets per agent role (answerer, judge, extractor, skeptic, arbiter) |
| `src/storage/` | SQLite run-artifact persistence; future FTS5 retrieval |

## Command To Service Mapping

Commands are organized into flat verbs and namespaced groups:

| Group | Click Name | Command Wrapper | Main Service |
| --- | --- | --- | --- |
| *(flat)* | `init` | `src/commands/init.py` | `src/services/project_service.py` and `src/services/config_service.py` |
| *(flat)* | `ingest` | `src/commands/ingest.py` | `src/services/ingest_service.py`, `src/services/normalization_service.py`, and `src/services/manifest_service.py` |
| *(flat)* | `compile` | `src/commands/compile.py` | `src/services/compile_service.py` (+ `src/services/concept_service.py` with `--with-concepts`) |
| `query` | `search` | `src/commands/search.py` | `src/services/search_service.py` |
| `query` | `ask` | `src/commands/query.py` | `src/services/query_service.py` |
| `check` | `lint` | `src/commands/lint.py` | `src/services/lint_service.py` |
| `check` | `review` | `src/commands/review.py` | `src/services/review_service.py` |
| `show` | `status` | `src/commands/status.py` | `src/services/status_service.py` |
| `show` | `diff` | `src/commands/diff.py` | `src/services/diff_service.py` |
| `export` | `vault` | `src/commands/export_vault.py` | `src/services/export_service.py` |

## Data Flow

| Stage | Input | Output |
| --- | --- | --- |
| Ingest | canonical markdown/plain-text files, Docling-routed PDFs, and a bounded MarkItDown-backed born-digital subset | raw source copy, normalized artifact, and manifest metadata |
| Compile | normalized canonical text plus manifest metadata | source pages with provider-generated summaries, wiki index, and compile log; optionally concept pages and source-page backlinks via `--with-concepts` |
| Diff | manifest metadata plus compile state | pre-compile source status preview |
| Search | compiled wiki artifacts | ranked matches |
| Query | user question plus compiled context | cited provider answer or self-consistency over a frozen evidence bundle; optionally saved as an analysis page |
| Lint | compiled wiki and metadata | structural findings for links, fragments, headings, titles, typed frontmatter, empty pages, and maintenance signals |
| Review | compiled wiki pages | semantic findings from deterministic overlap checks plus single-pass provider review or adversarial extractor/skeptic/arbiter findings; optionally persisted as a review run artifact |
| Export | compiled wiki | Obsidian-friendly vault view |

## Current Ingest Scope

- The current implementation ingests `.md`, `.markdown`, and `.txt` files directly, routes `.pdf` files through Docling, and uses MarkItDown for a bounded born-digital subset such as HTML, CSV, Office documents, notebooks, and EPUB.
- OCR-backed ingest is still deferred and should arrive as a provider-backed fallback, with Mistral OCR as the current preferred OCR path for scanned or image-heavy inputs.

## Planned Deliberation Pipelines

| Pipeline | CLI Flag | Stages | Key Principle |
| --- | --- | --- | --- |
| Self-consistency query | `--self-consistency N` | retrieve → freeze evidence → sample N answers in parallel → normalize claims → deterministic merge → final cited answer + SQLite run artifact | All candidates see the same frozen evidence bundle; merge claims, not paragraphs |
| Adversarial review | `--adversarial` | generate candidate pairs → extract claims → skeptic critique per pair → arbiter verdicts → SQLite run artifact | Disagreement is the product; `needs_review` is a success case |
| Fix proposal | `--propose` | proposer drafts patch → auditor checks citations → gate emits diff | Never auto-applies; mandatory user confirmation |

## Structural Rules

- Commands should stay thin and delegate quickly.
- Services should remain deterministic unless the feature explicitly requires model-backed synthesis.
- `kb check lint` checks links, fragments, headings, titles, and metadata deterministically; `kb check review` prepends deterministic overlap checks to a required provider-backed single-pass or adversarial pipeline.
- Raw sources remain the source of truth; compiled pages are derived artifacts.
- Compile should prefer the normalized canonical artifact when one exists rather than reparsing the original raw source.
- Optional LLM-based cleanup or reconstruction should remain an explicit provider-mediated step instead of a silent default ingest behavior.
- Query behavior should prefer the compiled wiki over direct raw-file prompting.
- Evaluation features should remain clearly separated from the core CLI workflow.
