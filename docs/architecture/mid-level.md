# Mid-Level Architecture

## Package Map

| Package | Responsibility |
| --- | --- |
| `src/cli.py` | CLI entrypoint and application bootstrap |
| `src/commands/` | Thin user-facing command wrappers |
| `src/services/` | Deterministic normalization, ingest, compile, diff, lint, review, search, query, export, status, config, and manifest services |
| `src/models/` | Shared command, source, tool, and wiki dataclasses |
| `src/engine/` | Command and tool registry boundaries |
| `src/providers/` | Provider abstraction layer with OpenAI, Anthropic, and Gemini implementations |
| `src/schemas/` | Pydantic models for claims, evidence bundles, review findings, and run artifacts |
| `src/prompts/` (planned) | Versioned prompt assets per agent role (answerer, judge, extractor, skeptic, arbiter) |
| `src/storage/` | SQLite run-artifact persistence; future FTS5 retrieval |

## Command To Service Mapping

| Command Wrapper | Main Service |
| --- | --- |
| `src/commands/init.py` | `src/services/project_service.py` and `src/services/config_service.py` |
| `src/commands/ingest.py` | `src/services/ingest_service.py`, `src/services/normalization_service.py`, and `src/services/manifest_service.py` |
| `src/commands/compile.py` | `src/services/compile_service.py` |
| `src/commands/diff.py` | `src/services/diff_service.py` |
| `src/commands/search.py` | `src/services/search_service.py` |
| `src/commands/query.py` | `src/services/query_service.py` |
| `src/commands/review.py` | `src/services/review_service.py` |
| `src/commands/lint.py` | `src/services/lint_service.py` |
| `src/commands/status.py` | `src/services/status_service.py` |
| `src/commands/export_vault.py` | `src/services/export_service.py` |

## Data Flow

| Stage | Input | Output |
| --- | --- | --- |
| Ingest | canonical markdown/plain-text files, Docling-routed PDFs, and a bounded MarkItDown-backed born-digital subset | raw source copy, normalized artifact, and manifest metadata |
| Compile | normalized canonical text plus manifest metadata | source pages, concept pages, wiki index, compile log |
| Diff | manifest metadata plus compile state | pre-compile source status preview |
| Search | compiled wiki artifacts | ranked matches |
| Query | user question plus compiled context | cited answer via heuristic assembly, single provider synthesis, or self-consistency over a frozen evidence bundle; optionally saved as an analysis page |
| Lint | compiled wiki and metadata | structural findings for links, fragments, headings, titles, typed frontmatter, empty pages, and maintenance signals |
| Review | compiled wiki pages | semantic findings via heuristics, single-pass provider review, or adversarial extractor/skeptic/arbiter findings; optionally persisted as a review run artifact |
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
- `kb lint` checks links, fragments, headings, titles, and metadata deterministically; `kb review` checks content-level coherence using heuristics, an optional single-pass provider review, or the explicit adversarial pipeline.
- Raw sources remain the source of truth; compiled pages are derived artifacts.
- Compile should prefer the normalized canonical artifact when one exists rather than reparsing the original raw source.
- Optional LLM-based cleanup or reconstruction should remain an explicit provider-mediated step instead of a silent default ingest behavior.
- Query behavior should prefer the compiled wiki over direct raw-file prompting.
- Evaluation features should remain clearly separated from the core CLI workflow.
