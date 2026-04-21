# Mid-Level Architecture

## Package Map

| Package | Responsibility |
| --- | --- |
| `src/cli.py` | CLI entrypoint and application bootstrap |
| `src/commands/` | Thin user-facing command wrappers |
| `src/services/` | Deterministic normalization, ingest, compile, concept, diff, lint, review, search, query, export, status, config, and manifest services |
| `src/models/` | Shared command, source, tool, and wiki dataclasses |
| `src/engine/` | Command and tool registry boundaries |
| `src/providers/` | Provider abstraction layer with OpenAI, Anthropic, and Gemini implementations; shared Tenacity retry decorator for transient failures |
| `src/schemas/` | Pydantic models for claims, evidence bundles, review findings, and run artifacts |
| `src/prompts/` (planned) | Versioned prompt assets per agent role (answerer, judge, extractor, skeptic, arbiter) |
| `src/storage/` | SQLite run-artifact persistence plus SQLite FTS5 chunk-index storage |

## Command To Service Mapping

All commands are flat top-level verbs:

| Click Name | Command Wrapper | Main Service |
| --- | --- | --- |
| `init` | `src/commands/init.py` | `src/services/project_service.py` and `src/services/config_service.py` |
| `add` | `src/commands/add.py` | `src/services/ingest_service.py`, `src/services/normalization_service.py`, and `src/services/manifest_service.py` |
| `update` | `src/commands/update.py` | `src/services/compile_service.py`, `src/services/concept_service.py`, `src/services/search_service.py` |
| `compile` | `src/commands/compile.py` | `src/services/compile_service.py`, `src/services/concept_service.py`, `src/services/search_service.py` |
| `find` | `src/commands/find.py` | `src/services/search_service.py` |
| `ask` | `src/commands/ask.py` | `src/services/query_service.py` |
| `lint` | `src/commands/lint.py` | `src/services/lint_service.py` |
| `review` | `src/commands/review.py` | `src/services/review_service.py` |
| `status` | `src/commands/status.py` | `src/services/status_service.py`, `src/services/diff_service.py` (with `--changed`) |
| `export` | `src/commands/export_cmd.py` | `src/services/export_service.py` |
| `doctor` | `src/commands/doctor.py` | `src/services/doctor_service.py` |
| `history` | `src/commands/history.py` | `src/storage/run_store.py` |
| `config` | `src/commands/config_cmd.py` | `src/services/config_service.py`, `src/services/model_registry_service.py` |
| `sources` | `src/commands/sources.py` | `src/services/manifest_service.py` |

## Data Flow

| Stage | Input | Output |
| --- | --- | --- |
| Ingest | canonical markdown/plain-text files, Docling-routed PDFs, and a bounded MarkItDown-backed born-digital subset | raw source copy, normalized artifact, and manifest metadata |
| Compile | normalized canonical text plus manifest metadata | source pages with provider-generated summaries, wiki index, and compile log; optionally concept pages and source-page backlinks via `--with-concepts` |
| Diff | manifest metadata plus compile state | pre-compile source status preview |
| Search | compiled wiki artifacts | ranked page matches derived from indexed chunks |
| Query | user question plus compiled context | cited provider answer or self-consistency over a frozen evidence bundle; optionally saved as an analysis page |
| Lint | compiled wiki and metadata | structural findings for links, fragments, headings, titles, typed frontmatter, empty pages, and maintenance signals |
| Review | compiled wiki pages | semantic findings from deterministic overlap checks plus single-pass provider review or adversarial extractor/skeptic/arbiter findings; optionally persisted as a review run artifact |
| Export | compiled wiki | Obsidian-friendly vault view |

## Current Ingest Scope

- The current implementation adds `.md`, `.markdown`, and `.txt` files directly, routes `.pdf` files through Docling, and uses MarkItDown for a bounded born-digital subset such as HTML, CSV, Office documents, notebooks, and EPUB.
- `kb add` is the primary ingestion command; `src/commands/ingest.py` provides the shared implementation.
- Directory inputs for `kb add` walk recursively by default, add only supported source files, and leave unsupported files untouched.
- OCR-backed ingestion is still deferred and should arrive as a provider-backed fallback, with Mistral OCR as the current preferred OCR path for scanned or image-heavy inputs.

## Planned Deliberation Pipelines

| Pipeline | CLI Flag | Stages | Key Principle |
| --- | --- | --- | --- |
| Self-consistency query | `--self-consistency N` | retrieve → freeze evidence → sample N answers in parallel → normalize claims → deterministic merge → final cited answer + SQLite run artifact | All candidates see the same frozen evidence bundle; merge claims, not paragraphs |
| Adversarial review | `--adversarial` | generate candidate pairs → extract claims → skeptic critique per pair → arbiter verdicts → SQLite run artifact | Disagreement is the product; `needs_review` is a success case |
| Fix proposal | `--propose` | proposer drafts patch → auditor checks citations → gate emits diff | Never auto-applies; mandatory user confirmation |

## Structural Rules

- Commands should stay thin and delegate quickly.
- The command layer owns terminal-only concerns such as section headings, list formatting, and progress display via Rich (`Console`, `Table`, `Progress`); long-running services expose callback-friendly hooks instead of writing directly to the terminal. User-supplied content is markup-escaped via `rich.markup.escape`.
- Services should remain deterministic unless the feature explicitly requires model-backed synthesis.
- `kb lint` checks links, fragments, headings, titles, and metadata deterministically; `kb review` prepends deterministic overlap checks to a required provider-backed single-pass or adversarial pipeline.
- `build_services()` resolves per-task providers through `ModelRegistryService`: update gets `fast`, ask gets `balanced`, review gets `balanced` by default. Global `--tier` and `--model` flags override. `--quality` on `kb ask` also implies a matching tier.
- Raw sources remain the source of truth; compiled pages are derived artifacts.
- Compile should prefer the normalized canonical artifact when one exists rather than reparsing the original raw source.
- Optional LLM-based cleanup or reconstruction should remain an explicit provider-mediated step instead of a silent default ingest behavior.
- Query behavior should prefer the compiled wiki over direct raw-file prompting.
- Evaluation features should remain clearly separated from the core CLI workflow.
