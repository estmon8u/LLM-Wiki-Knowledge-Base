# Mid-Level Architecture

## Package Map

| Package | Responsibility |
| --- | --- |
| `src/cli.py` | CLI entrypoint and application bootstrap |
| `src/commands/` | Thin user-facing command wrappers |
| `src/services/` | Deterministic normalization, ingest, compile, concept, diff, lint, review, search, query (ask), export, status, config, and manifest services |
| `src/models/` | Shared command, source, and wiki dataclasses |
| `src/engine/` | Command registry boundary |
| `src/providers/` | Provider abstraction layer with OpenAI, Anthropic, and Gemini implementations; shared Tenacity retry decorator for transient failures and catalog-backed provider resolution |
| `src/storage/` | Compile-run state persistence and SQLite FTS5 chunk-index storage |

## Command To Service Mapping

All commands are flat top-level verbs:

| Click Name | Command Wrapper | Main Service |
| --- | --- | --- |
| `init` | `src/commands/init.py` | `src/services/project_service.py` and `src/services/config_service.py` |
| `add` | `src/commands/add.py` | `src/services/ingest_service.py`, `src/services/normalization_service.py`, and `src/services/manifest_service.py` |
| `update` | `src/commands/update.py` | `src/services/compile_service.py`, `src/services/concept_service.py`, `src/services/search_service.py` |
| `find` | `src/commands/find.py` | `src/services/search_service.py` |
| `ask` | `src/commands/ask.py` | `src/services/query_service.py` |
| `lint` | `src/commands/lint.py` | `src/services/lint_service.py` |
| `review` | `src/commands/review.py` | `src/services/review_service.py` |
| `status` | `src/commands/status.py` | `src/services/status_service.py`, `src/services/diff_service.py` (with `--changed`) |
| `export` | `src/commands/export_cmd.py` | `src/services/export_service.py` |
| `doctor` | `src/commands/doctor.py` | `src/services/doctor_service.py` |
| `config` | `src/commands/config_cmd.py` | `src/services/config_service.py` |
| `sources` | `src/commands/sources.py` | `src/services/manifest_service.py` |

## Data Flow

| Stage | Input | Output |
| --- | --- | --- |
| Ingest | canonical markdown/plain-text files, Mistral OCR-routed native documents and images, rendered HTML-to-PDF OCR, and a bounded MarkItDown subset | raw source copy, normalized artifact, and manifest metadata |
| Compile | normalized canonical text plus manifest metadata | source pages with provider-generated summaries, wiki index, and compile log; provider-clustered concept pages with deterministic fallback and source-page backlinks |
| Diff | manifest metadata plus compile state | pre-compile source status preview |
| Search | compiled wiki artifacts | ranked page matches derived from indexed chunks |
| Ask | user question plus compiled context | cited provider answer; optionally saved as an analysis page |
| Lint | compiled wiki and metadata | structural findings for links, fragments, headings, titles, typed frontmatter, empty pages, and maintenance signals |
| Review | compiled wiki pages | semantic findings from deterministic overlap checks plus schema-guided single-pass provider review |
| Export | compiled wiki | Obsidian-friendly vault view |

## Current Ingest Scope

- The current implementation adds `.md`, `.markdown`, and `.txt` files directly; routes `.pdf`, `.docx`, `.pptx`, `.png`, `.jpg`, `.jpeg`, and `.avif` through Mistral OCR first; renders `.html` / `.htm` to PDF with `wkhtmltopdf` before OCR; and uses MarkItDown for the remaining bounded born-digital subset such as CSV, notebooks, EPUB, and Excel files.
- `kb add` is the primary ingestion command; `src/commands/ingest.py` provides the shared implementation.
- Directory inputs for `kb add` walk recursively by default, add only supported source files, and leave unsupported files untouched.
- Conversion quality gates reject empty, implausibly short, or suspiciously truncated outputs before `raw/normalized/` artifacts are written. PDF, DOCX, PPTX, and HTML routes then fall back explicitly to Docling or MarkItDown based on config.

## Structural Rules

- Commands should stay thin and delegate quickly.
- The command layer owns terminal-only concerns such as section headings, list formatting, and progress display via Rich (`Console`, `Table`, `Progress`); long-running services expose callback-friendly hooks instead of writing directly to the terminal. User-supplied content is markup-escaped via `rich.markup.escape`.
- Services should remain deterministic unless the feature explicitly requires model-backed synthesis.
- Shared parsing belongs in `src/services/markdown_document.py`: services should consume parser-backed markdown/frontmatter helpers instead of adding new ad hoc regex stacks.
- Config validation belongs in Pydantic models inside `ConfigService`, with compatibility wrappers preserved for tests and callers.
- Concept clustering is semantic and provider-backed when possible; keep deterministic clustering only as fallback and keep page writing/backlink maintenance deterministic.
- `kb lint` checks links, fragments, headings, titles, and metadata deterministically; `kb review` prepends deterministic overlap checks to a required provider-backed single-pass review.
- `build_services()` reads `kb.config.yaml`, resolves the active provider from the embedded `providers` section, and creates a single shared provider via `build_provider(config)`.
- `kb.schema.md` is the wiki's operational constitution. Relevant schema sections are injected into compile and ask prompts via `schema_excerpt()`.
- Raw sources remain the source of truth; compiled pages are derived artifacts.
- Compile should prefer the normalized canonical artifact when one exists rather than reparsing the original raw source.
- Optional LLM-based cleanup or reconstruction should remain an explicit provider-mediated step instead of a silent default ingest behavior.
- Ask behavior should prefer the compiled wiki over direct raw-file prompting.

## Structured Provider Output Contracts

Provider-backed semantic steps now request structured responses at the service boundary instead of parsing freeform text. Concept generation returns concept clusters with title, summary, topic terms, and source pages. Review returns JSON findings with severity, code, pages, and message. `kb ask` returns answer markdown, claims, citations, and an insufficient-evidence flag. Compile summaries return summary, key points, open questions, and a title suggestion.

OCR and normalization quality review intentionally remain outside the default structured-output path. Conversion quality is still handled by deterministic converter status and lightweight quality gates unless an explicit future fallback is added.
