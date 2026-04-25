# Low-Level Architecture

## Key Entry And Registry Files

| File | Responsibility |
| --- | --- |
| `src/cli.py` | Builds the CLI entrypoint and runtime context |
| `src/engine/command_registry.py` | Registers the available CLI commands |
| `src/providers/base.py` | Defines the provider abstraction: `ProviderRequest` (including optional response schema hints and per-request reasoning effort), diagnostic `ProviderResponse`, and `TextProvider` |
| `src/providers/__init__.py` | Factory helpers for provider validation, config resolution from `kb.config.yaml`, and `build_provider(config)` |
| `src/providers/retry.py` | Shared Tenacity retry decorator (`provider_retry()`) for all `generate()` calls: 3 attempts, exponential backoff with jitter, transient-only retry |
| `src/providers/structured.py` | Shared structured-output parser for direct JSON, fenced JSON, and common prose-prefaced JSON plus Pydantic model validation |
| `src/providers/openai_provider.py` | OpenAI chat-completions provider; `@provider_retry()` on `generate()` |
| `src/providers/anthropic_provider.py` | Anthropic messages provider; `@provider_retry()` on `generate()` |
| `src/providers/gemini_provider.py` | Google Gemini provider; `@provider_retry()` on `generate()` and JSON-schema cleanup for SDK-compatible structured output |

## Current Command Files

| File | Responsibility |
| --- | --- |
| `src/commands/common.py` | Shared Rich-based command helpers: initialization checks, `echo_section`, `echo_bullet`, `echo_kv`, `echo_status_line`, `make_table`, `progress_report`, `emit_json`; module-level `console` and `err_console` with automatic TTY, `NO_COLOR` detection, and replacement-mode output encoding |
| `src/commands/init.py` | Project initialization behavior |
| `src/commands/add.py` | Primary source-add command, delegates to `src/commands/ingest.py` for shared implementation |
| `src/commands/ingest.py` | Shared ingest implementation for single files and directory ingest that recurses by default |
| `src/commands/update.py` | Full update workflow: add → build wiki pages → concepts → search refresh, with progress bar; delegates to `UpdateService` |
| `src/commands/find.py` | Search the compiled wiki |
| `src/commands/ask.py` | Answer a question from compiled evidence |
| `src/commands/review.py` | Semantic review command |
| `src/commands/lint.py` | Deterministic structural lint command |
| `src/commands/status.py` | Status command; `--changed` for pre-update diff view |
| `src/commands/export_cmd.py` | Vault export command; `--clean` removes stale files |
| `src/commands/doctor.py` | Project health checks |
| `src/commands/config_cmd.py` | Config display and provider management |
| `src/commands/sources.py` | Source inventory management |

## Current Service Files

| File | Responsibility |
| --- | --- |
| `src/services/project_service.py` | Project layout, initialization, Unicode-aware slug generation, and shared atomic write/copy helpers |
| `src/services/markdown_document.py` | Shared `markdown-it-py` / `python-frontmatter` helpers for frontmatter, plain text, headings, paragraphs, sections, links, and fenced-code-aware lint behavior |
| `src/services/config_service.py` | Config loading, Pydantic-backed provider/conversion validation, schema defaults, `schema_excerpt()` helper for extracting schema sections by heading, and in-place migration of legacy `kb.config.yaml` versions |
| `src/services/manifest_service.py` | Raw-source manifest read/write behavior |
| `src/services/normalization_service.py` | Document-type normalization routing for direct text inputs, Mistral OCR-backed native documents and images, `wkhtmltopdf`-rendered HTML OCR, MarkItDown-backed born-digital converters, explicit Docling/MarkItDown fallbacks, and conversion quality gates |
| `src/services/ingest_service.py` | Raw-source copy, normalized-artifact write, duplicate detection, source registration, deterministic recursive directory ingest, and callback-friendly batch progress hooks used by `kb add` |
| `src/services/compile_service.py` | Derived wiki generation with provider-backed summary generation, deterministic summary fallback on weak provider output, sentence-safe excerpts, schema-excerpt-enhanced prompts, `type: source` frontmatter, analysis-page discovery for index, parseable heading-style log entries, callback-friendly compile planning/progress hooks, and persisted resume/failure tracking for interrupted compiles |
| `src/services/concept_service.py` | Provider-first structured concept clustering with source-digest cache, deterministic NLTK/collocation fallback, concept-page generation, and backlink maintenance |
| `src/services/diff_service.py` | Pre-update source diff reporting |
| `src/services/search_service.py` | Search over compiled artifacts using a SQLite FTS5 chunk index with page-level result deduplication, evidence-section chunking, metadata-section suppression, best-chunk section/index preservation for downstream citations, concept/analysis filtering controls for callers, and fallback markdown scanning if FTS5 is unavailable |
| `src/services/query_service.py` | Provider-backed query answer assembly from primary source-page evidence while excluding generated concept pages and saved analysis pages; schema-excerpt-enhanced prompts, low-reasoning structured ask requests with larger output budget, raw citation-ref cleanup in answer prose, parseable heading-style log entries, semantic answer/citation validation, provider-status capture, and optional save-to-wiki for non-blank analysis pages that also refresh the search index and wiki index immediately |
| `src/services/review_service.py` | Provider-required semantic review: deterministic source-page topic overlap, terminology checks over reviewable source/concept pages with inflection/specificity/negating-prefix suppression, and schema-guided JSON provider review over curated source-page excerpts that rejects malformed output and filters excerpt-boundary truncation claims |
| `src/services/lint_service.py` | Structural validation for wiki links, markdown links, fragments, headings, titles, typed frontmatter (including `missing-type` warning for legacy source pages), empty pages, and maintenance findings |
| `src/services/export_service.py` | Vault export generation with atomic copies into the Obsidian view |
| `src/services/status_service.py` | Project and corpus status reporting |
| `src/services/update_service.py` | Orchestrates the full update workflow: preflight → ingest → compile → concepts → search refresh |

## Current Model Files

| File | Responsibility |
| --- | --- |
| `src/models/command_models.py` | Command-facing dataclasses and result types |
| `src/models/source_models.py` | Source metadata models |
| `src/models/wiki_models.py` | Wiki-oriented dataclasses including `ReviewReport` |

## Storage Files

| File | Responsibility |
| --- | --- |
| `src/storage/__init__.py` | Re-exports `CompileRunStore` and `SearchIndexStore` |
| `src/storage/compile_run_store.py` | JSON-backed compile-run state at `graph/exports/compile_runs.json`: active run tracking, failed-run resume candidates, and compile history |
| `src/storage/search_index_store.py` | SQLite FTS5-backed chunk index at `graph/exports/search_index.sqlite3`: tracked wiki-file inventory, versioned chunker metadata, chunk table, FTS `snippet()` output, and best-hit chunk indices for citation refs |

## Supporting Project Files

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | Dependency pins, including OpenAI/Anthropic/Gemini SDKs, Pydantic, markdown/frontmatter/NLP helpers, Mistral SDK, pdfkit, Docling, and MarkItDown, plus CLI entrypoint, Black config, pytest and coverage settings |
| `.github/workflows/tests.yml` | CI for Poetry install, Black, pytest, and coverage artifact upload |
| `tests/` | Unit, CLI, regression, and golden-file coverage for the current command/service surface |

## Low-Level Guardrails

- Keep file additions aligned with the current layer split instead of mixing CLI, service, and model logic.
- Prefer extending existing services over adding duplicate helper modules.
- Do not add new hand-rolled Markdown/frontmatter parsers; extend `markdown_document.py` unless a service has a security-specific reason to preserve a stricter path.
- Do not add new manual config validation loops; extend the Pydantic config models and preserve compatibility messages where needed.
- Do not re-expand deterministic concept clustering unless it is fallback-only; semantic clustering belongs behind the provider boundary with cached structured output.
- Treat CI and formatter config as part of the architecture because they enforce the supported workflow.
- Keep converter-backed normalization in a dedicated service instead of mixing converter logic directly into command handlers or compile.
- Preserve the canonical-artifact contract: only write `raw/normalized/` outputs after the selected converter passes the normalization quality gate or an explicit fallback succeeds.

## Structured Provider Response Usage

- `src/services/concept_service.py`: provider-first concept clustering returns structured clusters, uses medium reasoning effort, and caches them by source-page digest.
- `src/services/review_service.py`: provider-backed review accepts only structured JSON findings, rejects malformed legacy pipe output, sends curated source-page excerpts, and filters provider findings that only describe excerpt truncation.
- `src/services/query_service.py`: `kb ask` requests structured answers with markdown, claims, citations, and an insufficient-evidence flag using low reasoning effort and a larger visible-output budget; provider answers must have non-empty markdown, grounded citation refs, and claims when evidence is sufficient. Source evidence excludes generated concept pages and saved analysis pages. Saved analysis pages persist claim, citation, and provider-status metadata and refuse blank answers.
- `src/services/compile_service.py`: compile summary generation requests structured summary metadata through the shared parser, uses low reasoning effort for summary JSON, and stores key points, open questions, and title suggestions when returned.
- `src/services/lint_service.py`: saved analysis pages expose citation counts and insufficient-evidence state as frontmatter so citation discipline can be checked without text scraping.
