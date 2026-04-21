# Low-Level Architecture

## Key Entry And Registry Files

| File | Responsibility |
| --- | --- |
| `src/cli.py` | Builds the CLI entrypoint and runtime context |
| `src/engine/command_registry.py` | Registers the available CLI commands |
| `src/engine/tool_registry.py` | Holds the internal tool boundary for future agent-style actions |
| `src/providers/base.py` | Defines the provider abstraction: `ProviderRequest`, `ProviderResponse`, `TextProvider` |
| `src/providers/__init__.py` | Factory `build_provider(config, resolved=)` — lazy-imports the right provider by name; accepts an optional `ResolvedProviderConfig` from the model registry for tier-driven reasoning effort and thinking budget |
| `src/providers/retry.py` | Shared Tenacity retry decorator (`provider_retry()`) for all `generate()` calls: 3 attempts, exponential backoff with jitter, transient-only retry |
| `src/providers/openai_provider.py` | OpenAI chat-completions provider; reasoning effort set by tier profile; `@provider_retry()` on `generate()` |
| `src/providers/anthropic_provider.py` | Anthropic messages provider; thinking budget set by tier profile; `@provider_retry()` on `generate()` |
| `src/providers/gemini_provider.py` | Google Gemini provider; reasoning effort set by tier profile; `@provider_retry()` on `generate()` |

## Current Command Files

| File | Responsibility |
| --- | --- |
| `src/commands/common.py` | Shared Rich-based command helpers: initialization checks, `echo_section`, `echo_bullet`, `echo_kv`, `echo_status_line`, `make_table`, `progress_report`, `emit_json`; module-level `console` and `err_console` with automatic TTY and `NO_COLOR` detection |
| `src/commands/init.py` | Project initialization behavior |
| `src/commands/add.py` | Primary source-add command, delegates to `src/commands/ingest.py` for shared implementation |
| `src/commands/ingest.py` | Shared ingest implementation for single files and directory ingest that recurses by default |
| `src/commands/update.py` | Full update workflow: add → build wiki pages → concepts → search refresh, with progress bar; delegates to `UpdateService` |
| `src/commands/find.py` | Search the compiled wiki |
| `src/commands/ask.py` | Answer a question from compiled evidence; `--quality` implies model tier |
| `src/commands/review.py` | Semantic review command; `--deep` runs extractor/skeptic/arbiter |
| `src/commands/lint.py` | Deterministic structural lint command |
| `src/commands/status.py` | Status command; `--changed` for pre-update diff view |
| `src/commands/export_cmd.py` | Vault export command; `--clean` removes stale files |
| `src/commands/compile.py` | Standalone compile command with `--force`, `--with-concepts`, and `--resume` flags |
| `src/commands/doctor.py` | Project health checks |
| `src/commands/history.py` | Run history display with public command names |
| `src/commands/config_cmd.py` | Config display, provider management (validated `click.Choice`), model tier inspection |
| `src/commands/sources.py` | Source inventory management |

## Current Service Files

| File | Responsibility |
| --- | --- |
| `src/services/project_service.py` | Project layout, initialization, and shared atomic write/copy helpers |
| `src/services/config_service.py` | Config loading, defaults, and in-place migration of legacy `kb.config.yaml` versions |
| `src/services/manifest_service.py` | Raw-source manifest read/write behavior |
| `src/services/normalization_service.py` | Document-type normalization routing for direct text inputs, Docling-backed PDFs, and bounded MarkItDown-backed born-digital converters |
| `src/services/ingest_service.py` | Raw-source copy, normalized-artifact write, duplicate detection, source registration, deterministic recursive directory ingest, and callback-friendly batch progress hooks used by `kb add` |
| `src/services/compile_service.py` | Derived wiki generation with provider-backed summary generation, callback-friendly compile planning/progress hooks, and persisted resume/failure tracking for interrupted compiles |
| `src/services/diff_service.py` | Pre-update source diff reporting |
| `src/services/search_service.py` | Search over compiled artifacts using a SQLite FTS5 chunk index with page-level result deduplication, best-chunk section/index preservation for downstream citations, and fallback markdown scanning if FTS5 is unavailable |
| `src/services/query_service.py` | Provider-backed query answer assembly from maintained wiki context; self-consistency sampling, claim normalization, deterministic merge, frozen evidence bundles that retain chunk refs, and optional save-to-wiki for analysis pages that also refresh the search index |
| `src/services/review_service.py` | Provider-required semantic review: deterministic topic overlap and terminology checks, single-pass provider review, adversarial extractor/skeptic/arbiter review, and review-run persistence |
| `src/services/lint_service.py` | Structural validation for wiki links, markdown links, fragments, headings, titles, typed frontmatter, empty pages, and maintenance findings |
| `src/services/export_service.py` | Vault export generation with atomic copies into the Obsidian view |
| `src/services/status_service.py` | Project and corpus status reporting |
| `src/services/model_registry_service.py` | Resolves (provider, tier, model) triples into `ResolvedProviderConfig`; built-in profiles per provider/tier; task-specific default tiers; priority: runtime `--model` > runtime `--tier` > config tier > config model > task default > balanced |
| `src/services/update_service.py` | Orchestrates the full update workflow: preflight → ingest → compile → concepts → search refresh |

## Current Model Files

| File | Responsibility |
| --- | --- |
| `src/models/command_models.py` | Command-facing dataclasses and result types |
| `src/models/source_models.py` | Source metadata models |
| `src/models/tool_models.py` | Tool-facing data structures |
| `src/models/provider_models.py` | `ModelProfile` and `ResolvedProviderConfig` dataclasses for the tier registry |
| `src/models/wiki_models.py` | Wiki-oriented dataclasses including `ReviewReport` with typed findings and optional `run_id` |

## Schema Files (Pydantic)

| File | Responsibility |
| --- | --- |
| `src/schemas/__init__.py` | Re-exports all schema types |
| `src/schemas/claims.py` | `EvidenceItem`, `EvidenceBundle` (with deterministic `context_hash`), `Claim`, `CandidateAnswer`, `MergedAnswer` |
| `src/schemas/review.py` | `Verdict` enum, `ReviewFinding` |
| `src/schemas/runs.py` | `RunRecord` — full deliberation artifact with auto-generated run ID and timestamp |

## Storage Files

| File | Responsibility |
| --- | --- |
| `src/storage/__init__.py` | Re-exports `CompileRunStore`, `RunStore`, and `SearchIndexStore` |
| `src/storage/compile_run_store.py` | JSON-backed compile-run state at `graph/exports/compile_runs.json`: active run tracking, failed-run resume candidates, and compile history |
| `src/storage/search_index_store.py` | SQLite FTS5-backed chunk index at `graph/exports/search_index.sqlite3`: tracked wiki-file inventory, chunk table, and FTS virtual table used by `SearchService`, including best-hit chunk indices for citation refs |
| `src/storage/run_store.py` | SQLite-backed persistence at `graph/exports/run_artifacts.sqlite3`: `runs` table (full record JSON + indexed columns), `run_citations` table (normalized claim→page/section index) |

## Supporting Project Files

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | Dependency pins, including Docling and MarkItDown, plus CLI entrypoint, Black config, pytest and coverage settings |
| `.github/workflows/tests.yml` | CI for Poetry install, Black, pytest, and coverage artifact upload |
| `tests/` | Unit, CLI, regression, and golden-file coverage for the current command/service surface |

## Low-Level Guardrails

- Keep file additions aligned with the current layer split instead of mixing CLI, service, and model logic.
- Prefer extending existing services over adding duplicate helper modules.
- Treat CI and formatter config as part of the architecture because they enforce the supported workflow.
- Keep converter-backed normalization in a dedicated service instead of mixing converter logic directly into command handlers or compile.
