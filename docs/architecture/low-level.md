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
| `src/providers/anthropic_provider.py` | Anthropic messages provider; detects current Claude adaptive-thinking model identifiers by version pattern, sends the adaptive-thinking flag plus effort, keeps legacy manual thinking budgets for older models, and applies `@provider_retry()` on `generate()` |
| `src/providers/gemini_provider.py` | Google Gemini provider with stable `gemini-2.5-flash` default; `@provider_retry()` on `generate()`, model-sensitive thinking budget/level configuration, and warning-backed JSON-schema cleanup for SDK-compatible structured output |

## Current Command Files

| File | Responsibility |
| --- | --- |
| `src/commands/common.py` | Shared Rich-based command helpers: initialization checks, `echo_section`, `echo_bullet`, `echo_kv`, `echo_status_line`, `make_table`, `progress_report`, `live_status`, `lazy_live_status`, `emit_json`; module-level `console` and `err_console` with automatic TTY, `NO_COLOR` detection, and replacement-mode output encoding |
| `src/commands/init.py` | Project initialization behavior |
| `src/commands/add.py` | Primary source-add command, delegates to `src/commands/ingest.py` for shared implementation |
| `src/commands/ingest.py` | Shared ingest implementation for single files and directory ingest that recurses by default |
| `src/commands/update.py` | Full update workflow: add → build wiki pages → concepts → search refresh → GraphRAG sync/index/export, with compile progress, lazy GraphRAG status rendering, graph-output path reporting, legacy search fallback warnings, `--graph-only`, and `--allow-partial`; normal missing GraphRAG credentials warn and skip graph indexing, while graph-only missing credentials fail; delegates to `UpdateService` |
| `src/commands/find.py` | Non-generative graph-aware search entry point over direct GraphRAG entity/relationship artifacts plus source, concept, analysis, and generated graph pages; avoids provider calls and does not route to legacy ask |
| `src/commands/ask.py` | GraphRAG-aware default answer entry point with TTY-aware query status rendering and `--show-source-trace`; rejects deprecated source-evidence `--limit`, delegates to the graph ask controller, and never routes to FTS5 |
| `src/commands/legacy.py` | Deprecated SQLite FTS5 search and ask command group that invokes the legacy search/query services |
| `src/commands/review.py` | Semantic review command |
| `src/commands/lint.py` | Deterministic structural lint command |
| `src/commands/status.py` | Status command with GraphRAG health summary; `--changed` for pre-update diff view |
| `src/commands/export_cmd.py` | Vault export command that refreshes graph wiki pages when output exists; `--clean` removes only vault markdown files absent from the current export set |
| `src/commands/doctor.py` | Project health checks, including GraphRAG dependency/workspace/key/input/index/export readiness |
| `src/commands/config_cmd.py` | Config display and provider management |
| `src/commands/sources.py` | Source inventory management |

## Current Service Files

| File | Responsibility |
| --- | --- |
| `src/services/project_service.py` | Project layout, initialization, Unicode-aware slug generation, and shared atomic write/copy helpers |
| `src/services/container.py` | Typed `ServiceContainer` for command contexts, with mapping compatibility for existing tests and legacy call sites |
| `src/services/markdown_document.py` | Shared `markdown-it-py` / `python-frontmatter` helpers for frontmatter, plain text, headings, paragraphs, sections, links, and fenced-code-aware lint behavior |
| `src/services/file_lock.py` | Cross-process lock helper used around JSON/config/log state files so read-modify-write operations remain serialized on Windows and POSIX |
| `src/services/config_service.py` | Config loading, Pydantic-backed provider/conversion/graph validation, schema defaults, `schema_excerpt()` helper for extracting schema sections by heading, and lock-protected migration of legacy `kb.config.yaml` versions only after the migrated payload validates |
| `src/services/manifest_service.py` | Raw-source manifest read/write behavior with lock-protected writes, version and required-field validation, duplicate-ID/slug detection, and duplicate normalized-content-hash rejection |
| `src/services/graphrag_workspace_service.py` | Prepares the project-local `graph/graphrag/` workspace, delegates reproducible non-interactive initialization to the GraphRAG Python-entrypoint adapter, renders managed settings without writing during planning, syncs `kb.config.yaml` graph settings into GraphRAG `settings.yaml` while preserving user-owned tuning, and refreshes bundled prompt templates |
| `src/services/graphrag_command_service.py` | Adapter over GraphRAG's installed Python initialization, indexing, update, and query entrypoints, preserving structured run-result metadata and command errors without spawning `python -m graphrag` subprocesses |
| `src/services/graphrag_defaults.py` | Shared GraphRAG model, embedding, and API-key environment defaults used by command defaults and user-facing setup guidance |
| `src/services/graphrag_status_service.py` | Reports GraphRAG workspace readiness, synced input counts, active output directory preferring the latest successful complete recorded run, normalized graph state, missing required output tables, vector-store readability, wiki export presence, row counts, and lock-protected ignored local index-run metadata |
| `src/services/graphrag_sync_service.py` | Coordinates GraphRAG sync/index decisions used by `kb update`: plans workspace settings/input without preflight side effects, applies GraphRAG input when graph work can proceed, compares source/runtime digests to the last successful index, chooses full rebuild, incremental update, retry after a failed latest attempt, or skip, and records reproducibility metadata |
| `src/services/graphrag_query_service.py` | Requires a ready graph index, runs explicit GraphRAG query modes, captures answer/raw output metadata, computes the synced-input hash through shared file-digest helpers, and saves optional GraphRAG analysis pages with lint-compatible analysis frontmatter, parsed `[Data: ...]` references when available, source trace, stdout-only raw answer text, and lock-protected wiki log entries |
| `src/services/graphrag_find_service.py` | Searches active GraphRAG entity and relationship Parquet artifacts directly for top-level `kb find`, returning `graph://...` result paths before maintained-wiki matches |
| `src/services/query_router_service.py` | Deterministically chooses `basic`, `local`, `global`, or `drift` for top-level `kb ask` based on question wording and known graph entity/document terms from the status service's active output resolver |
| `src/services/graph_ask_controller_service.py` | User-facing GraphRAG ask controller: checks graph readiness and credentials, asks the router for a method, delegates to `GraphRAGQueryService`, surfaces graph staleness warnings, assigns a conservative support level, and saves analysis pages with planner metadata |
| `src/services/graphrag_wiki_export_service.py` | Reads active GraphRAG Parquet output tables through PyArrow and generates marked `wiki/graph/` markdown pages for documents, text units, entities, relationships, communities, and the graph index; preserves manual graph notes, fences raw document/text-unit content, escapes table cells, and caps high-volume relationship page export |
| `src/services/graphrag_input_sync_service.py` | Plans or syncs manifest-backed normalized artifacts into `graph/graphrag/input/sources.json`, validates the GraphRAG workspace settings file, configures JSON input columns, lists provenance metadata fields for chunk prepending, and can report skipped missing normalized artifacts for tolerant normal updates while keeping strict graph-only behavior |
| `src/services/normalization_service.py` | Document-type normalization routing for direct text inputs, Mistral OCR-backed native documents and images, HTML OCR rendered through `wkhtmltopdf` first and `xhtml2pdf` as fallback, MarkItDown-backed born-digital converters, explicit Docling/MarkItDown fallbacks, and conversion quality gates |
| `src/services/ingest_service.py` | Raw-source copy, normalized-artifact write, origin-hash duplicate short-circuiting before expensive normalization, content duplicate detection after normalization, source registration, deterministic recursive directory ingest, and callback-friendly batch progress hooks used by `kb add` |
| `src/services/compile_service.py` | Derived wiki generation with provider-backed summary generation, deterministic summary fallback on weak provider output, deterministic sentence-safe excerpts without runtime NLTK punkt data, schema-excerpt-enhanced prompts, `type: source` frontmatter, analysis-page discovery for index, lock-protected parseable heading-style log entries, callback-friendly compile planning/progress hooks, and persisted resume/failure tracking for interrupted compiles |
| `src/services/concept_service.py` | Provider-first structured concept clustering with source-digest cache, deterministic NLTK/collocation fallback, bounded unique-slug resolution, concept-page generation, and backlink maintenance |
| `src/services/diff_service.py` | Pre-update source diff reporting |
| `src/services/search_service.py` | Deprecated legacy search over compiled artifacts using a SQLite FTS5 chunk index with page-level result deduplication, evidence-section chunking, metadata-section suppression, best-chunk section/index preservation for downstream citations, concept/analysis filtering controls for callers, force-refresh retry after transient FTS failures, and fallback markdown scanning if FTS5 is unavailable |
| `src/services/query_service.py` | Provider-backed query answer assembly from primary source-page evidence while excluding generated concept pages and saved analysis pages; schema-excerpt-enhanced prompts, low-reasoning structured ask requests with larger output budget, raw citation-ref cleanup in answer prose, lock-protected parseable heading-style log entries, semantic answer/citation validation, provider-status capture, and optional save-to-wiki for non-blank analysis pages that also refresh the search index and wiki index immediately |
| `src/services/review_service.py` | Provider-required semantic review: deterministic source-page topic overlap, terminology checks over reviewable source/concept pages with inflection/specificity/negating-prefix suppression, and schema-guided JSON provider review over bounded curated source-page excerpts that rejects malformed output and filters excerpt-boundary truncation claims |
| `src/services/lint_service.py` | Structural validation for wiki links, markdown links, fragments, headings, titles, typed frontmatter (including `missing-type` warning for legacy source pages), empty pages, manifest raw/normalized artifact drift, GraphRAG completeness/input/index/export staleness, and maintenance findings |
| `src/services/export_service.py` | Vault export generation with atomic copies into the Obsidian view and `--clean` deletion based on the current run's exact exported vault-path set |
| `src/services/status_service.py` | Project, corpus, and GraphRAG status reporting |
| `src/services/update_service.py` | Orchestrates the full update workflow: preflight → ingest → compile → concepts → search refresh → GraphRAG sync/index/export, plus graph-only maintenance, normal-update missing credential warnings, tolerant missing-normalized graph input warnings, search fallback warnings, export-on-skip for complete graph output, and explicit allow-partial graph failure handling |

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
| `src/storage/compile_run_store.py` | Lock-protected JSON-backed compile-run state at `graph/exports/compile_runs.json`: active run tracking, failed-run resume candidates, and compile history |
| `src/storage/search_index_store.py` | SQLite FTS5-backed chunk index at `graph/exports/search_index.sqlite3`: tracked wiki-file inventory, versioned chunker metadata, chunk table, FTS `snippet()` output, and best-hit chunk indices for citation refs |

## Supporting Project Files

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | Dependency pins, including Microsoft GraphRAG, OpenAI/Anthropic/Gemini SDKs, Pydantic, markdown/frontmatter/NLP helpers, Mistral SDK, pdfkit, xhtml2pdf, Docling, and MarkItDown, plus CLI entrypoint, Black config, pytest and coverage settings |
| `graph/graphrag/settings.yaml` | Initialized Microsoft GraphRAG workspace settings; managed provider/model/API-key and JSON input fields are synced from `kb.config.yaml` graph defaults while unrelated user GraphRAG tuning is preserved, and `chunking.prepend_metadata` is configured for source provenance fields |
| `graph/graphrag/prompts/` | Project-managed GraphRAG prompt templates generated by `graphrag init` and refreshed from bundled templates when repository prompt defaults change |
| `graph/graphrag/input/.gitkeep` | Tracked input-directory scaffold for generated GraphRAG JSON input |
| `graph/graphrag/input/sources.json` | Generated by `kb update`; ignored because it can contain local corpus text |
| `graph/graphrag/output/` | Generated by GraphRAG indexing; ignored because it is rebuildable local runtime output |
| `graph/runs/graph_index_runs.json` | Local index-run record written by `kb update`; ignored because it can contain local paths and command output. Records include method, dry-run status, success, input digest/hash, source hashes, runtime config digest, output state, and command tails. |
| `wiki/graph/` | Generated by `kb update` and `kb export`; tracked/user-visible graph artifact layer for GraphRAG documents, entities, relationships, communities, and text units |
| `eval/benchmark.yaml` | Phase 8 benchmark questions and expected methods/sources for deprecated FTS versus GraphRAG Basic, Local, Global, and DRIFT comparison |
| `eval/results/` | Evaluation report directory for summary and CSV metrics; per-question artifacts under `eval/results/artifacts/` are ignored because they can contain local corpus text or model output |
| `scripts/evaluation_lib.py` | Shared Phase 8 evaluation logic for loading the benchmark, invoking CLI commands, computing retrieval/answer metrics, writing CSVs, and rendering the summary |
| `scripts/evaluate_graph_modes.py` | Full evaluation runner for legacy FTS, auto-router method fit, and GraphRAG mode comparison |
| `scripts/evaluate_retrieval.py` | Retrieval-focused wrapper that writes Recall@5, multi-source coverage, method-fit, and latency metrics |
| `scripts/evaluate_answers.py` | Answer-focused wrapper for claim support, insufficient-evidence behavior, comprehensiveness, diversity, and latency metrics |
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
- `src/services/query_service.py`: `kb legacy ask` requests structured answers with markdown, claims, citations, and an insufficient-evidence flag using low reasoning effort and a larger visible-output budget; provider answers must have non-empty markdown, grounded citation refs, and claims when evidence is sufficient. Source evidence excludes generated concept pages and saved analysis pages. Saved analysis pages persist claim, citation, and provider-status metadata and refuse blank answers.
- `src/services/compile_service.py`: compile summary generation requests structured summary metadata through the shared parser, uses low reasoning effort for summary JSON, and stores key points, open questions, and title suggestions when returned.
- `src/services/lint_service.py`: saved analysis pages expose citation counts and insufficient-evidence state as frontmatter so citation discipline can be checked without text scraping.
