# Mid-Level Architecture

## Package Map

| Package | Responsibility |
| --- | --- |
| `src/graphwiki_kb/cli.py` | CLI entrypoint and application bootstrap |
| `src/graphwiki_kb/commands/` | Thin user-facing command wrappers |
| `src/graphwiki_kb/agents/` | Optional OpenAI Agents SDK control-plane runtime, prompts, context, typed tool registry, and service-backed tool wrappers for `kb agent` |
| `src/graphwiki_kb/services/` | Deterministic normalization, ingest, compile, concept, diff, lint, review, search, query (legacy ask), export, status, config, manifest, file-locking, GraphRAG workspace/input/index/status/query/export services, the default graph ask controller/router, and `kb agent` orchestration/research/recommendation services |
| `src/graphwiki_kb/models/` | Shared command, source, and wiki dataclasses |
| `src/graphwiki_kb/engine/` | Command registry boundary |
| `src/graphwiki_kb/providers/` | Provider abstraction layer with OpenAI Responses API plus Chat Completions fallback, Anthropic, and Gemini implementations; shared structured-output parser, retryable-status-aware Tenacity retry decorator, per-request reasoning/output controls, diagnostics on provider responses, OpenAI reasoning-argument gating, Anthropic adaptive thinking plus `output_config.format` schemas, Gemini schema subset normalization/reporting, and catalog-backed provider resolution |
| `src/graphwiki_kb/storage/` | Lock-protected compile-run state persistence and SQLite FTS5 chunk-index storage |
| `scripts/` | Operational scripts, including Phase 8 evaluation runners for retrieval and answer-mode comparison |
| `eval/` | Benchmark definitions, legacy captures, and generated evaluation reports |

## Command To Service Mapping

Most commands are flat top-level verbs. The GraphRAG pivot keeps the deprecated lexical path behind the explicit `legacy` group, while GraphRAG behavior is folded into the main command surface:

| Click Name | Command Wrapper | Main Service |
| --- | --- | --- |
| `init` | `src/graphwiki_kb/commands/init.py` | `src/graphwiki_kb/services/project_service.py`, `src/graphwiki_kb/services/config_service.py`, and `src/graphwiki_kb/services/graphrag_workspace_service.py` |
| `add` | `src/graphwiki_kb/commands/add.py` | `src/graphwiki_kb/services/ingest_service.py`, `src/graphwiki_kb/services/normalization_service.py`, and `src/graphwiki_kb/services/manifest_service.py` |
| `agent` | `src/graphwiki_kb/commands/agent.py` | `src/graphwiki_kb/services/agent_service.py`, `src/graphwiki_kb/agents/tool_registry.py`, `src/graphwiki_kb/services/research_service.py`, `src/graphwiki_kb/services/source_recommendation_store.py`, and `src/graphwiki_kb/services/web_source_acquisition_service.py` |
| `update` | `src/graphwiki_kb/commands/update.py` | `src/graphwiki_kb/services/compile_service.py`, `src/graphwiki_kb/services/concept_service.py`, `src/graphwiki_kb/services/search_service.py`, `src/graphwiki_kb/services/graphrag_sync_service.py`, and `src/graphwiki_kb/services/graphrag_wiki_export_service.py` |
| `find` | `src/graphwiki_kb/commands/find.py` | `src/graphwiki_kb/services/graphrag_find_service.py` for direct graph artifacts plus `src/graphwiki_kb/services/search_service.py` for non-generative maintained-wiki navigation, deduped and globally ranked after both candidate sets are collected |
| `ask` | `src/graphwiki_kb/commands/ask.py` | `src/graphwiki_kb/services/graph_ask_controller_service.py` and `src/graphwiki_kb/services/query_router_service.py` |
| `legacy find` / `legacy ask` | `src/graphwiki_kb/commands/legacy.py` | `src/graphwiki_kb/services/search_service.py` and `src/graphwiki_kb/services/query_service.py` |
| `lint` | `src/graphwiki_kb/commands/lint.py` | `src/graphwiki_kb/services/lint_service.py` and `src/graphwiki_kb/services/graphrag_status_service.py` |
| `review` | `src/graphwiki_kb/commands/review.py` | `src/graphwiki_kb/services/review_service.py` |
| `status` | `src/graphwiki_kb/commands/status.py` | `src/graphwiki_kb/services/status_service.py`, `src/graphwiki_kb/services/graphrag_status_service.py`, and `src/graphwiki_kb/services/diff_service.py` (with `--changed` and `--strict`) |
| `export` | `src/graphwiki_kb/commands/export_cmd.py` | `src/graphwiki_kb/services/export_service.py` and `src/graphwiki_kb/services/graphrag_wiki_export_service.py` |
| `doctor` | `src/graphwiki_kb/commands/doctor.py` | `src/graphwiki_kb/services/doctor_service.py` and `src/graphwiki_kb/services/graphrag_status_service.py` |
| `config` | `src/graphwiki_kb/commands/config_cmd.py` | `src/graphwiki_kb/services/config_service.py` |
| `sources` | `src/graphwiki_kb/commands/sources.py` | `src/graphwiki_kb/services/manifest_service.py` |

## Data Flow

| Stage | Input | Output |
| --- | --- | --- |
| Ingest | canonical markdown/plain-text files, Mistral OCR-routed native documents and images, rendered HTML-to-PDF OCR, and a bounded MarkItDown subset | raw source copy, normalized artifact, and manifest metadata |
| Compile | normalized canonical text plus manifest metadata | source pages with provider-generated summaries, wiki index, and compile log; optional legacy concept pages with deterministic fallback and source-page backlinks when concept generation is explicitly enabled |
| Diff | manifest metadata plus compile state | pre-compile source status preview |
| Graph-aware find | active GraphRAG entity/relationship tables plus compiled wiki artifacts | ranked top-level `kb find` matches from direct graph artifacts first, then source pages, generated concept pages, saved analysis pages, and generated graph pages derived from indexed chunks that skip wiki bookkeeping sections; graph artifact read failures are reported as diagnostics |
| Legacy search | compiled wiki artifacts | deprecated `kb legacy find` comparator matches from source pages, generated concept pages, and saved analysis pages, derived from indexed chunks that skip wiki bookkeeping sections |
| Legacy ask | user question plus source-page evidence, excluding generated concept pages and saved analysis pages | cited provider answer validated for parseability, non-empty content, and grounded citation refs; optionally saved as a non-blank analysis page |
| GraphRAG workspace | project GraphRAG workspace path plus `kb.config.yaml` graph defaults | GraphRAG Python-entrypoint-initialized settings, wheel-safe prompt templates, and input scaffold under `graph/graphrag/`, with managed provider/model/embedding/API-key values synced from config during `kb init` and `kb update` while unrelated GraphRAG tuning is preserved; prompt templates are copied only when missing and changed bundled defaults become `*.new` candidates |
| GraphRAG input/index | normalized artifacts plus manifest metadata during `kb update` | planned non-mutating preflight state; tolerant skipped-missing-artifact warnings during normal updates; compact `graph/graphrag/input/sources.json` JSON records with source text and provenance metadata when graph work applies; JSON input settings; explicit `--graph-method` override support; auto full/update/retry/skip index metadata; generated output tables, active output directory metadata from the latest successful complete run, ignored run metadata, and recorded early failures |
| Default graph ask | user question plus `--method auto|basic|local|global|drift` and graph readiness status | GraphRAG answer metadata with deterministic visible route reason, planner metadata, source trace, graph-input hash, raw-manifest hash when available, and conservative support level; returned non-streaming GraphRAG answers are surfaced when stdout is empty; optional saved analysis page with retriever/method/planner/claim_support/index-run metadata |
| Agent turn | natural-language request plus optional session id and approval mode | OpenAI Agents SDK run over typed tools; durable `agent-run-*.json` trace; optional `sessions.sqlite`; research runs and recommendations under `graph/runs/agent/`; web findings separated from local KB answers; write tools paused behind approval or `--yes` |
| GraphRAG wiki export | active complete GraphRAG Parquet output tables during `kb update` or `kb export` | generated markdown graph pages under `wiki/graph/` for documents, text units, entities, relationships, and communities; export also runs when update skips indexing because complete output is current; raw source text is fenced and high-volume relationship page export is capped while row counts remain visible |
| Evaluation | benchmark questions plus an initialized KB project | `eval/results/summary.md`, `retrieval_metrics.csv`, `answer_metrics.csv`, and ignored per-question command artifacts |
| Lint | compiled wiki and metadata | structural findings for links, fragments, headings, titles, typed frontmatter, empty pages, raw/normalized manifest artifact drift, graph staleness, and maintenance signals |
| Review | compiled source/concept pages | semantic findings from deterministic overlap checks over source pages, terminology-variant checks over reviewable source/concept pages, and schema-guided single-pass provider review over curated source-page excerpts |
| Export | compiled wiki | Obsidian-friendly vault view; clean mode removes only markdown paths absent from the current export set |

## Current Ingest Scope

- The current implementation adds `.md`, `.markdown`, and `.txt` files directly; routes `.pdf`, `.docx`, `.pptx`, `.png`, `.jpg`, `.jpeg`, and `.avif` through Mistral OCR first because OCR accuracy determines downstream wiki, graph, citation, and answer quality; renders `.html` / `.htm` to PDF with `wkhtmltopdf` first and pure-Python `xhtml2pdf` as fallback before OCR; and uses MarkItDown for the remaining bounded born-digital subset such as CSV, notebooks, EPUB, and Excel files.
- `kb add` is the primary ingestion command; `src/graphwiki_kb/commands/ingest.py` provides the shared implementation.
- Directory inputs for `kb add` walk recursively by default, add only supported source files, and leave unsupported files untouched.
- Conversion quality gates reject empty, implausibly short, or suspiciously truncated outputs before `raw/normalized/` artifacts are written. PDF routes then fall back through the configured ordered local chain, Docling then MarkItDown by default, while DOCX, PPTX, and HTML routes fall back explicitly to MarkItDown based on config.

## Structural Rules

- Commands should stay thin and delegate quickly.
- The command layer owns terminal-only concerns such as section headings, list formatting, progress bars, and lazy status spinners via Rich (`Console`, `Table`, `Progress`, `Status`); long-running services expose callback-friendly hooks instead of writing directly to the terminal. User-supplied content is markup-escaped via `rich.markup.escape`.
- Services should remain deterministic unless the feature explicitly requires model-backed synthesis.
- GraphRAG orchestration should wrap the official `graphrag` Python entrypoints/library instead of reimplementing graph indexing or query modes. The wrapper must inspect callable signatures, supply known additive defaults, filter unsupported kwargs, and fail early on unknown required parameters.
- `kb agent` tools should call existing services directly, preserve local KB versus web-research boundaries, and require approval for mutations. Research can recommend sources, but ingestion remains a separate approved action.
- Shared parsing belongs in `src/graphwiki_kb/services/markdown_document.py`: services should consume parser-backed markdown/frontmatter helpers instead of adding new ad hoc regex stacks.
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

Provider-backed semantic steps now request structured responses at the service boundary instead of parsing freeform text. Concept generation returns concept clusters with title, summary, topic terms, and source pages. Review returns JSON findings with severity, code, pages, and message, then filters findings that only reflect curated-excerpt boundaries. `kb legacy ask` returns answer markdown, claims, citations, and an insufficient-evidence flag, and rejects answers that are syntactically valid but empty or ungrounded. Compile summaries return summary, key points, open questions, and a title suggestion.

The provider request boundary also carries operation-specific reasoning effort and token budgets. Schema-bound operational tasks can request lower reasoning effort with enough visible-output budget for valid JSON, OpenAI receives reasoning arguments only for known reasoning-capable model families, Anthropic adaptive-thinking models receive the required thinking flag plus the requested effort using version-pattern detection for current Claude identifiers, and Gemini receives model-sensitive thinking configuration while preserving supported JSON Schema `additionalProperties`.

OCR and normalization quality review intentionally remain outside the default structured-output path. Conversion quality is still handled by deterministic converter status and lightweight quality gates unless an explicit future fallback is added.
