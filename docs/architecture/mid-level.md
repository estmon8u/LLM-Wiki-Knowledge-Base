# Mid-Level Architecture

## Package Map

| Package | Responsibility |
| --- | --- |
| `src/cli.py` | CLI entrypoint and application bootstrap |
| `src/commands/` | Thin user-facing command wrappers |
| `src/services/` | Deterministic normalization, ingest, compile, concept, diff, lint, review, search, query (legacy ask), export, status, config, manifest, GraphRAG workspace/input/index/status/query/export services, plus the default graph ask controller and router |
| `src/models/` | Shared command, source, and wiki dataclasses |
| `src/engine/` | Command registry boundary |
| `src/providers/` | Provider abstraction layer with OpenAI, Anthropic, and Gemini implementations; shared structured-output parser, Tenacity retry decorator for transient failures, per-request reasoning/output controls, diagnostics on provider responses, Gemini schema adaptation, and catalog-backed provider resolution |
| `src/storage/` | Compile-run state persistence and SQLite FTS5 chunk-index storage |
| `scripts/` | Operational scripts, including Phase 8 evaluation runners for retrieval and answer-mode comparison |
| `eval/` | Benchmark definitions, legacy captures, and generated evaluation reports |

## Command To Service Mapping

Most commands are flat top-level verbs. The GraphRAG pivot keeps the deprecated lexical path behind the explicit `legacy` group, while GraphRAG behavior is folded into the main command surface:

| Click Name | Command Wrapper | Main Service |
| --- | --- | --- |
| `init` | `src/commands/init.py` | `src/services/project_service.py`, `src/services/config_service.py`, and `src/services/graphrag_workspace_service.py` |
| `add` | `src/commands/add.py` | `src/services/ingest_service.py`, `src/services/normalization_service.py`, and `src/services/manifest_service.py` |
| `update` | `src/commands/update.py` | `src/services/compile_service.py`, `src/services/concept_service.py`, `src/services/search_service.py`, `src/services/graphrag_sync_service.py`, and `src/services/graphrag_wiki_export_service.py` |
| `find` | `src/commands/find.py` | Reserved GraphRAG guidance wrapper until graph query support lands |
| `ask` | `src/commands/ask.py` | `src/services/graph_ask_controller_service.py` and `src/services/query_router_service.py` |
| `legacy find` / `legacy ask` | `src/commands/legacy.py` | `src/services/search_service.py` and `src/services/query_service.py` |
| `lint` | `src/commands/lint.py` | `src/services/lint_service.py` and `src/services/graphrag_status_service.py` |
| `review` | `src/commands/review.py` | `src/services/review_service.py` |
| `status` | `src/commands/status.py` | `src/services/status_service.py`, `src/services/graphrag_status_service.py`, and `src/services/diff_service.py` (with `--changed`) |
| `export` | `src/commands/export_cmd.py` | `src/services/export_service.py` and `src/services/graphrag_wiki_export_service.py` |
| `doctor` | `src/commands/doctor.py` | `src/services/doctor_service.py` and `src/services/graphrag_status_service.py` |
| `config` | `src/commands/config_cmd.py` | `src/services/config_service.py` |
| `sources` | `src/commands/sources.py` | `src/services/manifest_service.py` |

## Data Flow

| Stage | Input | Output |
| --- | --- | --- |
| Ingest | canonical markdown/plain-text files, Mistral OCR-routed native documents and images, rendered HTML-to-PDF OCR, and a bounded MarkItDown subset | raw source copy, normalized artifact, and manifest metadata |
| Compile | normalized canonical text plus manifest metadata | source pages with provider-generated summaries, wiki index, and compile log; optional legacy concept pages with deterministic fallback and source-page backlinks when concept generation is explicitly enabled |
| Diff | manifest metadata plus compile state | pre-compile source status preview |
| Legacy search | compiled wiki artifacts | ranked page matches from source pages, generated concept pages, and saved analysis pages, derived from indexed chunks that skip wiki bookkeeping sections |
| Legacy ask | user question plus source-page evidence, excluding generated concept pages and saved analysis pages | cited provider answer validated for parseability, non-empty content, and grounded citation refs; optionally saved as a non-blank analysis page |
| GraphRAG workspace | project GraphRAG workspace path plus `kb.config.yaml` graph defaults | GraphRAG CLI-initialized settings, prompts, and input scaffold under `graph/graphrag/`, with provider/model/embedding/API-key values synced from config during `kb init` and `kb update` |
| GraphRAG input/index | normalized artifacts plus manifest metadata during `kb update` | `graph/graphrag/input/sources.json` JSON records with source text and provenance metadata; JSON input settings; auto full/update/retry/skip index metadata; generated output tables, active output directory metadata, and ignored run metadata |
| Default graph ask | user question plus `--method auto|basic|local|global|drift` and graph readiness status | GraphRAG answer metadata with deterministic route reason, planner metadata, source trace, and conservative support level; optional saved analysis page with retriever/method/planner/claim_support/index hash metadata |
| GraphRAG wiki export | active complete GraphRAG Parquet output tables during `kb update` or `kb export` | generated markdown graph pages under `wiki/graph/` for documents, text units, entities, relationships, and communities; export also runs when update skips indexing because complete output is current; raw source text is fenced and high-volume relationship page export is capped while row counts remain visible |
| Evaluation | benchmark questions plus an initialized KB project | `eval/results/summary.md`, `retrieval_metrics.csv`, `answer_metrics.csv`, and ignored per-question command artifacts |
| Lint | compiled wiki and metadata | structural findings for links, fragments, headings, titles, typed frontmatter, empty pages, graph staleness, and maintenance signals |
| Review | compiled source/concept pages | semantic findings from deterministic overlap checks over source pages, terminology-variant checks over reviewable source/concept pages, and schema-guided single-pass provider review over curated source-page excerpts |
| Export | compiled wiki | Obsidian-friendly vault view |

## Current Ingest Scope

- The current implementation adds `.md`, `.markdown`, and `.txt` files directly; routes `.pdf`, `.docx`, `.pptx`, `.png`, `.jpg`, `.jpeg`, and `.avif` through Mistral OCR first; renders `.html` / `.htm` to PDF with `wkhtmltopdf` first and pure-Python `xhtml2pdf` as fallback before OCR; and uses MarkItDown for the remaining bounded born-digital subset such as CSV, notebooks, EPUB, and Excel files.
- `kb add` is the primary ingestion command; `src/commands/ingest.py` provides the shared implementation.
- Directory inputs for `kb add` walk recursively by default, add only supported source files, and leave unsupported files untouched.
- Conversion quality gates reject empty, implausibly short, or suspiciously truncated outputs before `raw/normalized/` artifacts are written. PDF, DOCX, PPTX, and HTML routes then fall back explicitly to Docling or MarkItDown based on config.

## Structural Rules

- Commands should stay thin and delegate quickly.
- The command layer owns terminal-only concerns such as section headings, list formatting, progress bars, and lazy status spinners via Rich (`Console`, `Table`, `Progress`, `Status`); long-running services expose callback-friendly hooks instead of writing directly to the terminal. User-supplied content is markup-escaped via `rich.markup.escape`.
- Services should remain deterministic unless the feature explicitly requires model-backed synthesis.
- GraphRAG orchestration should wrap the official `graphrag` CLI/library instead of reimplementing graph indexing or query modes.
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

Provider-backed semantic steps now request structured responses at the service boundary instead of parsing freeform text. Concept generation returns concept clusters with title, summary, topic terms, and source pages. Review returns JSON findings with severity, code, pages, and message, then filters findings that only reflect curated-excerpt boundaries. `kb legacy ask` returns answer markdown, claims, citations, and an insufficient-evidence flag, and rejects answers that are syntactically valid but empty or ungrounded. Compile summaries return summary, key points, open questions, and a title suggestion.

The provider request boundary also carries operation-specific reasoning effort and token budgets. Schema-bound operational tasks can request lower reasoning effort with enough visible-output budget for valid JSON, while Gemini receives a schema subset that removes unsupported `additionalProperties` keys before SDK submission.

OCR and normalization quality review intentionally remain outside the default structured-output path. Conversion quality is still handled by deterministic converter status and lightweight quality gates unless an explicit future fallback is added.
