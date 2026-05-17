# High-Level Architecture

## Purpose

The project is pivoting into a CLI-first GraphRAG research-memory system with inspectable wiki artifacts. The target product ingests technical documents, preserves provenance, builds a graph-based retrieval index, answers local and global research questions, and exports human-readable markdown artifacts for inspection and maintenance.

The wiki is not the retrieval engine. The wiki is the human-readable artifact layer. GraphRAG is the retrieval and synthesis engine.

The architecture accepts heterogeneous source documents through a normalization step into canonical markdown or plain text. The current implementation routes canonical markdown and plain text directly, uses Mistral OCR as the paid, high-accuracy primary converter for explicitly supported native formats (`.pdf`, `.docx`, `.pptx`, `.png`, `.jpg`, `.jpeg`, `.avif`), and renders `.html` / `.htm` through `wkhtmltopdf` or the pure-Python `xhtml2pdf` fallback before sending the PDF bytes to the same OCR path. This is deliberate because conversion quality feeds every downstream compiled page, GraphRAG entity, relationship, retrieval result, citation, and answer. MarkItDown remains the default for the remaining bounded born-digital subset, while Docling and MarkItDown are lower-confidence fallbacks used only after the primary OCR route fails quality checks.

The product goal is not to act like a general-purpose coding agent. The goal is to ingest source material, compile inspectable wiki artifacts, answer questions through a graph-first retrieval and synthesis path, keep any old FTS5 path explicit and deprecated, and export an Obsidian-friendly vault.

## Core User Flow

1. Initialize a project workspace.
2. Ingest supported source documents, store the originals, and normalize them into canonical markdown or plain text in the raw layer.
3. Compile source pages, refresh the deprecated legacy FTS index used only by `kb legacy ...` commands, sync normalized artifacts into the GraphRAG workspace, and automatically refresh, retry, or skip the graph index based on source/runtime changes and latest index-run state.
4. Build graph outputs for graph-first retrieval modes when update selects a full or incremental GraphRAG index job, and refresh graph wiki pages from complete output even when indexing is skipped as current.
5. Search direct graph artifacts plus the maintained wiki index through top-level `kb find`, and ask GraphRAG-backed questions by default; access the old FTS path only through explicit `kb legacy ...` commands.
6. Optionally save useful answers back into the wiki as persistent analysis pages.
7. Evaluate deprecated FTS versus GraphRAG Basic, Local, Global, and DRIFT modes against the benchmark.
8. Lint the maintained knowledge base for broken structure or stale content (deterministic), including manifest artifact drift and GraphRAG input/index/export staleness.
9. Review the maintained knowledge base for contradictions, terminology drift, and topic overlap (semantic; requires a configured provider and combines deterministic overlap checks with provider-backed review).
10. Export the wiki into an Obsidian-friendly vault and refresh graph inspection pages when graph output exists.

## Data Domains

- `raw/` stores source-of-truth input files, normalized canonical artifacts, and manifest metadata.
- `wiki/` stores generated source pages, legacy concept pages, GraphRAG-derived graph pages, saved analysis pages, index data, and compile logs.
- `vault/` stores export-ready Obsidian-friendly markdown.
- `graph/` stores the SQLite FTS5 search index and compile-run state under `graph/exports/`, plus the initialized Microsoft GraphRAG workspace under `graph/graphrag/` and generated GraphRAG input under `graph/graphrag/input/sources.json`.

## System Boundaries

- Commands expose user-facing CLI behavior.
- Services own deterministic business logic.
- Models hold shared dataclasses and typed results.
- Engine modules register commands.
- Providers abstract model-backed behavior behind a small boundary with concrete implementations for OpenAI, Anthropic, and Google Gemini; compile, ask, and review require a configured provider, while the rest of the CLI remains deterministic. OpenAI uses the Responses API by default with a Chat Completions fallback. Provider responses include model/provider diagnostics when SDKs expose them, provider selection, built-in provider settings, and conversion defaults now live together in `kb.config.yaml`, and a single provider instance is shared across all services via `build_provider(config)`. A shared Tenacity retry policy (`src/graphwiki_kb/providers/retry.py`) wraps all `generate()` calls with exponential backoff and jitter for transient failures (rate limits, timeouts, server errors).
- CLI output uses Rich for styled tables, progress bars, lazy status spinners, and colored terminal output. All user-facing content is markup-escaped. The `NO_COLOR` environment variable and non-TTY detection are respected automatically, and stdout/stderr are configured to replace unsupported terminal characters instead of crashing on Windows code pages. Machine-readable `--json` flags are available on `doctor`, `find`, `status`, and `sources list`.
- Search storage currently persists a rebuildable SQLite FTS5 chunk index at `graph/exports/search_index.sqlite3` so lexical wiki navigation no longer scans every markdown file on each query. Top-level `kb find` reads direct GraphRAG entity/relationship artifacts and the maintained wiki search index across source, concept, analysis, and generated graph pages, then deduplicates and globally ranks the combined candidate set. In the GraphRAG pivot, legacy FTS behavior is temporary comparator infrastructure, not a peer default: `kb legacy find` is source-page-only lexical lookup and `kb legacy ask` is source-page-only provider synthesis. Top-level `kb ask` fails with clear next-step guidance when the GraphRAG workspace, input, output tables, or vector store are missing, empty, unreadable, or incompatible instead of silently falling back to FTS5.
- The GraphRAG workspace is initialized under `graph/graphrag/` through a signature-aware adapter over GraphRAG's installed Python initialization entrypoint, with tracked settings, prompts, and input scaffolding. `kb init` creates the workspace and syncs managed `kb.config.yaml` graph provider/model/embedding/API-key settings into `graph/graphrag/settings.yaml` while preserving unrelated user-owned GraphRAG tuning; bundled prompts are included in source and wheel builds and refreshed from packaged or repository templates when defaults change. `kb update` plans settings/input sync without mutating workspace files during preflight, applies the planned input when indexing can proceed, warns and skips isolated missing normalized artifacts during normal mixed wiki/graph updates while keeping `--graph-only` strict, records input digests, source hashes, runtime/settings/prompt digests, the installed GraphRAG version, and managed schema versions, then auto-selects full `fast`, incremental `fast-update`, retry after a failed latest run, or skip. `kb status` reports settings, input, the active output directory from the latest successful complete run, output tables, vector-store state, last recorded index-run metadata, and digest-based freshness so complete-looking output is stale when run metadata is missing or mismatched. `kb ask --method auto|basic|local|global|drift` is the user-facing GraphRAG answer controller with deterministic mode routing, method-specific readiness checks, active-output data-directory routing through GraphRAG's Python query entrypoints, returned-answer fallback when stdout is empty, blank-save refusal, unique saved analysis filenames, raw stdout/stderr preservation, and conservative support-level metadata; and `kb export` turns active GraphRAG output tables into inspectable markdown under `wiki/graph/` when output exists. Local `.env`, generated input, output, cache, logs, and graph run metadata are ignored. The old `kb graph` command group has been removed.
- Mutable local state files such as `kb.config.yaml`, `raw/_manifest.json`, `graph/exports/compile_runs.json`, GraphRAG index-run metadata, and `wiki/log.md` use lock-protected write paths where concurrent command runs could otherwise lose updates. GraphRAG workspace init/index/query/status/export operations also share a workspace-level lock so multi-file graph mutations and reads do not overlap inside one project.
- Phase 8 evaluation scripts compare the deprecated FTS path against GraphRAG Basic, Local, Global, and DRIFT modes using `eval/benchmark.yaml`. The default evaluator run is local-safe: it runs legacy find and auto-router checks, then skips provider-backed answer rows unless `--allow-provider-calls` is explicit.
- `kb.schema.md` is the wiki's operational constitution. Relevant schema sections are injected into compile and ask prompts via `schema_excerpt()`, so the LLM follows wiki-maintenance rules.
- Markdown and frontmatter parsing are centralized in `src/graphwiki_kb/services/markdown_document.py` using `markdown-it-py` and `python-frontmatter`; services consume parser-backed helpers instead of maintaining parallel regex/state-machine implementations.
- `kb.config.yaml` validation is Pydantic-backed while retaining user-facing validation messages for existing workflows.
- Provider requests can carry an optional response schema, per-operation reasoning effort, and per-operation output budget. Concept generation, review findings, legacy ask answers, and compile summaries use structured provider outputs for semantic tasks where the model response drives stored wiki state. A shared structured-output parser accepts direct JSON, fenced JSON, and common prose-prefaced JSON before schema validation; OpenAI receives reasoning arguments only for known reasoning-capable model families, Gemini preserves currently supported JSON Schema `additionalProperties` and rejects unsupported `xhigh` reasoning effort instead of silently downgrading it, and Anthropic adaptive-thinking support is based on current Claude model version patterns rather than one hard-coded model name. `kb legacy ask` then rejects semantically empty answers, missing/ungrounded claims, and unknown citation refs, while review rejects malformed or empty provider JSON instead of parsing legacy pipe-delimited lines and filters excerpt-only truncation claims.
- Concept clustering is provider-first during `kb update`: the provider returns structured concept clusters over source-page titles/summaries, results are cached by source-page digest, and the deterministic NLTK/collocation pipeline remains a fallback.
- `kb legacy ask --save` persists structured answer metadata (`insufficient_evidence`, claim count, citation count, structured claims, and provider status) so saved analysis pages can be linted without text heuristics, and refuses to write blank analysis pages. Saved analysis pages remain searchable through top-level `kb find`, but legacy find and legacy ask stay source-only to avoid recursively citing prior generated answers; legacy ask also excludes generated concept pages so provider answers stay grounded in primary compiled source pages. Repeated saves get unique log headings so `wiki/log.md` remains lint-clean.
- Conversion is config-driven rather than hard-coded by suffix alone. Mistral OCR is the default high-accuracy path for the explicitly supported native document and image formats, PDF fallbacks are ordered after Mistral as Docling then MarkItDown, HTML uses a rendered-PDF OCR route, inline OCR payloads are size-checked before bytes are read or sent to the SDK, and converter quality gates prevent partial or obviously truncated artifacts from becoming canonical markdown.
- Any post-OCR LLM cleanup or reconstruction should remain explicit fallback behavior rather than becoming part of the default ingest path.

## Reference-Project Roles

- Browzy.ai is the closest reference for the knowledge-base product shape.
- OpenClaude is the closest reference for command registration, tool contracts, provider abstraction, and shell discipline.

## Non-Goals

- A general-purpose coding-agent shell.
- A plugin or MCP platform.
- Uncontrolled autonomous expansion of the corpus.
- Replacing raw sources with opaque generated summaries.
- Reimplementing Microsoft GraphRAG from scratch.
- A general debate engine, persistent agent personas, or multi-round agent chat.
- Storing free-form reasoning traces as canonical artifacts; store structured outputs instead.
