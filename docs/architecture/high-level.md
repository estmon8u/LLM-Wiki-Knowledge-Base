# High-Level Architecture

## Purpose

The project is pivoting into a CLI-first GraphRAG research-memory system with inspectable wiki artifacts. The target product ingests technical documents, preserves provenance, builds a graph-based retrieval index, answers local and global research questions, and exports human-readable markdown artifacts for inspection and maintenance.

The wiki is not the retrieval engine. The wiki is the human-readable artifact layer. GraphRAG is the retrieval and synthesis engine.

The architecture accepts heterogeneous source documents through a normalization step into canonical markdown or plain text. The current implementation routes canonical markdown and plain text directly, uses Mistral OCR as the default converter for explicitly supported native formats (`.pdf`, `.docx`, `.pptx`, `.png`, `.jpg`, `.jpeg`, `.avif`), and renders `.html` / `.htm` through `wkhtmltopdf` before sending the PDF bytes to the same OCR path. MarkItDown remains the default for the remaining bounded born-digital subset, and Docling or MarkItDown are used only as explicit fallbacks when the primary route fails quality checks.

The product goal is not to act like a general-purpose coding agent. The goal is to ingest source material, compile inspectable wiki artifacts, answer questions through a graph-first retrieval and synthesis path, keep any old FTS5 path explicit and deprecated, and export an Obsidian-friendly vault.

## Core User Flow

1. Initialize a project workspace.
2. Ingest supported source documents, store the originals, and normalize them into canonical markdown or plain text in the raw layer.
3. Compile source pages and refresh the deprecated legacy FTS index used only by `kb legacy ...` commands.
4. Sync normalized artifacts into the initialized GraphRAG workspace with source metadata.
5. Build a graph index for graph-first retrieval modes.
6. Ask graph-grounded questions by default; access the old FTS path only through explicit `kb legacy ...` commands.
7. Optionally save useful answers back into the wiki as persistent analysis pages.
8. Lint the maintained knowledge base for broken structure or stale content (deterministic).
9. Review the maintained knowledge base for contradictions, terminology drift, and topic overlap (semantic; requires a configured provider and combines deterministic overlap checks with provider-backed review).
10. Export the wiki into an Obsidian-friendly vault.

## Data Domains

- `raw/` stores source-of-truth input files, normalized canonical artifacts, and manifest metadata.
- `wiki/` stores generated source pages, saved analysis pages, index data, and compile logs.
- `vault/` stores export-ready Obsidian-friendly markdown.
- `graph/` stores the SQLite FTS5 search index and compile-run state under `graph/exports/`, plus the initialized Microsoft GraphRAG workspace under `graph/graphrag/` and generated GraphRAG input under `graph/graphrag/input/sources.json`.

## System Boundaries

- Commands expose user-facing CLI behavior.
- Services own deterministic business logic.
- Models hold shared dataclasses and typed results.
- Engine modules register commands.
- Providers abstract model-backed behavior behind a small boundary with concrete implementations for OpenAI, Anthropic, and Google Gemini; compile, ask, and review require a configured provider, while the rest of the CLI remains deterministic. Provider responses include model/provider diagnostics when SDKs expose them, provider selection, built-in provider settings, and conversion defaults now live together in `kb.config.yaml`, and a single provider instance is shared across all services via `build_provider(config)`. A shared Tenacity retry policy (`src/providers/retry.py`) wraps all `generate()` calls with exponential backoff and jitter for transient failures (rate limits, timeouts, server errors).
- CLI output uses Rich for styled tables, progress bars, and colored terminal output. All user-facing content is markup-escaped. The `NO_COLOR` environment variable and non-TTY detection are respected automatically, and stdout/stderr are configured to replace unsupported terminal characters instead of crashing on Windows code pages. Machine-readable `--json` flags are available on `doctor`, `find`, `status`, and `sources list`.
- Search storage currently persists a rebuildable SQLite FTS5 chunk index at `graph/exports/search_index.sqlite3` so lexical retrieval no longer scans every markdown file on each query. In the GraphRAG pivot, this index is temporary legacy infrastructure, not a peer default, and it is reachable only through explicit `kb legacy find` / `kb legacy ask` commands with deprecation warnings. GraphRAG should fail with clear next-step guidance when its workspace or index is missing instead of silently falling back to FTS5.
- The GraphRAG workspace is initialized under `graph/graphrag/` with tracked settings, prompts, and input scaffolding. `kb graph sync` now syncs normalized source artifacts and manifest metadata into `graph/graphrag/input/sources.json`, configures JSON input, and prepends provenance metadata into chunks for later GraphRAG indexing. The next layer will delegate graph indexing and query modes to Microsoft GraphRAG and export selected graph artifacts back into the wiki layer for inspection. Local `.env`, generated input, output, cache, and log artifacts are ignored.
- `kb.schema.md` is the wiki's operational constitution. Relevant schema sections are injected into compile and ask prompts via `schema_excerpt()`, so the LLM follows wiki-maintenance rules.
- Markdown and frontmatter parsing are centralized in `src/services/markdown_document.py` using `markdown-it-py` and `python-frontmatter`; services consume parser-backed helpers instead of maintaining parallel regex/state-machine implementations.
- `kb.config.yaml` validation is Pydantic-backed while retaining user-facing validation messages for existing workflows.
- Provider requests can carry an optional response schema, per-operation reasoning effort, and per-operation output budget. Concept generation, review findings, legacy ask answers, and compile summaries use structured provider outputs for semantic tasks where the model response drives stored wiki state. A shared structured-output parser accepts direct JSON, fenced JSON, and common prose-prefaced JSON before schema validation; Gemini receives a schema subset compatible with its SDK. `kb legacy ask` then rejects semantically empty answers, missing/ungrounded claims, and unknown citation refs, while review rejects malformed or empty provider JSON instead of parsing legacy pipe-delimited lines and filters excerpt-only truncation claims.
- Concept clustering is provider-first during `kb update`: the provider returns structured concept clusters over source-page titles/summaries, results are cached by source-page digest, and the deterministic NLTK/collocation pipeline remains a fallback.
- `kb legacy ask --save` persists structured answer metadata (`insufficient_evidence`, claim count, citation count, structured claims, and provider status) so saved analysis pages can be linted without text heuristics, and refuses to write blank analysis pages. Saved analysis pages remain searchable through `kb legacy find`, but legacy ask excludes them from retrieval evidence to avoid recursively citing prior generated answers; it also excludes generated concept pages so provider answers stay grounded in primary compiled source pages. Repeated saves get unique log headings so `wiki/log.md` remains lint-clean.
- Conversion is config-driven rather than hard-coded by suffix alone. Mistral OCR is the default path for the explicitly supported native document and image formats, HTML uses a rendered-PDF OCR route, and converter quality gates prevent partial or obviously truncated artifacts from becoming canonical markdown.
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
