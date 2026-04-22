# High-Level Architecture

## Purpose

The project is a CLI-first workflow for building and maintaining a persistent markdown knowledge base from a curated source corpus.

The architecture accepts heterogeneous source documents through a normalization step into canonical markdown or plain text. The current implementation now routes canonical markdown and plain text directly, uses Docling for PDFs, and uses MarkItDown for the remaining bounded subset of born-digital formats. OCR-style image extraction and optional LLM-assisted reconstruction remain out of the default ingest path for now and are reserved for a later provider-backed fallback, with Mistral OCR as the current leading OCR candidate.

The product goal is not to act like a general-purpose coding agent. The goal is to ingest source material, compile a reusable wiki, answer questions from the compiled wiki with traceability, and export an Obsidian-friendly vault.

## Core User Flow

1. Initialize a project workspace.
2. Ingest supported source documents, store the originals, and normalize them into canonical markdown or plain text in the raw layer.
3. Compile source pages and refresh indexes into the wiki layer.
4. Search and query the compiled wiki instead of querying raw files directly.
5. Optionally save useful query answers back into the wiki as persistent analysis pages.
6. Lint the maintained knowledge base for broken structure or stale content (deterministic).
7. Review the maintained knowledge base for contradictions, terminology drift, and topic overlap (semantic; requires a configured provider and combines deterministic overlap checks with provider-backed review).
8. Export the wiki into an Obsidian-friendly vault.

## Data Domains

- `raw/` stores source-of-truth input files, normalized canonical artifacts, and manifest metadata.
- `wiki/` stores generated source pages, saved analysis pages, index data, and compile logs.
- `vault/` stores export-ready Obsidian-friendly markdown.
- `graph/` stores the SQLite FTS5 search index and compile-run state under `graph/exports/`.

## System Boundaries

- Commands expose user-facing CLI behavior.
- Services own deterministic business logic.
- Models hold shared dataclasses and typed results.
- Engine modules register commands.
- Providers abstract model-backed behavior behind a small boundary with concrete implementations for OpenAI, Anthropic, and Google Gemini; compile, query, and review require a configured provider, while the rest of the CLI remains deterministic. Provider selection and built-in provider settings now live together in `kb.config.yaml`, and a single provider instance is shared across all services via `build_provider(config)`. A shared Tenacity retry policy (`src/providers/retry.py`) wraps all `generate()` calls with exponential backoff and jitter for transient failures (rate limits, timeouts, server errors).
- CLI output uses Rich for styled tables, progress bars, and colored terminal output. All user-facing content is markup-escaped. The `NO_COLOR` environment variable and non-TTY detection are respected automatically. Machine-readable `--json` flags are available on `doctor`, `find`, `status`, and `sources list`.
- Search storage persists a rebuildable SQLite FTS5 chunk index at `graph/exports/search_index.sqlite3` so lexical retrieval no longer scans every markdown file on each query.
- `kb.schema.md` is the wiki's operational constitution. Relevant schema sections are injected into compile and query prompts via `schema_excerpt()`, so the LLM follows wiki-maintenance rules.
- Provider-backed OCR or LLM cleanup should remain explicit fallback behavior rather than becoming part of the default deterministic normalization path.

## Reference-Project Roles

- Browzy.ai is the closest reference for the knowledge-base product shape.
- OpenClaude is the closest reference for command registration, tool contracts, provider abstraction, and shell discipline.

## Non-Goals

- A general-purpose coding-agent shell.
- A plugin or MCP platform.
- Uncontrolled autonomous expansion of the corpus.
- Replacing raw sources with opaque generated summaries.
- A general debate engine, persistent agent personas, or multi-round agent chat.
- Storing free-form reasoning traces as canonical artifacts; store structured outputs instead.
