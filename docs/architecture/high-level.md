# High-Level Architecture

## Purpose

The project is a CLI-first workflow for building and maintaining a persistent markdown knowledge base from a curated source corpus.

It now supports both one-shot commands and a prompt-toolkit full-screen terminal workspace through `kb tui`, while still staying inside the same command/service architecture.

The architecture accepts heterogeneous source documents through a normalization step into canonical markdown or plain text. The current implementation now routes canonical markdown and plain text directly, uses Docling for PDFs, and uses MarkItDown for the remaining bounded subset of born-digital formats. OCR-style image extraction and optional LLM-assisted reconstruction remain out of the default ingest path for now and are reserved for a later provider-backed fallback, with Mistral OCR as the current leading OCR candidate.

The product goal is not to act like a general-purpose coding agent. The goal is to ingest source material, compile a reusable wiki, answer questions from the compiled wiki with traceability, and export an Obsidian-friendly vault.

## Core User Flow

1. Initialize a project workspace.
2. Ingest supported source documents, store the originals, and normalize them into canonical markdown or plain text in the raw layer.
3. Compile source pages, concept pages, and indexes into the wiki layer.
4. Search and query the compiled wiki instead of querying raw files directly.
5. Lint the maintained knowledge base for broken structure or stale content.
6. Export the wiki into an Obsidian-friendly vault.
7. Optionally work through repeated maintenance and question-answering tasks inside the terminal workspace.
8. Optionally compare the maintained wiki against simpler RAG or graph-style baselines.

## Data Domains

- `raw/` stores source-of-truth input files, normalized canonical artifacts, and manifest metadata.
- `wiki/` stores generated source pages, index data, and compile logs.
- `vault/` stores export-ready Obsidian-friendly markdown.
- `graph/` is optional and should stay evaluation-oriented rather than becoming the primary storage layer.

## System Boundaries

- Commands expose user-facing CLI behavior.
- Services own deterministic business logic.
- The terminal workspace is a UI layer over the same services rather than a separate execution path.
- Snapshot previews and full-screen interaction should reflect the same pane state and command results.
- Models hold shared dataclasses and typed results.
- Engine modules register commands and tools.
- Providers abstract future model-backed behavior behind a small boundary.
- Provider-backed OCR or LLM cleanup should remain explicit fallback behavior rather than becoming part of the default deterministic normalization path.

## Reference-Project Roles

- Browzy.ai is the closest reference for the knowledge-base product shape.
- OpenClaude is the closest reference for command registration, tool contracts, provider abstraction, and shell discipline.

## Non-Goals

- A general-purpose coding-agent shell.
- A plugin or MCP platform.
- Uncontrolled autonomous expansion of the corpus.
- Replacing raw sources with opaque generated summaries.
