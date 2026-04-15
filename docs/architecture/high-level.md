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
8. Optionally fix lint and review issues via `kb fix`: deterministic fixes apply directly, LLM-backed fixes show a diff for user approval.
9. Export the wiki into an Obsidian-friendly vault.
10. Optionally compare the maintained wiki against simpler RAG or graph-style baselines.

## Data Domains

- `raw/` stores source-of-truth input files, normalized canonical artifacts, and manifest metadata.
- `wiki/` stores generated source pages, saved analysis pages, index data, and compile logs.
- `vault/` stores export-ready Obsidian-friendly markdown.
- `graph/` is optional and should stay evaluation-oriented rather than becoming the primary storage layer.

## System Boundaries

- Commands expose user-facing CLI behavior.
- Services own deterministic business logic.
- Models hold shared dataclasses and typed results.
- Schemas define Pydantic models (`Claim`, `EvidenceBundle`, `CandidateAnswer`, `MergedAnswer`, `ReviewFinding`, `RunRecord`) shared across query, review, and concept synthesis.
- Engine modules register commands and tools.
- Providers abstract model-backed behavior behind a small boundary with concrete implementations for OpenAI, Anthropic, and Google Gemini; compile, query, and review now require a configured provider, while the rest of the CLI remains deterministic.
- A bounded deliberation layer now sits between the service layer and the provider boundary for opt-in multi-sample workflows. `kb query --self-consistency N` freezes retrieved evidence, fans out parallel provider calls, normalizes sentence-level claims, and merges grounded claims deterministically. `kb review --adversarial` now builds candidate page pairs, runs extractor/skeptic/arbiter prompts, and emits typed review findings. `kb fix --propose` remains planned.
- Run-artifact storage persists every deliberation run in SQLite (`RunStore`) at `graph/exports/run_artifacts.sqlite3`: retrieval sets, candidate outputs, review findings, merge decisions, model id, prompt version, context hash, token cost, wall time, and unresolved-disagreement flag.
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
