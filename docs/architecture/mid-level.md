# Mid-Level Architecture

## Package Map

| Package | Responsibility |
| --- | --- |
| `src/cli.py` | CLI entrypoint and application bootstrap |
| `src/commands/` | Thin user-facing command wrappers |
| `src/services/` | Deterministic normalization, ingest, compile, lint, search, query, export, status, config, manifest, and prompt-toolkit terminal-workspace orchestration |
| `src/models/` | Shared command, source, tool, and wiki dataclasses |
| `src/engine/` | Command and tool registry boundaries |
| `src/providers/` | Future provider abstraction layer |

## Command To Service Mapping

| Command Wrapper | Main Service |
| --- | --- |
| `src/commands/init.py` | `src/services/project_service.py` and `src/services/config_service.py` |
| `src/commands/ingest.py` | `src/services/ingest_service.py`, `src/services/normalization_service.py`, and `src/services/manifest_service.py` |
| `src/commands/compile.py` | `src/services/compile_service.py` |
| `src/commands/search.py` | `src/services/search_service.py` |
| `src/commands/query.py` | `src/services/query_service.py` |
| `src/commands/lint.py` | `src/services/lint_service.py` |
| `src/commands/status.py` | `src/services/status_service.py` |
| `src/commands/export_vault.py` | `src/services/export_service.py` |
| `src/commands/tui.py` | `src/services/tui_service.py` |

## Data Flow

| Stage | Input | Output |
| --- | --- | --- |
| Ingest | canonical markdown/plain-text files and a bounded MarkItDown-backed converter subset | raw source copy, normalized artifact, and manifest metadata |
| Compile | normalized canonical text plus manifest metadata | source pages, concept pages, wiki index, compile log |
| Search | compiled wiki artifacts | ranked matches |
| Query | user question plus compiled context | cited answer based on maintained wiki |
| Lint | compiled wiki and metadata | structural findings and maintenance signals |
| Export | compiled wiki | Obsidian-friendly vault view |
| Terminal workspace | user input plus existing services | full-screen terminal session with panes, history, and repeated commands/questions |

## Current Ingest Scope

- The current implementation ingests `.md`, `.markdown`, and `.txt` files directly and also supports a bounded MarkItDown-backed subset of formats such as HTML, CSV, Office documents, notebooks, EPUB, and PDFs.
- OCR-backed ingest is still deferred and should arrive as a provider-backed fallback, with Mistral OCR as the current preferred OCR path for scanned or image-heavy inputs.

## Structural Rules

- Commands should stay thin and delegate quickly.
- Services should remain deterministic unless the feature explicitly requires model-backed synthesis.
- `kb tui` should remain a thin orchestration layer over existing services instead of duplicating command business logic.
- Static preview rendering and the full-screen terminal session should share the same underlying pane state where practical.
- Raw sources remain the source of truth; compiled pages are derived artifacts.
- Compile should prefer the normalized canonical artifact when one exists rather than reparsing the original raw source.
- Optional LLM-based cleanup or reconstruction should remain an explicit provider-mediated step instead of a silent default ingest behavior.
- Query behavior should prefer the compiled wiki over direct raw-file prompting.
- Evaluation features should remain clearly separated from the core CLI workflow.
