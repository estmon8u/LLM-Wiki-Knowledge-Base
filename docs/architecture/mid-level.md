# Mid-Level Architecture

## Package Map

| Package | Responsibility |
| --- | --- |
| `src/cli.py` | CLI entrypoint and application bootstrap |
| `src/commands/` | Thin user-facing command wrappers |
| `src/services/` | Deterministic ingest, compile, lint, search, query, export, status, config, manifest, and terminal-workspace orchestration |
| `src/models/` | Shared command, source, tool, and wiki dataclasses |
| `src/engine/` | Command and tool registry boundaries |
| `src/providers/` | Future provider abstraction layer |

## Command To Service Mapping

| Command Wrapper | Main Service |
| --- | --- |
| `src/commands/init.py` | `src/services/project_service.py` and `src/services/config_service.py` |
| `src/commands/ingest.py` | `src/services/ingest_service.py` and `src/services/manifest_service.py` |
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
| Ingest | currently canonical markdown/plain-text files, later converter outputs from other document types | raw source copy plus manifest metadata |
| Compile | raw sources and manifest | source pages, concept pages, wiki index, compile log |
| Search | compiled wiki artifacts | ranked matches |
| Query | user question plus compiled context | cited answer based on maintained wiki |
| Lint | compiled wiki and metadata | structural findings and maintenance signals |
| Export | compiled wiki | Obsidian-friendly vault view |
| Terminal workspace | user input plus existing services | persistent terminal session for repeated commands and questions |

## Current Ingest Scope

- The scaffold currently ingests `.md`, `.markdown`, and `.txt` files directly.
- Heterogeneous document support is still expected to arrive through converter-backed normalization before `IngestService`.

## Structural Rules

- Commands should stay thin and delegate quickly.
- Services should remain deterministic unless the feature explicitly requires model-backed synthesis.
- `kb tui` should remain a thin orchestration layer over existing services instead of duplicating command business logic.
- Raw sources remain the source of truth; compiled pages are derived artifacts.
- Query behavior should prefer the compiled wiki over direct raw-file prompting.
- Evaluation features should remain clearly separated from the core CLI workflow.
