# GraphWiki KB

A CLI-first GraphRAG research-memory system for ingesting technical documents, building a graph-based retrieval index, answering local/global research questions, and exporting inspectable wiki artifacts with provenance and citations.

The wiki is not the retrieval engine. The wiki is the human-readable artifact layer. GraphRAG is the retrieval and synthesis engine.

This branch is in the GraphRAG pivot. GraphRAG is the target default retrieval and synthesis path. The existing SQLite FTS5 search and source-grounded ask workflow is now explicit legacy behavior under `kb legacy find` and `kb legacy ask` with deprecation warnings. Top-level `kb ask` is a GraphRAG-aware answer controller that checks graph readiness, chooses a query mode with deterministic routing by default, calls GraphRAG, and can save analysis pages with graph metadata. Top-level `kb find` remains reserved for a later graph search controller. `kb graph ask --method local|global|drift|basic` stays available for explicit GraphRAG query-mode control, and `kb graph export-wiki` exports GraphRAG output tables back into inspectable markdown under `wiki/graph/`. See [docs/graphrag-pivot.md](docs/graphrag-pivot.md) for the pivot rationale and target architecture.

## Requirements

- Python 3.11.x
- [Poetry](https://python-poetry.org/) (installed at user level, not inside the project virtualenv)

## Installation

```bash
cd LLM-Wiki-Knowledge-Base
poetry install
```

This creates a local `.venv` and installs all dependencies. The CLI entrypoint is registered as `kb`.

## GraphRAG Workspace

Microsoft GraphRAG is installed as a library/CLI dependency, not a separate paid hosted service. Running real GraphRAG indexing or query jobs can still create model and embedding costs through the configured provider.

The repository contains an initialized GraphRAG workspace under `graph/graphrag/`. The committed scaffold includes `settings.yaml`, default prompts, and `input/`; local runtime files such as `.env`, generated `input/sources.json`, `output/`, `cache/`, `logs/`, and run metadata under `graph/runs/*.json` stay ignored.

```bash
poetry run graphrag --help
poetry run kb graph init
poetry run kb graph init --model gpt-4.1-mini --embedding text-embedding-3-small
```

GraphRAG runtime defaults live in the `graph` section of `kb.config.yaml`. `kb graph init` reads that config and syncs the selected completion provider, embedding provider, models, and resolved API-key environment variables into `graph/graphrag/settings.yaml`; `--model` and `--embedding` are per-run model overrides. By default, GraphRAG reuses the OpenAI provider entry and references `OPENAI_API_KEY`, so a separate `GRAPHRAG_API_KEY` is not required unless you explicitly set a graph-specific override.

`kb graph sync` uses the existing knowledge-base corpus as the source of truth. It reads `raw/_manifest.json` plus the normalized artifacts in `raw/normalized/`, writes `graph/graphrag/input/sources.json`, and configures GraphRAG for JSON input with metadata prepended into chunks.

`kb graph index` wraps `python -m graphrag index` with explicit method choices (`standard`, `fast`, `standard-update`, `fast-update`) and supports `--dry-run` before a real index job. `kb graph status` checks whether settings, synced input, GraphRAG output tables, and the last recorded index run are present. `kb graph ask` wraps `python -m graphrag query` with explicit Local, Global, DRIFT, and Basic modes and can save raw GraphRAG answers as analysis pages. `kb graph export-wiki` reads GraphRAG Parquet tables and writes human-readable documents, entities, relationships, communities, and text units under `wiki/graph/`.

## Evaluation Harness

Phase 8 adds an evaluation harness for comparing the deprecated legacy FTS path against GraphRAG Basic, Local, Global, and DRIFT query modes. The benchmark lives in `eval/benchmark.yaml`; generated reports are written under `eval/results/`.

```bash
# Local-safe baseline: legacy find + auto-router fit, provider-backed rows skipped
poetry run python scripts/evaluate_graph_modes.py

# Retrieval-only CSV
poetry run python scripts/evaluate_retrieval.py

# Provider-backed answer comparison when the graph provider/API key is ready
poetry run python scripts/evaluate_graph_modes.py --allow-provider-calls --include-legacy-ask
```

The default run does not call model providers. It records skipped rows for GraphRAG and legacy answer commands unless `--allow-provider-calls` is passed, because those jobs can incur provider costs and may include local corpus text in generated artifacts. Per-question JSON artifacts are written under `eval/results/artifacts/`, which is ignored by Git.

## Current Transitional Quick Start

```bash
# 1. Initialize a new project
poetry run kb init

# 2. Add some source documents
poetry run kb add path/to/paper.pdf
poetry run kb add path/to/notes.md
poetry run kb add path/to/slides.pptx
poetry run kb add path/to/research-folder

# 3. Update the knowledge base (generates wiki artifacts and the temporary legacy FTS index)
poetry run kb update

# 4. Initialize or refresh the GraphRAG workspace settings
poetry run kb graph init

# 5. Sync normalized artifacts into GraphRAG input
poetry run kb graph sync

# 6. Check graph readiness before running a provider-backed index
poetry run kb graph status
poetry run kb graph index --method fast --dry-run

# 7. After a real index, ask through the default GraphRAG controller
poetry run kb ask "What are the main retrieval patterns?"
poetry run kb ask --method global "What are the main retrieval patterns?"

# 8. Use an explicit GraphRAG mode when debugging retrieval behavior
poetry run kb graph ask "What are the main retrieval patterns?" --method global

# 9. Export GraphRAG artifacts into wiki/graph
poetry run kb graph export-wiki

# 10. Search the deprecated legacy/wiki path
poetry run kb legacy find "knowledge base traceability"

# 11. Ask through the deprecated legacy source-grounded path
poetry run kb legacy ask "How does the wiki handle stale pages?"
```

That's the current GraphRAG-first workflow: **add -> update -> graph init -> graph sync -> graph status/index -> kb ask**.
The legacy commands exist only for comparison and exact lexical lookup while
later GraphRAG phases add broader graph health checks.
There is no silent fallback from GraphRAG to FTS5.

For a slower first-run walkthrough that keeps the repository and knowledge-base
project in separate directories, see [docs/start-guide.md](docs/start-guide.md).

Before running `kb update`, `kb legacy ask`, or `kb review`, configure the active
provider in `kb.config.yaml` and set the matching API key environment
variable. If you ingest `.pdf`, `.docx`, `.pptx`, supported images, or
`.html` / `.htm`, also set `MISTRAL_API_KEY`. HTML inputs additionally require
`wkhtmltopdf` on your `PATH` or configured in `kb.config.yaml`. See
`Provider Configuration` and `Conversion Configuration` below.

## Global Options

| Option | Description |
| --- | --- |
| `--project-root PATH` | Run the CLI against a specific project directory instead of the current working directory. |
| `--provider NAME` | Override the configured provider (`openai`, `anthropic`, `gemini`) for this invocation. Clears any stale model setting. |
| `--verbose` | Enable verbose output. |
| `-h`, `--help` | Show help for any command. |

Example:

```bash
poetry run kb --project-root /path/to/project status
```

## Commands

### Everyday Commands

These are the commands you will use most often today. The GraphRAG-first happy
path is **init -> add -> update -> graph init -> graph sync -> graph index -> ask**.

| Command | Description |
| --- | --- |
| `init` | Create project folders, config, schema, and manifest |
| `add` | Add and normalize source documents |
| `update` | Build wiki pages, generate concepts, and refresh indexes |
| `find` | Reserved GraphRAG search entry point; currently fails with guidance |
| `ask` | GraphRAG-aware answer controller with deterministic auto-routing |
| `graph` | GraphRAG workspace commands for init, input sync, indexing, explicit query modes, graph wiki export, and status |
| `legacy` | Deprecated SQLite FTS5 search and ask commands for comparison |
| `status` | Show project state and what to do next |

### Advanced Commands

Use these for maintenance, quality checks, and deeper analysis.

| Command | Description |
| --- | --- |
| `lint` | Deterministic structural checks on the wiki |
| `review` | Semantic review for contradictions and terminology drift |
| `export` | Export the wiki to an Obsidian vault |
| `doctor` | Validate structure, provider, API keys, and converters |
| `config` | View and manage project configuration |
| `sources` | Manage source inventory |

### `kb init`

Create the project folders, config, schema, and manifest files.

```bash
poetry run kb init
```

Creates:

- `kb.config.yaml` — project configuration
- `kb.schema.md` — compilation schema for domain-specific rules
- `raw/_manifest.json` — source manifest
- `raw/sources/` — directory for original source files
- `raw/normalized/` — directory for canonical markdown/text artifacts
- `wiki/sources/` — directory for compiled source pages
- `vault/obsidian/` — directory for vault export

Running `init` again is safe — it skips files that already exist.
The scaffold writes `kb.config.yaml` at config version 6. Older configs are migrated in place on load so deprecated fields are removed and newer sections such as `providers`, `conversion`, `graph`, and `storage.raw_normalized_dir` are persisted automatically.

### `kb doctor`

Run health checks on the project: structure, config, provider selection,
provider API keys, Mistral OCR readiness, HTML renderer availability, and
converter dependencies.

```bash
poetry run kb doctor
poetry run kb doctor --strict
```

Prints formatted health-check sections with `[OK]`, `[WARNING]`, or `[FAIL]` entries. Exits with code 1 if any check fails. Without `--strict`, a missing provider or API key is a warning rather than a failure, so new projects pass doctor out of the box.

| Option | Default | Description |
| --- | --- | --- |
| `--strict` | off | Treat warnings (missing provider, API key) as errors. |
| `--json` | off | Output results as JSON for scripting. |

### `kb add <source_path>`

Add and normalize a source file into the raw corpus, or recursively add a directory of supported source files.

```bash
poetry run kb add path/to/document.pdf
poetry run kb add path/to/research-folder
```

What happens:

1. The original file is copied into `raw/sources/`.
2. A canonical markdown or plain-text artifact is created in `raw/normalized/` (for non-text formats).
3. The source is registered in `raw/_manifest.json` with metadata including the content hash, converter used, and timestamps.

Duplicate detection: if you add the same file again, it will be detected and skipped.

When the input is a directory, the command walks the directory tree recursively by default, adds only supported file types, and prints a batch summary showing how many files were created or skipped as duplicates. Unsupported files inside the directory are ignored.
Interactive terminals show directory-ingest progress; non-interactive runs print a simple `Ingesting N source file(s)...` preamble before the summary.

### `kb update [SOURCE_PATHS...]`

Bring the knowledge base current. Optionally add new sources first, then build wiki pages, generate concepts, and refresh indexes.

```bash
poetry run kb update
poetry run kb update path/to/new-paper.pdf
poetry run kb update --force
poetry run kb update --resume
```

| Option | Default | Description |
| --- | --- | --- |
| `--force` | off | Rebuild every source page even if nothing changed. |
| `--resume` | off | Resume the most recent failed or interrupted update run. Cannot be combined with `--force`. |

When source paths are provided, the command adds them first. Always generates concept pages and refreshes the search index after building.
Compile summaries request structured provider output (`summary`, `key_points`, `open_questions`, and `title_suggestion`) and persist the extra fields when returned. Concept clustering uses the configured provider when available, parses direct, fenced, or prefaced JSON through the shared structured-output parser, caches valid cluster output by source-page digest, and falls back to deterministic collocation-based grouping if provider clustering fails.

### `kb graph init`

Initialize or refresh the local GraphRAG workspace settings.

```bash
poetry run kb graph init
poetry run kb graph init --model gpt-4.1-mini --embedding text-embedding-3-small
poetry run kb graph init --json
```

The wrapper delegates to the official GraphRAG CLI through `python -m graphrag init` and writes settings under `graph/graphrag/`. It reads `graph.provider`, `graph.model`, `graph.embedding_provider`, `graph.embedding_model`, and optional graph-specific API-key overrides from `kb.config.yaml`, then resolves API-key environment variables from the centralized `providers` catalog and syncs those values into `graph/graphrag/settings.yaml`. `--model` and `--embedding` override the config for one init run. The command still uses `--force` by default so project setup is reproducible and non-interactive.

### `kb graph sync`

Sync normalized source artifacts into the initialized GraphRAG workspace.

```bash
poetry run kb graph sync
poetry run kb graph sync --json
```

The command reads `raw/_manifest.json` and each source's `raw/normalized/`
artifact, then writes `graph/graphrag/input/sources.json` as JSON records with
`id`, `title`, `text`, and provenance fields such as `source_id`, `slug`,
`source_hash`, `raw_path`, `normalized_path`, `converter`, and
`normalization_route`. It also configures `graph/graphrag/settings.yaml` for
GraphRAG JSON input and metadata prepending through `chunking.prepend_metadata`.
Generated `sources.json` can contain local corpus text and is ignored by Git.

### `kb graph index`

Run the official GraphRAG indexer against the synced JSON input.

```bash
poetry run kb graph index --method fast --dry-run
poetry run kb graph index --method fast
poetry run kb graph index --method standard-update --json
```

| Option | Default | Description |
| --- | --- | --- |
| `--method` | `fast` | GraphRAG index method: `standard`, `fast`, `standard-update`, or `fast-update`. |
| `--dry-run` | off | Ask GraphRAG to validate the index command without running a full job. |
| `--cache` / `--no-cache` | `--cache` | Forward GraphRAG cache behavior. |
| `--skip-validation` | off | Forward GraphRAG's validation skip flag. |
| `--verbose` | off | Forward GraphRAG verbose output. |
| `--json` | off | Include command output and recorded run metadata as JSON. |

The command requires an initialized workspace and a non-empty synced `sources.json`. A real non-dry-run index can incur model and embedding costs through the configured GraphRAG provider.

### `kb graph status`

Report GraphRAG workspace and index readiness.

```bash
poetry run kb graph status
poetry run kb graph status --json
```

Status checks whether the workspace is initialized, synced input exists, input records are present, GraphRAG output Parquet tables are present for documents, text units, entities, relationships, communities, and community reports, and whether a previous `kb graph index` run was recorded.

### `kb graph ask <question>`

Ask a question through an explicit GraphRAG query mode.

```bash
poetry run kb graph ask "How does REALM differ from RAG?" --method local
poetry run kb graph ask "What are the main retrieval design patterns?" --method global
poetry run kb graph ask "Compare RAG, REALM, FiD, Self-RAG, and GraphRAG." --method drift
poetry run kb graph ask "What is dense passage retrieval used for?" --method basic
poetry run kb graph ask "How does REALM differ from RAG?" --method drift --save
```

| Option | Default | Description |
| --- | --- | --- |
| `--method` | required | GraphRAG query method: `local`, `global`, `drift`, or `basic`. |
| `--community-level` | | Forward GraphRAG's community-level option. |
| `--dynamic-community-selection` / `--no-dynamic-selection` | GraphRAG default | Forward GraphRAG dynamic community selection behavior. |
| `--response-type` | GraphRAG default | Forward GraphRAG's response type option. |
| `--save` | off | Save the answer as a GraphRAG-backed analysis page under `wiki/analysis/`. |
| `--save-as` | | Save with a custom analysis slug. Implies `--save`. |
| `--json` | off | Include the answer, raw GraphRAG output, command, `retriever`, `method`, `index_run_id`, and `input_manifest_hash` as JSON. |

The command requires synced input and GraphRAG index output. Saved analysis pages use `type: analysis`, `retriever: graphrag`, `method`, `question`, `created_at`, `index_run_id`, and `input_manifest_hash` frontmatter, then store the answer, retrieval mode metadata, source trace, and raw GraphRAG CLI output. The explicit graph command preserves raw output for debugging and comparison. For the default user-facing answer path, use `kb ask`, which adds planner/method metadata and graph readiness checks on top of the same GraphRAG query service.

### `kb graph export-wiki`

Export GraphRAG Parquet output tables into inspectable markdown under `wiki/graph/`.

```bash
poetry run kb graph export-wiki
poetry run kb graph export-wiki --json
```

The export reads standard GraphRAG tables such as `documents`, `text_units`, `entities`, `relationships`, `communities`, and `community_reports` from `graph/graphrag/output/`. It writes:

- `wiki/graph/index.md`
- `wiki/graph/documents/*.md`
- `wiki/graph/entities/*.md`
- `wiki/graph/relationships/*.md`
- `wiki/graph/communities/*.md`
- `wiki/graph/text-units/*.md`

Generated graph pages use frontmatter types such as `graph_entity`, `graph_relationship`, `graph_community`, `graph_text_unit`, and `graph_document`. Existing `wiki/concepts/` pages are not deleted; they are now legacy LLM-wiki concept pages beside the GraphRAG-derived `wiki/graph/` layer.

### `kb find <terms>`

Reserved GraphRAG search entry point. Until a default GraphRAG search controller
lands, this command fails with guidance to use `kb legacy find` for deprecated
FTS5 lookup.

### `kb legacy find <terms>`

Search the deprecated SQLite FTS5 wiki index for relevant pages and snippets.

```bash
poetry run kb legacy find "traceability citation"
poetry run kb legacy find --limit 10 "agent architecture"
poetry run kb legacy find --json "REALM vs RAG"
```

| Option | Default | Description |
| --- | --- | --- |
| `--limit` | 5 | Maximum number of results to return. |
| `--json` | off | Output results as JSON for scripting. |

Uses a SQLite FTS5 chunk index stored at `graph/exports/search_index.sqlite3`.
This is temporary legacy behavior rather than the final retrieval engine.
The command searches source pages, generated concept pages, and saved analysis
pages, ranks hits with BM25-style FTS ordering, and returns page-level results
using the best matching chunk snippet. JSON output includes
`retriever: "legacy-fts"` and `deprecated: true` metadata. Evidence chunks skip
metadata-only sections such as `Source Details`, `Source Pages`,
`Related Concept Pages`, and `Citations` so retrieval and citations point at
content rather than wiki bookkeeping.

### `kb ask <question>`

Ask a question through the GraphRAG-aware answer controller.

```bash
poetry run kb ask "How does REALM differ from RAG?"
poetry run kb ask --method global "What are the main retrieval themes across the corpus?"
poetry run kb ask --method drift --save "Compare RAG, REALM, FiD, Self-RAG, and GraphRAG."
poetry run kb ask --show-evidence "How does the graph index support source traceability?"
```

| Option | Default | Description |
| --- | --- | --- |
| `--method` | `auto` | Use deterministic auto-routing or force `basic`, `local`, `global`, or `drift`. |
| `--community-level` | | Forward GraphRAG's community-level option. |
| `--dynamic-community-selection` / `--no-dynamic-selection` | GraphRAG default | Forward GraphRAG dynamic community selection behavior. |
| `--response-type` | GraphRAG default | Forward GraphRAG's response type option. |
| `--save` | off | Save the graph answer as an analysis page under `wiki/analysis/`. |
| `--save-as` | | Save with a custom analysis slug. Implies `--save`. |
| `--show-evidence` | off | Print source trace, route reason, and current claim-support status before the answer. |
| `--json` | off | Include GraphRAG answer metadata as JSON. |

The controller checks workspace, input, and index readiness before querying.
It does not silently fall back to FTS5. If the graph is missing or not ready, run
`kb graph init`, `kb graph sync`, and `kb graph index` as directed by the error.
Saved pages use `retriever: graph`, `method`, `planner`, `claim_support`,
`index_run_id`, and `input_manifest_hash` metadata.

### `kb legacy ask <question>`

Answer a question from compiled source-page evidence with provider-backed synthesis and citations through the deprecated SQLite FTS5 retrieval path.

```bash
poetry run kb legacy ask "How does the wiki handle stale pages?"
poetry run kb legacy ask --limit 5 "What normalization converters are supported?"
poetry run kb legacy ask --save "What does the update pipeline do?"
poetry run kb legacy ask --save-as freshness "How is freshness tracked?"
poetry run kb legacy ask --show-evidence "What formats are supported?"
```

| Option | Default | Description |
| --- | --- | --- |
| `--limit` | 3 | Maximum number of source pages to use as evidence. |
| `--save` | off | Save the answer as an analysis page in the wiki. |
| `--save-as` | | Save the answer as an analysis page with a custom slug. |
| `--show-evidence` | off | Print the retrieved evidence snippets before the answer. |

Requires a configured provider. Retrieves the best-matching source-page chunks as evidence, excluding saved analysis pages and generated concept pages so answers cite primary compiled sources. It packages the top chunk snippets into a frozen evidence bundle, asks the provider for structured output (`answer_markdown`, `claims`, `citations`, and `insufficient_evidence`), semantically validates the response, strips raw inline citation-ref markers from answer prose, and prints the answer followed by a Citations section.

Provider-backed answers must be both parseable and useful. Empty `answer_markdown`, missing claims when `insufficient_evidence` is false, claims without citation refs, and citation refs outside the retrieved evidence set fail the command instead of being treated as a successful answer. Provider failures include response diagnostics such as finish reason and token counts when the selected SDK exposes them.

Use `--save` or `--save-as` to persist the answer as a markdown analysis page in `wiki/analysis/` with YAML frontmatter (`type: analysis`), the question, a timestamp, `insufficient_evidence`, claim/citation counts, structured claims/provider citations when available, provider status diagnostics, and backlinks to cited source chunks. Blank answers are refused rather than saved. Saved analysis pages are indexed for `kb legacy find` and appear in `wiki/index.md` immediately, but later legacy ask runs exclude analysis pages from the evidence set so saved answers are not recursively cited as primary evidence. Repeated saves use unique `wiki/log.md` headings so lint does not treat a rerun as a duplicate-heading issue.

### `kb lint`

Run deterministic structural lint checks over the maintained wiki.

```bash
poetry run kb lint
```

Checks for:

- Broken internal links in both wiki-link and standard markdown-link form
- Missing heading fragments for links such as `[[Page#Section]]` and `[text](page.md#section)`
- Repeated source/concept page titles, repeated headings, skipped heading levels, and multiple H1 headings. Saved analysis pages may share a question title when comparing provider runs.
- Missing frontmatter or provenance metadata on compiled pages
- Typed frontmatter validation: string, date (ISO format), and list fields
- Empty compiled pages that have no body content beyond headings
- Source pages missing `type` frontmatter field (warning; run `kb update --force` to refresh)
- Orphan pages with no inbound wiki or markdown links
- Stale compiled pages whose source hash changed
- Other structural issues

Exits with code 1 if any errors are found. Warnings and suggestions are printed but don't cause a nonzero exit.

### `kb status`

Show what exists, what changed, what is stale, and what to do next.

```bash
poetry run kb status
poetry run kb status --changed
```

| Option | Default | Description |
| --- | --- | --- |
| `--changed` | off | Show a preview of new, changed, and up-to-date sources. |
| `--json` | off | Output results as JSON for scripting. |

Default view shows a Knowledge Base overview with source and wiki counts, the last update timestamp, and a suggestion for what to do next. With `--changed`, shows each source with a status tag (`[NEW]`, `[CHANGED]`, `[OK]`) followed by a summary section with counts.

### `kb review`

Run semantic review checks for contradictions and terminology drift across the maintained wiki.

```bash
poetry run kb review
```

Runs deterministic overlap and terminology-variant checks plus a single-pass model-backed review. Requires a configured provider.

Checks for:

- **Overlapping topics** — Source pages with heavily overlapping terminology that may benefit from a shared concept page.
- **Terminology variants** — The same root term appearing in different forms across pages.

This is the semantic complement to `kb lint`. Lint checks structural health deterministically; review checks content-level coherence through heuristics and a provider pass.
Provider-backed review requests structured JSON when the selected SDK supports schema hints and rejects malformed or empty provider review output instead of parsing legacy pipe-delimited lines. The deterministic pass can inspect reviewable source/concept pages, while the provider pass reviews curated source-page excerpts rather than maintenance metadata. Excerpt-boundary truncation claims are filtered unless they are backed by actual page content. Deterministic terminology-variant checks suppress simple inflections, specificity-only pairs, and obvious negating-prefix variants before raising drift suggestions.

### `kb export`

Export the compiled wiki to the configured target (defaults to Obsidian vault).

```bash
poetry run kb export
poetry run kb export --clean
```

| Option | Default | Description |
| --- | --- | --- |
| `--clean` | off | Remove stale vault files that no longer exist in the wiki. |

Copies compiled wiki pages into `vault/obsidian/` in a format compatible with [Obsidian](https://obsidian.md/). With `--clean`, any markdown files in the vault that no longer correspond to a wiki page are deleted automatically.


### `kb config`

View project configuration as YAML, or manage the active provider selection.

```bash
poetry run kb config
poetry run kb config show
poetry run kb config provider set openai
poetry run kb config provider set anthropic --model claude-opus-4-6
poetry run kb config provider clear
```

| Subcommand | Description |
| --- | --- |
| `show` | Display the current configuration (default). |
| `provider set <name>` | Set the LLM provider. `--model` pins a specific model. Changing provider clears any stale model. |
| `provider clear` | Remove the LLM provider setting. |

### `kb sources`

Manage source inventory. With no subcommand, lists all ingested sources.

```bash
poetry run kb sources
poetry run kb sources list
poetry run kb sources show <slug>
```

| Subcommand | Description |
| --- | --- |
| `list` | List all ingested sources with status. |
| `show <slug>` | Show full details for a single source. |

## Supported File Types

| Format | Extension(s) | Converter |
| --- | --- | --- |
| Markdown | `.md`, `.markdown` | Direct (no conversion needed) |
| Plain text | `.txt` | Direct (no conversion needed) |
| PDF | `.pdf` | Mistral OCR (Docling fallback) |
| HTML | `.htm`, `.html` | `wkhtmltopdf` → Mistral OCR (MarkItDown fallback) |
| CSV | `.csv` | MarkItDown |
| Word | `.docx` | Mistral OCR (MarkItDown fallback) |
| PowerPoint | `.pptx` | Mistral OCR (MarkItDown fallback) |
| Excel | `.xls`, `.xlsx` | MarkItDown |
| EPUB | `.epub` | MarkItDown |
| Jupyter Notebook | `.ipynb` | MarkItDown |
| Images | `.png`, `.jpg`, `.jpeg`, `.avif` | Mistral OCR |

All non-text formats are normalized into canonical markdown and stored in `raw/normalized/` before compilation.

## Provider Configuration

`kb update`, `kb legacy ask`, and `kb review` require a configured provider.
If the provider is missing, those commands fail with a configuration error.
`kb legacy ask` and `kb review` fail on provider execution errors; `kb update` keeps
deterministic fallbacks for compile summaries and concept clustering after a
provider call failure.

Provider responses carry the returned text plus model/provider diagnostics such
as finish reason and token counts when the SDK exposes them. Saved `kb legacy ask`
analysis pages persist those parsed/validated diagnostics under `provider_status`
when the answer comes from a provider. Structured provider
outputs are parsed through the shared JSON parser, which accepts direct JSON,
fenced JSON, and common prose-prefaced JSON before schema and semantic
validation. Services can override provider reasoning effort and output budgets
per operation; schema-bound commands use lower reasoning settings and larger
visible-output budgets where needed. Gemini receives a provider-compatible JSON
schema subset with unsupported `additionalProperties` fields removed before the
SDK call.

Provider configuration lives entirely in `kb.config.yaml`. The top-level
`provider.name` selects the active provider, and the `providers` section holds
the built-in settings for `openai`, `anthropic`, and `gemini`:

```yaml
version: 6
provider:
  name: openai
providers:
  openai:
    model: gpt-5.4-mini
    api_key_env: OPENAI_API_KEY
    reasoning_effort: high
  anthropic:
    model: claude-sonnet-4-6
    api_key_env: ANTHROPIC_API_KEY
    thinking_budget: 10000
  gemini:
    model: gemini-3.1-flash-lite-preview
    api_key_env: GEMINI_API_KEY
    reasoning_effort: high
conversion:
  mistral_ocr:
    model: mistral-ocr-latest
    api_key_env: MISTRAL_API_KEY
    table_format: markdown
  html:
    renderer: wkhtmltopdf
    wkhtmltopdf_path: null
  fallbacks:
    pdf: docling
    docx: markitdown
    pptx: markitdown
    html: markitdown
graph:
  provider: openai
  model: gpt-4.1-mini
  embedding_provider: openai
  embedding_model: text-embedding-3-small
  api_key_env: null
  embedding_api_key_env: null
```

To customize a provider, edit its entry under `providers`:

```yaml
provider:
  name: anthropic
providers:
  anthropic:
    model: claude-opus-4-6
    api_key_env: MY_ANTHROPIC_KEY
    thinking_budget: 2048
```

The `graph` section controls GraphRAG runtime setup independently from the text-provider section used by `kb update`, `kb review`, and `kb legacy ask`, but it does not duplicate API keys by default. `api_key_env: null` means "resolve this from `providers.<graph.provider>.api_key_env`"; `embedding_api_key_env: null` does the same for `providers.<graph.embedding_provider>.api_key_env`. Run `kb graph init` after editing it so `graph/graphrag/settings.yaml` is refreshed.

OpenAI and Google Gemini both expose embedding models that can be configured for GraphRAG. Anthropic does not currently provide its own embedding model; Anthropic's embedding guidance points users to Voyage AI instead. GraphRAG uses LiteLLM underneath and supports non-OpenAI providers, but its own docs say OpenAI GPT-4-series models remain the most thoroughly tested path.

`kb.config.yaml` is versioned. The current schema version is 6, and the CLI automatically migrates older files when it loads project configuration.

You can also override the provider per-invocation without editing the config file:

```bash
kb --provider anthropic legacy ask "How does REALM differ from RAG?"
kb --provider gemini review
```

Set the matching API key as an environment variable. These are the defaults
seeded into `kb.config.yaml`:

| Provider | Default model | Env variable | Alternatives |
| --- | --- | --- | --- |
| `openai` | `gpt-5.4-mini` | `OPENAI_API_KEY` | `gpt-5.4`, `gpt-5.4-nano` |
| `anthropic` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | `claude-opus-4-6`, `claude-haiku-4-5` |
| `gemini` | `gemini-3.1-flash-lite-preview` | `GEMINI_API_KEY` | `gemini-3.1-pro-preview`, `gemini-2.5-flash` |

See the official model documentation for the full list of available models, pricing, and capabilities:

- **OpenAI:** [platform.openai.com/docs/models](https://platform.openai.com/docs/models)
- **Anthropic:** [docs.anthropic.com/en/docs/about-claude/models](https://docs.anthropic.com/en/docs/about-claude/models)
- **Google Gemini:** [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models)

If the provider is not configured, `kb update`, `kb legacy ask`, and `kb review`
fail with a configuration error. If the provider is configured but the API key
is missing, those commands fail before provider-backed work begins. Provider
calls retry transient failures (rate limits, timeouts, server errors)
automatically with exponential backoff and jitter; after retries are exhausted,
`kb legacy ask` and `kb review` fail while `kb update` can fall back for summaries and
concept clustering.

### Windows and Corporate TLS

Mistral OCR and provider SDK calls use Python's certificate trust path. On
Windows machines behind corporate TLS inspection, Python may fail with
`CERTIFICATE_VERIFY_FAILED` even when the browser works. Point `SSL_CERT_FILE`
or `REQUESTS_CA_BUNDLE` at a PEM bundle trusted by Python, then rerun
`kb doctor --strict` and retry the ingest/update command:

```powershell
$env:SSL_CERT_FILE = "D:\path\to\corporate-ca-bundle.pem"
$env:REQUESTS_CA_BUNDLE = $env:SSL_CERT_FILE
poetry run kb doctor --strict
```

## Conversion Configuration

Document conversion settings also live in `kb.config.yaml`. The `conversion`
section controls which converter is used for Mistral-native document types and
how HTML is rendered before OCR:

```yaml
conversion:
  mistral_ocr:
    model: mistral-ocr-latest
    api_key_env: MISTRAL_API_KEY
    table_format: markdown
  html:
    renderer: wkhtmltopdf
    wkhtmltopdf_path: null
  fallbacks:
    pdf: docling
    docx: markitdown
    pptx: markitdown
    html: markitdown
```

Default routing is:

- `.pdf`, `.docx`, `.pptx` → Mistral OCR first
- `.png`, `.jpg`, `.jpeg`, `.avif` → Mistral OCR first
- `.htm`, `.html` → `wkhtmltopdf` then Mistral OCR first
- `.csv`, `.epub`, `.ipynb`, `.xls`, `.xlsx` → MarkItDown
- `.md`, `.markdown`, `.txt` → direct passthrough

If a Mistral-first conversion fails quality checks, the configured fallback
converter is tried before the file is rejected. Image inputs currently have no
local fallback and fail cleanly if Mistral OCR cannot process them.

## Environment Variables

| Variable | Description |
| --- | --- |
| `OPENAI_API_KEY` | API key for the OpenAI provider. |
| `ANTHROPIC_API_KEY` | API key for the Anthropic provider. |
| `GEMINI_API_KEY` | API key for the Google Gemini provider. |
| `MISTRAL_API_KEY` | API key for Mistral OCR document conversion. |
| `NO_COLOR` | When set (any value), disables colored CLI output. Respected automatically by Rich. |

You can override provider and conversion settings in that same file:

```yaml
provider:
  name: openai
providers:
  openai:
    api_key_env: MY_CUSTOM_OPENAI_KEY
graph:
  provider: openai
  embedding_provider: openai
  api_key_env: null
  embedding_api_key_env: null
conversion:
  mistral_ocr:
    api_key_env: MY_CUSTOM_MISTRAL_KEY
  html:
    wkhtmltopdf_path: C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe
```

## Project Layout

After initialization and a few ingested sources, the project directory looks like this:

```text
project-root/
├── kb.config.yaml          # Project configuration (providers + conversion)
├── kb.schema.md            # Compilation schema
├── raw/
│   ├── _manifest.json      # Source metadata, hashes, converter info
│   ├── sources/            # Original ingested files
│   └── normalized/         # Canonical markdown/text artifacts
├── wiki/
│   ├── sources/            # Generated source pages
│   ├── concepts/           # Legacy generated concept pages
│   ├── graph/              # GraphRAG-derived markdown artifacts
│   │   ├── index.md
│   │   ├── documents/
│   │   ├── entities/
│   │   ├── relationships/
│   │   ├── communities/
│   │   └── text-units/
│   ├── analysis/           # Saved analysis pages from graph/legacy ask --save
│   ├── index.md            # Wiki index (human-readable)
│   ├── _index.json         # Wiki index (machine-readable)
│   └── log.md              # Update activity log
├── graph/
│   ├── exports/
│   │   ├── compile_runs.json      # Resume/failure state for update runs
│   │   └── search_index.sqlite3   # Temporary legacy SQLite FTS5 index
│   ├── runs/
│   │   └── graph_index_runs.json  # Local GraphRAG index run metadata, ignored
│   └── graphrag/
│       ├── settings.yaml          # GraphRAG JSON input configuration
│       ├── prompts/               # GraphRAG prompt templates
│       ├── output/                # Generated GraphRAG index tables, ignored
│       └── input/
│           └── sources.json       # Generated by kb graph sync, ignored by Git
└── vault/
    └── obsidian/           # Obsidian-friendly export
        └── sources/
```
