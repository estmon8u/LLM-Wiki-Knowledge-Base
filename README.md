# GraphWiki KB

A CLI-first GraphRAG research-memory system for ingesting technical documents, building a graph-based retrieval index, answering local/global research questions, and exporting inspectable wiki artifacts with provenance and source traceability.

The wiki is the human-readable artifact layer. Microsoft GraphRAG and the custom WikiGraphRAG backend are retrieval/synthesis engines; choose explicitly with `kb ask --engine graphrag` or `kb ask --engine wikigraph` when comparing them.

Backends are compared with a research-grounded RAG evaluation harness (rank-aware retrieval metrics + RAGAS + bias-mitigated LLM judge + bootstrap CIs) — see [docs/rag_evaluation.md](docs/rag_evaluation.md) and `scripts/evaluate_rag.py`.

WikiGraphRAG ships two engines selected by `wikigraph.mode`: the default `classic` wiki-page-first graph, and a new LightRAG-style `lightrag` backend (source-chunk-first entity/relation graph with dual-level local/global/hybrid retrieval). Switch with `kb update --wikigraph-mode lightrag` and see [docs/wikigraph_lightrag.md](docs/wikigraph_lightrag.md).

WikiGraphRAG is integrated into the main CLI instead of a sidecar command group.
It enters through `kb update`, `kb find --engine wikigraph|all`, and
`kb ask --engine wikigraph`. The backend is config-driven and optional-extra
safe: non-WikiGraphRAG commands can run without NetworkX installed, while
`kb update --export-wikigraph-artifacts` writes generated inspection cards
under `wiki/wikigraph/` when artifact export is enabled: classic mode exports
entity, community, chunk, and TextUnit cards; LightRAG mode exports entity,
relation, source-chunk, index, and diagnostics pages. `kb export
--wikigraph-modes all` can refresh both modes into
`wiki/wikigraph/classic/` and `wiki/wikigraph/lightrag/` before vault export
so their inspection cards can coexist in one Obsidian vault. Unset
WikiGraphRAG CLI flags defer to `wikigraph.enabled` and
`wikigraph.export_generated_artifacts`, context assembly honors
`max_context_tokens`, `lexical_backend: simple` forces the pure-Python ranker,
and `kb ask --engine wikigraph --show-source-trace` prints retrieved seed
entities, communities, sub-questions, per-context traces, and provider status.

WikiGraphRAG is the current default `kb ask` engine because it is local, inspectable, and can run provider-free; Microsoft GraphRAG remains available explicitly with `kb ask --engine graphrag` for provider-backed graph queries and side-by-side comparison. The existing SQLite FTS5 search and source-grounded ask workflow is explicit source-page-only legacy behavior under `kb find --engine legacy` and `kb ask --engine legacy` with deprecation warnings. Top-level `kb find` is a non-generative search over direct GraphRAG entity/relationship artifacts, the maintained wiki index, and WikiGraphRAG contexts, including generated graph pages when they exist. GraphRAG setup and maintenance are folded into the main command surface: `kb init` creates the graph workspace, `kb update` syncs input, indexes when needed, and exports graph wiki pages from complete graph output even when indexing is skipped as current, `kb status` reports graph health, and `kb export` refreshes graph inspection pages when complete graph output exists. The old `kb graph` command group has been removed. See [docs/graphrag-pivot.md](docs/graphrag-pivot.md) for the pivot rationale and target architecture.

## Requirements

- Python 3.11 or 3.12. The GraphRAG dependency currently documents Python 3.10-3.12 support, so GraphWiki KB blocks GraphRAG workflows on Python 3.13 until upstream support lands.
- [Poetry](https://python-poetry.org/) (installed at user level, not inside the project virtualenv)

## Installation

```bash
cd LLM-Wiki-Knowledge-Base
poetry install --with dev --all-extras
```

This creates a local `.venv` with the development tools and optional provider/converter/export dependencies used by the full test and real-document workflows. Minimal package installs can choose extras such as `graphwiki-kb[openai]`, `graphwiki-kb[agent]`, `graphwiki-kb[pdf]`, `graphwiki-kb[wikigraph]`, `graphwiki-kb[wikigraph-eval]`, or `graphwiki-kb[all]`; the `wikigraph` extras include the NetworkX/BM25, NumPy, and tiktoken dependencies needed by the classic and LightRAG backends. The CLI entrypoint is registered as `kb`.

## GraphRAG Workspace

Microsoft GraphRAG is installed as a library/CLI dependency, not a separate paid hosted service. Running real GraphRAG indexing or query jobs can still create model and embedding costs through the configured provider.

The repository contains an initialized GraphRAG workspace under `graph/graphrag/`. The committed scaffold includes `settings.yaml`, default prompts, and `input/`; local runtime files such as `.env`, generated `input/sources.json`, `output/`, `cache/`, `logs/`, run metadata under `graph/runs/*.json`, and `kb agent` session/run artifacts under `graph/runs/agent/` stay ignored.

```bash
poetry run graphrag --help
poetry run kb init
poetry run kb update
```

GraphRAG runtime defaults live in the `graph` section of `kb.config.yaml`. `kb init` calls the installed GraphRAG Python initialization entrypoint to create the workspace, then syncs the selected completion provider, embedding provider, models, JSON input settings, chunking defaults, technical extraction defaults, prompt paths, source-size limit, and resolved API-key environment variables into `graph/graphrag/settings.yaml`. The CLI owns those managed fields, but it preserves unrelated user-owned GraphRAG settings such as cache, vector-store, and search tuning when it rewrites the file. Bundled prompt templates under `graph/graphrag/prompts/` are copied only when missing; if a bundled template changes and a user-tuned prompt already exists, GraphWiki KB writes a sibling `*.new` file instead of overwriting the tuned prompt. By default, GraphRAG reuses the OpenAI provider entry and references `OPENAI_API_KEY`, so a separate `GRAPHRAG_API_KEY` is not required unless you explicitly set a graph-specific override. The standalone `graphrag` command is still available for diagnostics. `kb` uses signature-aware Python entrypoint adapters first and can fall back to the documented `graphrag` CLI command shape when an upstream entrypoint contract changes.

`kb update` uses the existing knowledge-base corpus as the source of truth. It reads `raw/_manifest.json` plus the normalized artifacts in `raw/normalized/`, refuses manifest paths that resolve outside the project `raw/` tree, rejects normalized sources above `graph.input.max_source_bytes`, writes compact `graph/graphrag/input/sources.json`, configures GraphRAG for JSON input with metadata prepended into chunks, and then auto-decides whether to run a full index, incremental update, retry after a failed latest attempt, or skip because the graph is already current. WikiGraphRAG consumes the same normalized artifacts as deterministic TextUnits, producing `source_document` and `text_unit` nodes, `documents.json`, and `text_units.json` under `graph/wikigraph/` so the custom backend is compared against GraphRAG's source-text retrieval layer instead of only curated wiki pages. Use `--graph-method auto|standard|fast|standard-update|fast-update` to request a specific GraphRAG index method. The preflight decision path plans those settings and input changes without mutating `settings.yaml` or `input/sources.json`, so skipped or credential-blocked graph runs do not leave partial workspace state behind. Normal updates can skip isolated missing normalized artifacts with a warning so one stale manifest entry does not block the entire graph refresh; run `kb lint` to catch and fix the manifest drift. Successful and failed index attempts record source digest, source hashes, runtime settings/prompt digest, selected method, active output directory when available, and command result under ignored run metadata for reproducibility. If GraphRAG credentials are missing during a normal `kb update`, the wiki compile still completes and graph indexing is skipped with a warning; `kb update --graph-only` fails clearly because the requested graph-only operation cannot run. Use `kb update --no-graph` when you intentionally want the legacy wiki compile/index refresh without touching the graph.

`kb status` checks whether settings, synced input, the active complete GraphRAG output tables, vector store, and last recorded index run are present. It distinguishes missing, unreadable, and dependency-missing Parquet table states; missing, empty, unreadable, incompatible, and ready vector-store states; and corrupt run metadata. `kb status --strict` exits non-zero unless the project sources are compiled and the GraphRAG index is complete, fresh, and query-ready. Status compares the current graph input digest, source hashes, settings, prompts, and GraphRAG runtime identity against the last successful index run, so complete-looking output is still marked stale when reproducibility metadata is missing or no longer matches. It prefers the active output directory recorded by the latest successful run when that directory is still complete, rather than trusting the newest output folder blindly. `kb ask --engine graphrag --method auto|basic|local|global|drift` checks the artifacts required by the selected query method, calls GraphRAG query entrypoints, passes the active output directory explicitly to GraphRAG, prints the selected route reason, and preserves returned non-streaming answers even when GraphRAG logs instead of printing them. Saved GraphRAG answers become unique analysis pages with support-level and source-trace metadata plus separate raw stdout/stderr sections; blank answers are refused. `kb export` reads GraphRAG Parquet tables in batches and writes human-readable documents, entities, relationships, communities, and text units under `wiki/graph/` when graph output exists.

### GraphRAG compatibility

GraphWiki KB currently pins Microsoft GraphRAG to `>=3.0.9,<3.1` and supports GraphRAG workflows on Python 3.11 and 3.12. Minor GraphRAG upgrades can change generated workspace settings and prompts, so after changing the GraphRAG version or graph runtime config, run `kb init` and then `kb update` to refresh managed workspace files and rebuild stale graph output. Local `.env`, generated input, output, cache, logs, and `graph/runs/*.json` metadata are safe runtime artifacts; hand edits to CLI-managed fields in `graph/graphrag/settings.yaml` will be overwritten from `kb.config.yaml`.

## Evaluation Harness

The benchmark in `eval/benchmark.yaml` now supports two complementary evaluation paths:

1. `scripts/evaluate_rag.py` compares legacy, Microsoft GraphRAG, `wikigraph-classic`, and `wikigraph-lightrag` with rank-aware retrieval metrics plus optional provider-backed RAGAS and judge scoring.
2. `scripts/evaluate_backends.py` remains as a thin compatibility wrapper for older `--backends ...` invocations.

The evaluator includes Microsoft GraphRAG as a first-class backend rather than
comparing only WikiGraphRAG against legacy FTS. The split WikiGraphRAG backends
let the classic and LightRAG-style modes run side by side when both indexes are
built. GraphRAG, legacy, and direct answer rows stay opt-in behind
`--allow-provider-calls`.

```bash
# Local-safe cross-backend retrieval comparison
poetry run python scripts/evaluate_rag.py --retrieval-only \
  --methods legacy graphrag wikigraph-classic wikigraph-lightrag

# Provider-backed answer comparison when the graph provider/API key is ready
poetry run python scripts/evaluate_rag.py --allow-provider-calls --ragas --judge \
  --methods legacy graphrag wikigraph-classic wikigraph-lightrag
```

Default/local-safe runs do not prove answer superiority: provider answer rows are skipped unless `--allow-provider-calls` is passed, while retrieval-only rows are diagnostics for source selection and routing. Treat headline answer comparisons as valid only when WikiGraphRAG and Microsoft GraphRAG both have successful comparable provider-backed rows. Per-run outputs are written under `eval/rag_eval/`.

## Quick Start

```bash
# 1. Initialize project and graph workspace
poetry run kb init

# 2. Add source documents or folders
poetry run kb add path/to/research-papers/

# 3. Refresh wiki + GraphRAG artifacts
#    This compiles, builds the graph index, and exports graph wiki pages.
poetry run kb update

# 4. Ask questions (WikiGraphRAG by default; Microsoft GraphRAG explicitly)
poetry run kb ask "How does REALM differ from RAG?"
poetry run kb ask --engine graphrag "What are the main retrieval themes across the corpus?" --method global

# 5. Check project health
poetry run kb status
poetry run kb doctor

# 6. Export to Obsidian vault
poetry run kb export --wikigraph-modes all
```

Use `kb add` when you want to stage one or more files or folders before running
maintenance. `kb update` then syncs normalized sources into GraphRAG input, runs
a preflight check, auto-detects whether a full, incremental, or retry graph index
is needed, and exports graph artifacts to the wiki when complete output exists.

Shortcut: `kb update path/to/research-papers/` adds sources and runs the
maintenance pipeline in one command. Use `--no-graph` for quick wiki-only
updates, or `--force` for a full rebuild including the graph index.

That's the current graph-backed workflow: **init -> add -> update -> status -> ask -> export**.
`kb ask` defaults to WikiGraphRAG, Microsoft GraphRAG is explicit with
`--engine graphrag`, and the deprecated FTS path exists only through
`--engine legacy` for comparison and exact lexical lookup. There is no silent
fallback from graph-backed engines to FTS5.

For a slower first-run walkthrough that keeps the repository and knowledge-base
project in separate directories, see [docs/start-guide.md](docs/start-guide.md).

Before running `kb update`, `kb ask --engine legacy`, or `kb review`, configure the active
provider in `kb.config.yaml` and set the matching API key environment
variable. If you ingest `.pdf`, `.docx`, `.pptx`, supported images, or
`.html` / `.htm`, also set `MISTRAL_API_KEY`. HTML inputs additionally require
either `wkhtmltopdf` on your `PATH` or the bundled pure-Python `xhtml2pdf`
fallback. See
`Provider Configuration` and `Conversion Configuration` below.

## Architecture Flow

```text
Documents
  |
  v
kb init
  - project scaffold
  - GraphRAG workspace and settings
  |
  v
kb add
  - ingest and normalize files or folders
  - update the source manifest
  |
  v
kb update
  - optionally add sources passed on the command line
  - compile source pages with provenance
  - optionally refresh legacy concept pages and FTS comparator index
  - sync GraphRAG JSON input
  - build WikiGraphRAG wiki chunks plus normalized source TextUnits
  - auto-select full index, incremental update, or skip
  - export graph tables to wiki/graph/
  |
  v
kb ask
  - defaults to WikiGraphRAG
  - can compare WikiGraphRAG, Microsoft GraphRAG, and legacy FTS with --engine
  - saves per-engine analysis pages when requested
  |
  v
kb status / kb doctor / kb lint
  - project and graph health
  - hash-based freshness checks
  |
  v
kb export
  - Obsidian vault
  - refreshed graph wiki inspection pages
```

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

These are the commands you will use most often today. The graph-backed happy
path is **init -> add -> update -> status -> ask -> export**. `update` can also
accept source paths as a shortcut when you want add-and-refresh in one command.

| Command | Description |
| --- | --- |
| `init` | Create project folders, config, schema, manifest, and GraphRAG workspace |
| `add` | Add and normalize source files or folders without running the full update |
| `update` | Build wiki pages, refresh the legacy comparator index, sync/index GraphRAG, and export graph pages |
| `find` | Search direct graph artifacts, WikiGraphRAG contexts, the maintained wiki index, or `--engine legacy` source-page FTS |
| `ask` | WikiGraphRAG-default answer command; use `--engine graphrag`, `--engine legacy`, comma lists, or `--engine all` for comparison |
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

Running `init` again is safe — it skips files that already exist. If an existing config file is malformed, `init` writes a timestamped `kb.config.yaml.bak.*` copy before regenerating the default file.
The scaffold writes `kb.config.yaml` at config version 9. Older configs are migrated in place on load so deprecated fields are removed and newer sections such as `providers`, `conversion`, `graph.routing`, `agent`, `research`, `conversion.html.allow_local_file_access`, `storage.raw_normalized_dir`, `wikigraph.mode`, `wikigraph.lightrag`, and top-level `embeddings` are persisted automatically.

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

Bring the knowledge base current. Optionally add new sources first, then build wiki pages, sync GraphRAG, and refresh indexes.

```bash
poetry run kb update
poetry run kb update path/to/new-paper.pdf
poetry run kb update --force
poetry run kb update --resume
poetry run kb update --graph-only
poetry run kb update --graph-method fast
```

| Option | Default | Description |
| --- | --- | --- |
| `--force` | off | Rebuild every source page even if nothing changed. |
| `--resume` | off | Resume the most recent failed or interrupted update run. Cannot be combined with `--force`. |
| `--no-graph` | off | Skip GraphRAG input sync, index refresh, and graph wiki export. |
| `--graph-only` | off | Run GraphRAG sync/index/export without legacy compile or search refresh. |
| `--graph-method` | `auto` | Request `auto`, `standard`, `fast`, `standard-update`, or `fast-update` for GraphRAG indexing. |
| `--allow-partial` | off | Treat GraphRAG sync/index/export failures as warnings during a normal update. |
| `--concepts` / `--no-concepts` | config | Opt in or out of legacy concept page generation for this run. |

When source paths are provided, the command adds them first. Legacy concept pages are skipped by default now that GraphRAG is the default cross-document retrieval layer; pass `--concepts` or set `concepts.enabled: true` in `kb.config.yaml` to refresh them.
Compile summaries request structured provider output (`summary`, `key_points`, `open_questions`, and `title_suggestion`) and persist the extra fields when returned. When concept generation is enabled, concept clustering uses deterministic collocation-based grouping by default. Set `concepts.provider_backed: true` to let the configured provider propose clusters; provider output is parsed through the shared structured-output parser, cached by source-page digest, and falls back to deterministic grouping if provider clustering fails.

`kb update` also owns routine GraphRAG maintenance. It prints `Mode: full`, `Mode: graph-only`, or `Mode: wiki-only` at the start, creates the graph workspace if needed, syncs normalized artifacts into `graph/graphrag/input/sources.json`, checks the current GraphRAG output and digest-based run metadata, selects a full `fast` build, `fast-update`, retry after the latest failed attempt, or skip, runs the selected index when graph credentials are available, and exports graph inspection pages under `wiki/graph/` whenever complete output exists. It prints the GraphRAG output path after indexing and surfaces a search warning if the legacy SQLite FTS5 refresh falls back to markdown scanning. Graph indexing is skipped with an explicit warning when provider credentials are missing during a normal update; the compile/wiki update still completes. Normal updates skip isolated missing normalized graph inputs with a warning instead of failing every source; `kb update --graph-only` keeps strict graph-input behavior and treats missing graph credentials as a hard error because no non-graph work was requested.

### Main GraphRAG behavior

The `kb graph` command group has been removed. GraphRAG setup, input sync, indexing, status, querying, and wiki export are now handled by the main commands:

| Main command | GraphRAG behavior |
| --- | --- |
| `kb init` | Initializes `graph/graphrag/` through GraphRAG's Python initialization entrypoint, then syncs project input, prompt, provider, model, and key-env settings. |
| `kb update` | Syncs normalized sources, auto-selects the graph index action, runs GraphRAG indexing when credentials are present, records run metadata, prints the graph output path after indexing, warns on skipped missing normalized inputs or legacy search fallback, and exports graph wiki pages from complete output even when indexing is skipped. |
| `kb update --force` | Rebuilds source pages and forces a full GraphRAG rebuild. |
| `kb update --no-graph` | Updates the wiki and legacy index without syncing or indexing GraphRAG. |
| `kb status` / `kb status --json` / `kb status --strict` | Includes GraphRAG workspace, input, active output directory, output-table, vector-store, freshness, last-index-run, row-count, strict-readiness, and next-action fields; in `wikigraph.mode: lightrag`, JSON also includes `wikigraph_status` and terminal output reports LightRAG tier, freshness, and graph counts. |
| `kb ask --engine graphrag --method auto|basic|local|global|drift` | Queries Microsoft GraphRAG after method-specific readiness checks. |
| `kb agent "..."` | Natural-language KB control plane backed by the OpenAI Agents SDK. Routes requests to `ask_kb`, `find_kb`, `status`, `lint`, `review`, `research`, `list_recommendations`, `ingest_recommendation`, and `update_kb` tools. Read-only by default; write tools require `--yes` or an interactive approval prompt. Web research uses the OpenAI Responses `web_search` tool and produces durable, numbered source recommendations under `graph/runs/agent/` without ingesting them. One-shot calls are sessionless; pass `--session ID` to share conversation state across runs. Install with `poetry install -E agent`. |
| `kb export` | Exports the vault and refreshes `wiki/graph/` when GraphRAG output exists. |

GraphRAG indexing uses the same decision table that powered the former graph sync wrapper:

| State | Default action |
| --- | --- |
| No GraphRAG Parquet output or vector store exists | Full rebuild with `fast` |
| Output exists but required tables or vector store are incomplete | Full rebuild with `fast` |
| Source hashes changed since the last successful index | Incremental update with `fast-update` |
| Latest index attempt failed after an older success | Retry with `fast-update` when complete output exists, otherwise `fast` |
| Graph runtime settings, prompts, GraphRAG package version, or managed schema version changed | Full rebuild with `fast` |
| Complete output exists but successful-run digest metadata is missing | Full rebuild with `fast` |
| Sources and runtime config match the last successful index | Skip indexing |

Graph wiki export reads standard GraphRAG tables such as `documents`, `text_units`, `entities`, `relationships`, `communities`, and `community_reports` from the active complete output directory under `graph/graphrag/output/`. A complete output directory means those tables are present and readable, and the configured vector store is ready. It writes:

- `wiki/graph/index.md`
- `wiki/graph/documents/*.md`
- `wiki/graph/entities/*.md`
- `wiki/graph/relationships/*.md`
- `wiki/graph/communities/*.md`
- `wiki/graph/text-units/*.md`

Generated graph pages use frontmatter types such as `graph_entity`, `graph_relationship`, `graph_community`, `graph_text_unit`, and `graph_document`. Document and text-unit pages render raw GraphRAG text in fenced code blocks so paper-internal markdown links and headings remain inspectable without becoming wiki lint targets. Large relationship tables are counted in `wiki/graph/index.md`, but only the strongest 500 relationship pages are materialized and entity pages show a bounded relationship table. Existing `wiki/concepts/` pages are not deleted; they are now legacy LLM-wiki concept pages beside the GraphRAG-derived `wiki/graph/` layer and are refreshed only when legacy concept generation is explicitly enabled.

Troubleshooting paths:

- Workspace: `graph/graphrag/`
- Synced GraphRAG input: `graph/graphrag/input/sources.json`
- Generated GraphRAG output: `graph/graphrag/output/`
- GraphRAG vector store: normally `graph/graphrag/output/lancedb/`
- Active output in JSON status: `graph_status.active_output_dir`
- Graph wiki pages: `wiki/graph/`
- Local index-run metadata: `graph/runs/graph_index_runs.json`

### `kb find <terms>`

Searches direct GraphRAG entity and relationship artifacts and the maintained
wiki index, then deduplicates and ranks all candidates together so weak graph
hits do not hide stronger wiki pages. In `--engine wikigraph` mode,
`--method auto|basic|local|global|hybrid|drift-lite` controls the
WikiGraphRAG retrieval path, including LightRAG dual-level retrieval when
`wikigraph.mode: lightrag` is active. Use
`kb find --engine legacy` only when you specifically need the deprecated source-page-only
FTS5 comparator path.

```bash
poetry run kb find "traceability citation"
poetry run kb find --limit 10 "agent architecture"
poetry run kb find --engine wikigraph --method hybrid --json "REALM vs RAG"
poetry run kb find --json "REALM vs RAG"
```

| Option | Default | Description |
| --- | --- | --- |
| `--limit` | 5 | Maximum number of results to return. |
| `--engine` | `auto` | Search `auto`, `graphrag`, `wiki`, `wikigraph`, `legacy`, or `all`. |
| `--method` | `auto` | WikiGraphRAG retrieval method; ignored by GraphRAG artifact, wiki, and legacy search. |
| `--json` | off | Output results as JSON for scripting. |

This is non-generative navigation over the maintained wiki search index. JSON
output includes `retriever: "wiki-index"`. It does not call GraphRAG query
modes or the deprecated FTS ask path.

### `kb find --engine legacy <terms>`

Search the deprecated SQLite FTS5 comparator index for source-page snippets.

```bash
poetry run kb find --engine legacy "traceability citation"
poetry run kb find --engine legacy --limit 10 "agent architecture"
poetry run kb find --engine legacy --json "REALM vs RAG"
```

| Option | Default | Description |
| --- | --- | --- |
| `--limit` | 5 | Maximum number of results to return. |
| `--json` | off | Output results as JSON for scripting. |

Uses a SQLite FTS5 chunk index stored at `graph/exports/search_index.sqlite3`.
This is temporary legacy behavior rather than the final retrieval engine.
The command searches compiled source pages only, ranks hits with BM25-style FTS
ordering, and returns page-level results using the best matching chunk snippet.
JSON output includes
`retriever: "legacy-fts"` and `deprecated: true` metadata. Evidence chunks skip
metadata-only sections such as `Source Details`, `Source Pages`,
`Related Concept Pages`, and `Citations` so retrieval and citations point at
content rather than wiki bookkeeping.

### `kb ask <question>`

Ask a question through the GraphRAG-aware answer controller.

```bash
poetry run kb ask "How does REALM differ from RAG?"
poetry run kb ask --method global "What are the main retrieval themes across the corpus?"
poetry run kb ask --method hybrid --save "Compare RAG, REALM, FiD, Self-RAG, and GraphRAG."
poetry run kb ask --engine graphrag --method drift "Compare graph communities across the corpus."
poetry run kb ask --show-source-trace "How does the graph index support source traceability?"
```

| Option | Default | Description |
| --- | --- | --- |
| `--method` | `auto` | Use backend-specific routing or force a supported method. WikiGraphRAG supports `basic`, `local`, `global`, `hybrid`, and `drift-lite`; Microsoft GraphRAG supports `basic`, `local`, `global`, and `drift`. |
| `--community-level` | | Forward GraphRAG's community-level option. |
| `--dynamic-community-selection` / `--no-dynamic-selection` | GraphRAG default | Forward GraphRAG dynamic community selection behavior. |
| `--response-type` | GraphRAG default | Forward GraphRAG's response type option when the installed entrypoint accepts it; GraphRAG Basic Search builds that omit this argument are handled automatically. |
| `--save` | off | Save the graph answer as an analysis page under `wiki/analysis/`. |
| `--save-as` | | Save with a custom analysis slug. Implies `--save`. |
| `--show-source-trace` | off | Print source trace before the answer. GraphRAG shows route and support metadata; WikiGraphRAG shows seed entities, communities, sub-questions, per-context traces, and provider status. |
| `--json` | off | Include GraphRAG answer metadata as JSON. |

### `kb agent [natural language request]`

Run a natural-language control plane over the local KB through the optional
OpenAI Agents SDK. The agent decides which tool to call and routes the request
through existing services. The current toolset covers:

- read tools: `ask_kb` (GraphRAG or WikiGraphRAG), `find_kb`
  (graph/wiki/WikiGraphRAG fusion), `status` (including WikiGraphRAG index
  state), `lint`, `review`
- research: `research` (local KB plus optional OpenAI Responses `web_search`
  with structured source recommendations and persistent run records)
- recommendation follow-ups: `list_recommendations` (reads the saved research
  run record without re-running web search)
- write tools (require explicit user approval or `--yes`):
  `ingest_recommendation`, `update_kb`

```bash
poetry install --extras agent
poetry run kb agent "What does my KB say about GraphRAG evaluation?"
poetry run kb agent --show-plan "Is my graph stale?"
poetry run kb agent --json "Find notes about agent memory"
poetry run kb agent "Research current GraphRAG evaluation work and recommend sources"
poetry run kb agent "show previous recommendations"
poetry run kb agent --yes "add recommendation 2 and update the KB"
poetry run kb agent
```

| Option | Default | Description |
| --- | --- | --- |
| `--show-plan` | off | Print the tool calls used before the answer. |
| `--json` | off | Return the serialized agent run record. Requires a one-shot prompt. |
| `--yes` | off | Auto-approve safe writes (`ingest_recommendation`, `update_kb`). Destructive future tools still require explicit approval. |
| `--session ID` | sessionless (one-shot) / `repl` (interactive) | SQLite session id for cross-call memory. One-shot calls default to no shared session so unrelated commands cannot inherit stale context. |

The command keeps the CLI/service boundary intact: `commands/agent.py` owns
Click and REPL behavior, `AgentService` owns one-turn execution, the OpenAI SDK
runtime owns tool selection and session history, and every tool calls an
existing typed service instead of shelling out to another `kb` command. The
`update_kb` tool drives the full `UpdateService` pipeline (ingest, compile,
search refresh, GraphRAG sync, wiki export) and falls back to a `kb update`
subprocess when invoked from a worker thread where signal handlers used by
GraphRAG indexing would otherwise fail. Approval is enforced by local
`PendingApproval` records because the current SDK approval interruption support
does not cover these local function tools.

Agent tool wrappers remain synchronous, record structured trace entries on
failure, and call the existing KB services rather than creating a second
implementation of ask, find, lint, review, research, ingest, or update behavior.
The agent also mirrors the main WikiGraphRAG CLI surface: comparison prompts
call `ask_kb` once per requested engine, `find_kb` uses the same fused retrieval
options as `kb find`, `status` includes a WikiGraphRAG block, and `update_kb`
threads the WikiGraphRAG build flags through the approval payload.

Research runs are persisted under ignored `graph/runs/agent/`:

- `research-*.json` — full run record (question, KB gaps, web findings,
  numbered `SourceRecommendation` objects).
- `wiki/analysis/agent-research-*.md` — Markdown research report.
- `latest.json` — pointer to the most recent run with at least one
  recommendation, so "add recommendation 2" remains resolvable after a
  follow-up research call that produced no recommendations.
- `agent-run-*.json` — durable trace of each agent turn (tool calls,
  approvals, final output).
- `sessions.sqlite` — Agents SDK conversation memory for sessions you
  explicitly opt into via `--session` or the interactive REPL.

The controller checks workspace, input, freshness, and the artifacts required by
the selected query method before querying. For example, `global` queries require
community tables, while `basic`, `local`, and `drift` require a readable vector
store. It does not silently fall back to FTS5. If the graph is missing or not
ready, run `kb init` and `kb update` as directed by the error.
The deprecated top-level `--limit` compatibility flag is rejected for GraphRAG
answers because result limiting belongs to the legacy source-page evidence path.
Saved pages use `retriever: graph`, `method`, `planner`, `claim_support`,
`index_run_id`, `graph_input_hash`, `input_manifest_hash`, and source-trace
metadata. `graph_input_hash` is the hash of the synced GraphRAG input file;
`input_manifest_hash` is the raw manifest hash recorded in that input when
available. The current GraphRAG wrapper records command traces and parsed
`[Data: ...]` references, including GraphRAG report/community-report labels,
when GraphRAG emits them; it does not claim full citation evidence until source
spans are parsed. Raw stdout and stderr are preserved in separate sections for
audit, answer prose is cleaned before the `## Answer` section, blank GraphRAG
answers are refused, and repeated saves use unique filenames instead of
overwriting the prior analysis page.

### `kb ask --engine legacy <question>`

Answer a question from compiled source-page evidence with provider-backed synthesis and citations through the deprecated SQLite FTS5 retrieval path.

```bash
poetry run kb ask --engine legacy "How does the wiki handle stale pages?"
poetry run kb ask --engine legacy --limit 5 "What normalization converters are supported?"
poetry run kb ask --engine legacy --save "What does the update pipeline do?"
poetry run kb ask --engine legacy --save-as freshness "How is freshness tracked?"
poetry run kb ask --engine legacy --show-evidence "What formats are supported?"
```

| Option | Default | Description |
| --- | --- | --- |
| `--limit` | 3 | Maximum number of source pages to use as evidence. |
| `--save` | off | Save the answer as an analysis page in the wiki. |
| `--save-as` | | Save the answer as an analysis page with a custom slug. |
| `--show-evidence` | off | Print the retrieved evidence snippets before the answer. |

Requires a configured provider. Retrieves the best-matching source-page chunks as evidence, excluding saved analysis pages and generated concept pages so answers cite primary compiled sources. It packages the top chunk snippets into a frozen evidence bundle, asks the provider for structured output (`answer_markdown`, `claims`, `citations`, and `insufficient_evidence`), semantically validates the response, strips raw inline citation-ref markers from answer prose, and prints the answer followed by a Citations section.

Provider-backed answers must be both parseable and useful. Empty `answer_markdown`, missing claims when `insufficient_evidence` is false, claims without citation refs, and citation refs outside the retrieved evidence set fail the command instead of being treated as a successful answer. Provider failures include response diagnostics such as finish reason and token counts when the selected SDK exposes them.

Use `--save` or `--save-as` to persist the answer as a markdown analysis page in `wiki/analysis/` with YAML frontmatter (`type: analysis`), the question, a timestamp, `insufficient_evidence`, claim/citation counts, structured claims/provider citations when available, provider status diagnostics, and backlinks to cited source chunks. Blank answers are refused rather than saved. Saved analysis pages are indexed for top-level `kb find` and appear in `wiki/index.md` immediately, but `kb find --engine legacy` and later legacy ask runs stay source-only so saved answers are not recursively cited as primary evidence. Repeated saves use unique `wiki/log.md` headings so lint does not treat a rerun as a duplicate-heading issue.

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
- Manifest entries whose raw source file or normalized artifact no longer exists
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
| `--changed` | off | Show a preview of new, changed, missing, and up-to-date sources. |
| `--json` | off | Output results as JSON for scripting. |

Default view shows a Knowledge Base overview with source and wiki counts, the last update timestamp, and a suggestion for what to do next. With `--changed`, shows each source with a status tag (`[NEW]`, `[CHANGED]`, `[MISSING]`, `[OK]`) followed by a summary section with counts.

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
poetry run kb export --wikigraph-modes all
```

| Option | Default | Description |
| --- | --- | --- |
| `--clean` | off | Remove stale vault markdown files that were not exported from the current wiki set. |
| `--wikigraph-modes` | `none` | Refresh generated WikiGraphRAG cards before vault export. Use `current` for `wiki/wikigraph/`, `classic` or `lightrag` for one mode under `wiki/wikigraph/<mode>/`, or `all` for both mode-separated artifact sets. |

Copies compiled wiki pages and common display assets (`.png`, `.jpg`, `.jpeg`, `.gif`, `.svg`, `.webp`) into `vault/obsidian/` in a format compatible with [Obsidian](https://obsidian.md/). With `--clean`, stale markdown files are deleted only after the current export set is built, so the cleanup decision uses the exact destination paths created by the run.


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
| PDF | `.pdf` | Mistral OCR primary (Docling, then MarkItDown fallback) |
| HTML | `.htm`, `.html` | `wkhtmltopdf` or `xhtml2pdf` → Mistral OCR (MarkItDown fallback; preserves the source `<title>` when available) |
| CSV | `.csv` | MarkItDown |
| Word | `.docx` | Mistral OCR (MarkItDown fallback) |
| PowerPoint | `.pptx` | Mistral OCR (MarkItDown fallback) |
| Excel | `.xls`, `.xlsx` | MarkItDown |
| EPUB | `.epub` | MarkItDown |
| Jupyter Notebook | `.ipynb` | MarkItDown |
| Images | `.png`, `.jpg`, `.jpeg`, `.avif` | Mistral OCR |

All non-text formats are normalized into canonical markdown and stored in
`raw/normalized/` before compilation. PDFs intentionally use the paid Mistral
OCR API as the primary converter because conversion accuracy determines the
quality of every downstream source page, graph index, retrieval result, and
answer. Local PDF converters are resilience fallbacks only; fallback metadata is
recorded so lower-confidence conversions can be reviewed or rerun with Mistral.
For HTML, the source document `<title>` is extracted before rendering and kept as
the normalized page title when available, which prevents navigation chrome such
as "Skip to content" from becoming the durable source title after OCR.

## Provider Configuration

`kb update`, `kb ask --engine legacy`, and `kb review` require a configured provider
for legacy compile/review work. `kb update --graph-only` can refresh GraphRAG
without the legacy text-provider section when the GraphRAG provider credentials
are available.
If the provider is missing, those commands fail with a configuration error.
`kb ask --engine legacy` and `kb review` fail on provider execution errors; `kb update`
keeps deterministic fallbacks for compile summaries and concept clustering after
a provider call failure. During a normal `kb update`, missing GraphRAG credentials
skip graph indexing with an explicit warning after the wiki compile finishes.
`kb update --graph-only` fails on missing graph credentials. Other GraphRAG
sync/index/export failures are hard failures by default; use
`kb update --allow-partial` only when you intentionally want the legacy wiki
refresh to finish while graph work is reported as a warning.

Provider responses carry the returned text plus model/provider diagnostics such
as finish reason and token counts when the SDK exposes them. Saved `kb ask --engine legacy`
analysis pages persist those parsed/validated diagnostics under `provider_status`
when the answer comes from a provider. Structured provider
outputs are parsed through the shared JSON parser, which accepts direct JSON,
fenced JSON, and common prose-prefaced JSON before schema and semantic
validation. Services can override provider reasoning effort and output budgets
per operation; schema-bound commands use lower reasoning settings and larger
visible-output budgets where needed. OpenAI reasoning arguments are sent only to
known reasoning-capable model families, and configured effort names are
validated before SDK calls. Anthropic schema-bound requests use
`output_config.format` instead of prompt-only JSON instructions. Gemini
recursively resolves local schema definitions where possible, strips unsupported
JSON Schema keywords such as `additionalProperties`, and records a
schema-transformation report whenever the provider schema is weakened for Gemini
compatibility.

Provider configuration lives entirely in `kb.config.yaml`. The top-level
`provider.name` selects the active provider, and the `providers` section holds
the built-in settings for `openai`, `anthropic`, and `gemini`:

```yaml
version: 9
provider:
  name: openai
providers:
  openai:
    model: gpt-5.4-nano
    api_key_env: OPENAI_API_KEY
    reasoning_effort: high
    api: responses
    store_responses: false
  anthropic:
    model: claude-sonnet-4-6
    api_key_env: ANTHROPIC_API_KEY
    thinking_effort: medium
  gemini:
    model: gemini-2.5-flash
    api_key_env: GEMINI_API_KEY
    reasoning_effort: high
wikigraph:
  mode: classic
  lightrag:
    chunk_token_size: 1200
    chunk_overlap_tokens: 100
    extraction:
      extractor: deterministic
    retrieval:
      default_method: hybrid
    embeddings:
      required_for_strict_lightrag: true
      local_fallback: bm25
embeddings:
  provider: openai
  model: text-embedding-3-large
  dimension: 3072
  api_key_env: null
conversion:
  mistral_ocr:
    model: mistral-ocr-latest
    api_key_env: MISTRAL_API_KEY
    table_format: markdown
  html:
    renderer: wkhtmltopdf
    wkhtmltopdf_path: null
    allow_local_file_access: false
  fallbacks:
    pdf:
      - docling
      - markitdown
    docx:
      - markitdown
    pptx:
      - markitdown
    html:
      - markitdown
graph:
  provider: openai
  model: gpt-5.4-nano
  embedding_provider: openai
  embedding_model: text-embedding-3-small
  api_key_env: null
  embedding_api_key_env: null
  chunking:
    size: 1200
    overlap: 150
  extraction:
    entity_types:
      - concept
      - technology
      - method
      - algorithm
      - dataset
      - model
      - benchmark
      - framework
      - component
      - api
      - paper
      - claim
    max_gleanings: 2
  input:
    max_source_bytes: 26214400
  routing:
    aliases: {}
agent:
  enabled: true
  model: gpt-5.5
  max_turns: 8
  require_approval_for_writes: true
  save_runs: true
  trace: true
  session_backend: sqlite
research:
  web_enabled: true
  web_model: gpt-5.5
  search_context_size: medium
  max_recommendations: 5
  max_web_sources: 12
  require_approval_for_ingest: true
  default_domains_allowlist: []
  default_domains_blocklist:
    - reddit.com
    - quora.com
```

To customize a provider, edit its entry under `providers`:

```yaml
provider:
  name: anthropic
providers:
  anthropic:
    model: claude-opus-4-6
    api_key_env: MY_ANTHROPIC_KEY
    thinking_effort: high
```

Modern Claude 4.6-style models use Anthropic adaptive thinking through
`thinking_effort`. The provider sends Anthropic's adaptive-thinking flag plus
the requested effort, and schema-bound Anthropic requests use the API-enforced
structured-output format. A legacy `thinking_budget` value is still accepted
for older models that require manual extended-thinking token budgets. Gemini
uses model-sensitive thinking configuration: Gemini 2.5 receives a
`thinking_budget`, while Gemini 3.x receives a `thinking_level`. Config
validation accepts `none`, `minimal`, `low`, `medium`, and `high` for Gemini
reasoning effort; `xhigh` is rejected instead of silently mapping down to
`high`. Gemini schemas are normalized to the subset accepted by the API before
generation.
The adaptive-thinking detector is version-pattern based for Claude 4.6 and
newer Sonnet/Opus model identifiers rather than locked to a single model name.
OpenAI uses the Responses API by default for reasoning controls and structured
JSON output; set `providers.openai.api: chat_completions` only when you need the
legacy fallback path. OpenAI response storage is disabled by default through
`providers.openai.store_responses: false`; set it to `true` only when you
explicitly want provider-side response retention for your account.

The `graph` section controls GraphRAG runtime setup independently from the text-provider section used by `kb update`, `kb review`, and `kb ask --engine legacy`, but it does not duplicate API keys by default. `api_key_env: null` means "resolve this from `providers.<graph.provider>.api_key_env`"; `embedding_api_key_env: null` does the same for `providers.<graph.embedding_provider>.api_key_env`. `graph.chunking`, `graph.extraction`, and `graph.input.max_source_bytes` are CLI-managed GraphRAG settings; the default entity taxonomy is tuned for technical and research corpora rather than person/place/news corpora. Run `kb init` after editing it to refresh the managed provider/model/API-key/chunking/extraction/input fields in `graph/graphrag/settings.yaml`; `kb update` also syncs those managed fields before deciding whether model or prompt changes require a rebuild while preserving unrelated GraphRAG tuning in the workspace settings file.

OpenAI and Google Gemini both expose embedding models that can be configured for GraphRAG. Anthropic does not currently provide its own embedding model; Anthropic's embedding guidance points users to Voyage AI instead. GraphRAG uses LiteLLM underneath and supports non-OpenAI providers, but its own docs say OpenAI GPT-4-series models remain the most thoroughly tested path.

`kb.config.yaml` is versioned. The current schema version is 8, and the CLI automatically migrates older files when it loads project configuration. The `agent` and `research` sections control the optional OpenAI Agents SDK control plane used by `kb agent` and the web-backed research pipeline.

You can also override the provider per-invocation without editing the config file:

```bash
kb --provider anthropic ask --engine legacy "How does REALM differ from RAG?"
kb --provider gemini review
```

Set the matching API key as an environment variable. These are the defaults
seeded into `kb.config.yaml`:

| Provider | Default model | Env variable | Alternatives |
| --- | --- | --- | --- |
| `openai` | `gpt-5.4-nano` | `OPENAI_API_KEY` | `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini` |
| `anthropic` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | `claude-opus-4-6`, `claude-haiku-4-5` |
| `gemini` | `gemini-2.5-flash` | `GEMINI_API_KEY` | `gemini-3.1-pro-preview`, `gemini-2.5-flash` |

As of the latest OpenAI model docs checked for this project, `gpt-5.5` is the flagship model, while the cost-sensitive current smaller variants are `gpt-5.4-mini` and `gpt-5.4-nano`; there is no documented `gpt-5.5-mini` or `gpt-5.5-nano` model ID. OpenAI also documents `gpt-5-nano`, but recommends starting with `gpt-5.4-nano` for most new speed- and cost-sensitive workloads. The project uses `gpt-5.4-nano` as the default to keep graph rebuilds and routine provider-backed maintenance inexpensive, with `gpt-5.4-mini`, `gpt-5.4`, or `gpt-5.5` available as explicit quality/cost upgrades.

See the official model documentation for the full list of available models, pricing, and capabilities:

- **OpenAI:** [platform.openai.com/docs/models](https://platform.openai.com/docs/models)
- **Anthropic:** [docs.anthropic.com/en/docs/about-claude/models](https://docs.anthropic.com/en/docs/about-claude/models)
- **Google Gemini:** [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models)

If the provider is not configured, `kb update`, `kb ask --engine legacy`, and `kb review`
fail with a configuration error. If the provider is configured but the API key
is missing, those commands fail before provider-backed work begins. Provider
calls retry transient failures (rate limits, timeouts, server errors)
automatically with exponential backoff and jitter; after retries are exhausted,
`kb ask --engine legacy` and `kb review` fail while `kb update` can fall back for summaries and
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
section keeps Mistral OCR as the primary route for PDFs and the other
Mistral-native formats, then controls the ordered fallback chain used only after
that primary route fails. This is intentional: paying for higher OCR accuracy is
the right default for this project because bad conversion contaminates compile
summaries, citations, graph entities, relationships, retrieval, and final
answers. HTML rendering tries `wkhtmltopdf` first when available and falls back
to the bundled pure-Python `xhtml2pdf` renderer before OCR:

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
    pdf:
      - docling
      - markitdown
    docx:
      - markitdown
    pptx:
      - markitdown
    html:
      - markitdown
```

Default routing is:

- `.pdf` → Mistral OCR first, then Docling, then MarkItDown if needed
- `.docx`, `.pptx` → Mistral OCR first, then MarkItDown if needed
- `.png`, `.jpg`, `.jpeg`, `.avif` → Mistral OCR first
- `.htm`, `.html` → `wkhtmltopdf` first, then `xhtml2pdf` if needed, then Mistral OCR, with the original HTML `<title>` retained when available
- `.csv`, `.epub`, `.ipynb`, `.xls`, `.xlsx` → MarkItDown
- `.md`, `.markdown`, `.txt` → direct passthrough

If a Mistral-first conversion fails quality checks, the configured fallback
chain is tried before the file is rejected. Treat `fallback_used: true` in the
manifest as a lower-confidence conversion signal; for serious corpus work, set
`MISTRAL_API_KEY` and rerun the source through Mistral instead of accepting a
local PDF fallback as equivalent. Image inputs currently have no local fallback
and fail cleanly if Mistral OCR cannot process them.
HTML rendering disables `wkhtmltopdf` local-file access by default; set
`conversion.html.allow_local_file_access: true` only for trusted local HTML that
must load adjacent local assets.
Inline Mistral OCR payloads are preflight-limited to 100 MB before file bytes
are read or sent to the SDK. Split larger documents or switch to a provider
upload/document-url workflow before ingesting them.

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
    allow_local_file_access: false
```

## Quality Gates

Python source and tests are formatted with Black, linted with Ruff, and checked
with mypy across `src/graphwiki_kb`. CI runs those checks on Python 3.11 and
3.12 before the full pytest coverage suite, then builds the wheel, installs it
with all extras in a clean environment, and smoke-tests the installed `kb`
entrypoint:

```powershell
poetry run black --check src tests
poetry run ruff check src tests
poetry run mypy src/graphwiki_kb
poetry run pytest tests -q
poetry build
```

The same Black, Ruff, and mypy checks are available through pre-commit hooks.

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
│   │   ├── agent/                 # kb agent SQLite sessions, research runs, traces, ignored
│   │   └── graph_index_runs.json  # Local GraphRAG index run metadata, ignored
│   └── graphrag/
│       ├── settings.yaml          # GraphRAG JSON input configuration
│       ├── prompts/               # GraphRAG prompt templates
│       ├── output/                # Generated GraphRAG index tables, ignored
│       └── input/
│           └── sources.json       # Generated by kb update, ignored by Git
└── vault/
    └── obsidian/           # Obsidian-friendly export
        └── sources/
```
