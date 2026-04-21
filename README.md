# Capstone KB

A CLI-first tool for building and maintaining a citation-grounded markdown knowledge base from heterogeneous technical documents.

## Requirements

- Python 3.11.x
- [Poetry](https://python-poetry.org/) (installed at user level, not inside the project virtualenv)

## Installation

```bash
cd LLM-Wiki-Knowledge-Base
poetry install
```

This creates a local `.venv` and installs all dependencies. The CLI entrypoint is registered as `kb`.

## Quick Start

```bash
# 1. Initialize a new project
poetry run kb init

# 2. Add some source documents
poetry run kb add path/to/paper.pdf
poetry run kb add path/to/notes.md
poetry run kb add path/to/slides.pptx
poetry run kb add path/to/research-folder

# 3. Update the knowledge base (generates pages, concepts, and search indexes)
poetry run kb update

# 4. Search the wiki
poetry run kb find "knowledge base traceability"

# 5. Ask a question with citations
poetry run kb ask "How does the wiki handle stale pages?"
```

That's the everyday workflow: **add → update → find / ask**. Everything else
is optional.

Before running `kb update`, `kb ask`, or `kb review`, add a `provider`
section to `kb.config.yaml` and set the matching API key environment variable.
See `Provider Configuration` below.

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

These are the commands you will use most often. The happy path is
**init → add → update → find / ask**.

| Command | Description |
| --- | --- |
| `init` | Create project folders, config, schema, and manifest |
| `add` | Add and normalize source documents |
| `update` | Build wiki pages, generate concepts, and refresh indexes |
| `find` | Search the wiki |
| `ask` | Answer a question from compiled evidence |
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
The scaffold writes `kb.config.yaml` at config version 2. Older version 1 configs are migrated in place on load so deprecated fields are removed and newer sections such as `provider` and `storage.raw_normalized_dir` are persisted automatically.

### `kb doctor`

Run health checks on the project: structure, config, provider, API keys, converters, and run-artifact database.

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

### `kb find <terms>`

Search the wiki for relevant pages and snippets.

```bash
poetry run kb find "traceability citation"
poetry run kb find --limit 10 "agent architecture"
```

| Option | Default | Description |
| --- | --- | --- |
| `--limit` | 5 | Maximum number of results to return. |
| `--json` | off | Output results as JSON for scripting. |

Uses a SQLite FTS5 chunk index stored at `graph/exports/search_index.sqlite3`. Indexes all wiki pages (source pages, concept pages, and saved analysis pages), ranks hits with BM25-style FTS ordering, and returns page-level results using the best matching chunk snippet.

### `kb ask <question>`

Answer a question from compiled wiki evidence with provider-backed synthesis and citations.

```bash
poetry run kb ask "How does the wiki handle stale pages?"
poetry run kb ask --limit 5 "What normalization converters are supported?"
poetry run kb ask --save "What does the update pipeline do?"
poetry run kb ask --save-as freshness "How is freshness tracked?"
poetry run kb ask --show-evidence "What formats are supported?"
```

| Option | Default | Description |
| --- | --- | --- |
| `--limit` | 3 | Maximum number of source pages to use as evidence. |
| `--save` | off | Save the answer as an analysis page in the wiki. |
| `--save-as` | | Save the answer as an analysis page with a custom slug. |
| `--show-evidence` | off | Print the retrieved evidence snippets before the answer. |

Requires a configured provider. Retrieves the best-matching indexed wiki chunks as evidence, packages the top chunk snippets into a frozen evidence bundle, sends that evidence to the configured provider, and prints the answer followed by a Citations section.

Use `--save` or `--save-as` to persist the answer as a markdown analysis page in `wiki/analysis/` with YAML frontmatter (`type: analysis`), the question, a timestamp, and backlinks to cited source chunks. Saved analysis pages are indexed for future searches.

### `kb lint`

Run deterministic structural lint checks over the maintained wiki.

```bash
poetry run kb lint
```

Checks for:

- Broken internal links in both wiki-link and standard markdown-link form
- Missing heading fragments for links such as `[[Page#Section]]` and `[text](page.md#section)`
- Duplicate page titles, repeated headings, skipped heading levels, and multiple H1 headings
- Missing frontmatter or provenance metadata on compiled pages
- Typed frontmatter validation: string, date (ISO format), and list fields
- Empty compiled pages that have no body content beyond headings
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

View project configuration as YAML, or manage provider and model settings.

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
| PDF | `.pdf` | Docling |
| HTML | `.htm`, `.html` | MarkItDown |
| CSV | `.csv` | MarkItDown |
| Word | `.docx` | MarkItDown |
| PowerPoint | `.pptx` | MarkItDown |
| Excel | `.xls`, `.xlsx` | MarkItDown |
| EPUB | `.epub` | MarkItDown |
| Jupyter Notebook | `.ipynb` | MarkItDown |

All non-text formats are normalized into canonical markdown and stored in `raw/normalized/` before compilation.

## Provider Configuration

`kb update`, `kb ask`, and `kb review` require a configured provider.
If the provider is missing, those commands fail with a configuration error. If a
configured provider call fails, they fail instead of falling back. Configure the
provider in `kb.config.yaml`:

```yaml
provider:
  name: openai          # openai | anthropic | gemini
  model: gpt-5.4-mini   # optional — defaults to a cost-effective model per provider
```

`kb.config.yaml` is versioned. The current schema version is 2, and the CLI automatically migrates older version 1 files when it loads project configuration.

You can also override the provider per-invocation without editing the config file:

```bash
kb --provider anthropic ask "How does REALM differ from RAG?"
kb --provider gemini review
```

Set the matching API key as an environment variable:

| Provider | Default model | Env variable | Alternatives |
| --- | --- | --- | --- |
| `openai` | `gpt-5.4-mini` | `OPENAI_API_KEY` | `gpt-5.4`, `gpt-5.4-nano` |
| `anthropic` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | `claude-opus-4-6`, `claude-haiku-4-5` |
| `gemini` | `gemini-3.1-flash-lite-preview` | `GEMINI_API_KEY` | `gemini-3.1-pro-preview`, `gemini-2.5-flash` |

If the provider is not configured, `kb update`, `kb ask`, and `kb review`
fail with a configuration error. If the provider is configured but the API key
is missing or the provider call fails, those commands fail instead of falling
back. Provider calls retry transient failures (rate limits, timeouts, server
errors) automatically with exponential backoff and jitter.

## Environment Variables

| Variable | Description |
| --- | --- |
| `OPENAI_API_KEY` | API key for the OpenAI provider. |
| `ANTHROPIC_API_KEY` | API key for the Anthropic provider. |
| `GEMINI_API_KEY` | API key for the Google Gemini provider. |
| `NO_COLOR` | When set (any value), disables colored CLI output. Respected automatically by Rich. |

You can override the environment variable name with `api_key_env`:

```yaml
provider:
  name: openai
  api_key_env: MY_CUSTOM_KEY
```

## Project Layout

After initialization and a few ingested sources, the project directory looks like this:

```text
project-root/
├── kb.config.yaml          # Project configuration
├── kb.schema.md            # Compilation schema
├── raw/
│   ├── _manifest.json      # Source metadata, hashes, converter info
│   ├── sources/            # Original ingested files
│   └── normalized/         # Canonical markdown/text artifacts
├── wiki/
│   ├── sources/            # Generated source pages
│   ├── concepts/           # Generated concept pages
│   ├── analysis/           # Saved analysis pages from kb ask --save
│   ├── index.md            # Wiki index (human-readable)
│   ├── _index.json         # Wiki index (machine-readable)
│   └── log.md              # Update activity log
├── graph/
│   └── exports/
│       ├── compile_runs.json      # Resume/failure state for update runs
│       └── search_index.sqlite3   # SQLite FTS5 chunk index
└── vault/
    └── obsidian/           # Obsidian-friendly export
        └── sources/
```

## Development

### Run tests

```bash
poetry run pytest tests -q
```

### Format code

```bash
poetry run black src tests
```

### Coverage report

```bash
poetry run pytest tests --cov=src --cov-report=term-missing
```

Coverage must stay at or above 97%.

### Real corpus smoke test

To exercise the full CLI against a real source corpus in a disposable project root:

```bash
poetry run python scripts/run_real_corpus_smoke.py \
    --raw-root path/to/raw-corpus \
    --project-root path/to/disposable-project
```

The script runs `help`, `init`, `status`, `add`, `status --changed`, `update`, `find`, `ask`, `lint`, `review`, and `export`, writes a consolidated log file under the disposable project root, and exits nonzero if any supported-source ingest fails, lint reports errors, or another command fails unexpectedly.

Unsupported files found under the raw corpus are probed separately to confirm they are rejected cleanly.

This is a manual smoke-test workflow, not a GitHub Actions dependency. The tracked pytest coverage only exercises the smoke-test script against a temporary corpus created inside the test itself.
