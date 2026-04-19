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

# 2. Check project health (provider, converters, structure)
poetry run kb doctor

# 3. Ingest some source documents (`kb add` is a friendlier alias)
poetry run kb add path/to/paper.pdf
poetry run kb ingest path/to/notes.md
poetry run kb add path/to/slides.pptx
poetry run kb add path/to/research-folder

# 4. Compile the wiki (add --with-concepts to generate concept pages)
poetry run kb compile
poetry run kb compile --with-concepts

# 5. Search the compiled wiki
poetry run kb query search "knowledge base traceability"

# 6. Ask a question with citations
poetry run kb query ask "How does the wiki handle stale pages?"
poetry run kb query ask --self-consistency 3 "What normalization converters are supported?"
# → After showing the answer, you'll be prompted to save it as an analysis page

# 7. Check wiki health (deterministic structural checks)
poetry run kb check lint

# 8. Run semantic review for contradictions and terminology drift
poetry run kb check review
poetry run kb check review --adversarial

# 9. See project status
poetry run kb show status

# 10. Preview what needs compiling
poetry run kb show diff

# 11. Export to an Obsidian-friendly vault (--clean removes stale files)
poetry run kb export vault
poetry run kb export vault --clean
```

Before running `kb compile`, `kb query`, or `kb review`, add a `provider`
section to `kb.config.yaml` and set the matching API key environment variable.
See `Provider Configuration` below.

## Global Options

| Option | Description |
| --- | --- |
| `--project-root PATH` | Run the CLI against a specific project directory instead of the current working directory. |
| `--verbose` | Enable verbose output. |
| `-h`, `--help` | Show help for any command. |

Example:

```bash
poetry run kb --project-root /path/to/project show status
```

## Commands

Commands are organized into flat verbs and namespaced groups:

| Group | Subcommands | Description |
| --- | --- | --- |
| *(flat)* | `init`, `add`, `ingest`, `compile`, `doctor` | Core workflow verbs |
| `query` | `search`, `ask` | Search the wiki or ask provider-backed questions |
| `check` | `lint`, `review` | Structural and semantic quality checks |
| `show` | `status`, `diff` | Project state inspection |
| `export` | `vault` | Export to external formats |

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

### `kb doctor`

Run health checks on the project: structure, config, provider, API keys, converters, and run-artifact database.

```bash
poetry run kb doctor
```

Prints formatted health-check sections with `[OK]` or `[FAIL]` entries and exits with code 1 if any check fails.

### `kb ingest <source_path>` / `kb add <source_path>`

Ingest and normalize a source file into the raw corpus, or recursively ingest a directory of supported source files.

```bash
poetry run kb add path/to/document.pdf
poetry run kb ingest path/to/document.pdf
poetry run kb add path/to/research-folder
```

`kb add` is a first-class alias for `kb ingest`. Both commands use the same
normalization, duplicate-detection, and manifest-registration path.

What happens:

1. The original file is copied into `raw/sources/`.
2. A canonical markdown or plain-text artifact is created in `raw/normalized/` (for non-text formats).
3. The source is registered in `raw/_manifest.json` with metadata including the content hash, converter used, and timestamps.

Duplicate detection: if you ingest the same file again, it will be detected and skipped.

When the input is a directory, the command walks the directory tree recursively by default, ingests only supported file types, and prints a batch summary showing how many files were created or skipped as duplicates. Unsupported files inside the directory are ignored.
Interactive terminals show directory-ingest progress; non-interactive runs print a simple `Ingesting N source file(s)...` preamble before the summary.

### `kb compile`

Compile source pages, refresh the wiki index, and update the activity log. Requires a configured provider — summaries are generated by the provider, not extracted heuristically.

```bash
poetry run kb compile
poetry run kb compile --force           # Rebuild every page, even unchanged ones
poetry run kb compile --with-concepts   # Also generate concept pages and backlinks
```

| Option | Default | Description |
| --- | --- | --- |
| `--force` | off | Rebuild every source page even if nothing changed. |
| `--with-concepts` | off | After compiling, generate concept pages in `wiki/concepts/` by grouping related source pages and maintain managed backlink sections in source pages. |

Reads the normalized artifacts from `raw/normalized/` (or directly from `raw/sources/` for markdown/text files) and generates source pages under `wiki/sources/`. Each source page summary is produced by sending document content to the configured provider. Also updates `wiki/index.md`, `wiki/_index.json`, and `wiki/log.md`.
Interactive terminals show compile progress for source pages; non-interactive runs print a simple `Compiling N source page(s)...` preamble before the compile summary.

When `--with-concepts` is passed, the concept service scans all compiled source pages, groups related pages by term overlap (Jaccard similarity), generates deterministic concept pages with `type: concept` frontmatter, and inserts managed backlink sections into source pages. Stale generated concept pages are automatically removed.

### `kb query search <terms>`

Search the compiled wiki for relevant pages and snippets.

```bash
poetry run kb query search "traceability citation"
poetry run kb query search --limit 10 "agent architecture"
```

| Option | Default | Description |
| --- | --- | --- |
| `--limit` | 5 | Maximum number of results to return. |

Returns ranked search results with page title, file path, relevance score, and a snippet.

### `kb query ask <question>`

Answer a question from compiled wiki evidence with provider-backed synthesis and citations.

```bash
poetry run kb query ask "How does the wiki handle stale pages?"
poetry run kb query ask --limit 5 "What normalization converters are supported?"
poetry run kb query ask --self-consistency 3 "How is traceability preserved?"
```

| Option | Default | Description |
| --- | --- | --- |
| `--limit` | 3 | Maximum number of source pages to use as evidence. |
| `--self-consistency` | 1 | Sample N independent provider answers from the same frozen evidence bundle and merge claims deterministically. |

`kb query ask` requires a configured provider. It retrieves the best-matching wiki pages as evidence, sends that evidence to the configured provider, and prints the provider answer followed by a Citations section listing which pages were used. The output begins with a `[mode: …]` tag — `provider:<model>` when a single LLM call synthesizes the answer, or `self-consistency:<model>:N` when multiple provider samples are merged.

When `--self-consistency N` is greater than 1, `kb query ask` retrieves evidence once, freezes that evidence bundle, samples `N` independent provider answers in parallel, normalizes them into typed claims, merges near-duplicate grounded claims deterministically, and stores the full run artifact in `graph/exports/run_artifacts.sqlite3`. If the provider is missing or any provider call fails, the command exits with an error instead of falling back.

After displaying the answer, if citations are present the command prompts `Save this answer as an analysis page? [y/N]`. Accepting writes a markdown analysis page to `wiki/concepts/` with YAML frontmatter (`type: analysis`), the question, a timestamp, and backlinks to the cited source pages. Saved analysis pages make your explorations compound in the wiki instead of disappearing in terminal history. In non-interactive contexts the prompt is silently skipped.

### `kb check lint`

Run structural lint checks over the maintained wiki.

```bash
poetry run kb check lint
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

### `kb show status`

Show high-level project, corpus, and compile state.

```bash
poetry run kb show status
```

Displays:

- `project_root` — current project directory
- `initialized` — whether `kb init` has been run
- `source_count` — number of ingested sources
- `compiled_source_count` — number of compiled source pages
- `concept_page_count` — number of concept pages
- `last_compile_at` — timestamp of the last compile

### `kb show diff`

Show a pre-compile preview of source status: new (not yet compiled), changed (source modified since last compile), or up-to-date.

```bash
poetry run kb show diff
```

Displays each source with a status tag:

- `[NEW]` — ingested but not yet compiled
- `[CHANGED]` — source content changed since last compile (either through re-ingest or manual file edits)
- `[OK]` — compiled and up-to-date

Followed by a formatted summary section with counts for new, changed, and up-to-date sources.

`diff` recomputes hashes from the actual normalized files on disk, so it detects manual edits to normalized files that would not be visible from manifest metadata alone.

### `kb check review`

Run semantic review checks for contradictions and terminology drift across the maintained wiki.

```bash
poetry run kb check review
poetry run kb check review --adversarial
```

| Option | Default | Description |
| --- | --- | --- |
| `--adversarial` | off | Run extractor, skeptic, and arbiter review over candidate source-page pairs and persist a review run artifact. |

By default, `kb review` runs deterministic overlap and terminology-variant checks plus a single-pass model-backed review. With `--adversarial`, the command keeps the deterministic checks, then builds candidate source-page pairs, runs extractor/skeptic/arbiter prompts over each pair, emits typed findings, and stores the run in `graph/exports/run_artifacts.sqlite3`. `kb review` requires a configured provider; provider failures stop the command instead of falling back.

Checks for:

- **Overlapping topics** — Source pages with heavily overlapping terminology that may benefit from a shared concept page.
- **Terminology variants** — The same root term appearing in different forms across pages (e.g., `knowledge-base` vs `knowledgebase`).
- **Adversarial findings** — Typed contradiction, term-drift, and needs-review findings from extractor/skeptic/arbiter evaluation over candidate page pairs.

This is the semantic complement to `kb check lint`. Lint checks structural health deterministically; review checks content-level coherence through heuristics, a single provider pass, or the adversarial review pipeline.

### `kb export vault`

Export the compiled wiki into the Obsidian-friendly vault folder.

```bash
poetry run kb export vault
poetry run kb export vault --clean
```

| Option | Default | Description |
| --- | --- | --- |
| `--clean` | off | Remove stale vault files that no longer exist in the wiki. |

Copies compiled wiki pages into `vault/obsidian/` in a format compatible with [Obsidian](https://obsidian.md/).
Open that folder in Obsidian with `Open folder as vault`. Treat `vault/obsidian/`
as derived output: running `kb export vault` copies files from `wiki/` again and
can overwrite direct edits made inside the exported vault.

With `--clean`, any markdown files in the vault that no longer correspond to a
wiki page are deleted automatically.

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

`kb compile`, `kb query`, and `kb review` require a configured provider.
If the provider is missing, those commands fail with a configuration error. If a
configured provider call fails, they fail instead of falling back. Configure the
provider in `kb.config.yaml`:

```yaml
provider:
  name: openai          # openai | anthropic | gemini
  model: gpt-5.4-mini   # optional — defaults to a cost-effective model per provider
```

You can also override the provider per-invocation without editing the config file:

```bash
kb --provider anthropic query ask "How does REALM differ from RAG?"
kb --provider gemini check review --adversarial
```

This makes it easy to compare all three providers against the same compiled wiki without maintaining separate project directories.

Set the matching API key as an environment variable:

| Provider | Default model | Env variable | Alternatives |
| --- | --- | --- | --- |
| `openai` | `gpt-5.4-mini` | `OPENAI_API_KEY` | `gpt-5.4`, `gpt-5.4-nano` |
| `anthropic` | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` | `claude-opus-4-6`, `claude-haiku-4-5` |
| `gemini` | `gemini-3.1-flash-lite-preview` | `GEMINI_API_KEY` | `gemini-3.1-pro-preview`, `gemini-2.5-flash` |

If the provider is not configured, `kb compile`, `kb query`, and `kb review`
fail with a configuration error. If the provider is configured but the API key
is missing or the provider call fails, those commands fail instead of falling
back.

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
│   ├── concepts/           # Saved analysis pages and future concept pages
│   ├── index.md            # Wiki index (human-readable)
│   ├── _index.json         # Wiki index (machine-readable)
│   └── log.md              # Compile activity log
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

Coverage must stay at or above 98%.

### Real corpus smoke test

To exercise the full CLI against a real source corpus in a disposable project root:

```bash
poetry run python scripts/run_real_corpus_smoke.py \
    --raw-root path/to/raw-corpus \
    --project-root path/to/disposable-project
```

The script runs `help`, `init`, `show status`, `ingest`, `show diff`, `compile`, `query search`, `query ask`, `check lint`, `check review`, and `export vault`, writes a consolidated log file under the disposable project root, and exits nonzero if any supported-source ingest fails, lint reports errors, or another command fails unexpectedly.

Unsupported files found under the raw corpus are probed separately to confirm they are rejected cleanly.

This is a manual smoke-test workflow, not a GitHub Actions dependency. The tracked pytest coverage only exercises the smoke-test script against a temporary corpus created inside the test itself.
