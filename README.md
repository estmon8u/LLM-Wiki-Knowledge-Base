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

# 2. Ingest some source documents
poetry run kb ingest path/to/paper.pdf
poetry run kb ingest path/to/notes.md
poetry run kb ingest path/to/slides.pptx

# 3. Compile the wiki
poetry run kb compile

# 4. Search the compiled wiki
poetry run kb search "knowledge base traceability"

# 5. Ask a question with citations
poetry run kb query "How does the wiki handle stale pages?"

# 6. Check wiki health
poetry run kb lint

# 7. See project status
poetry run kb status

# 8. Preview what needs compiling
poetry run kb diff

# 9. Export to an Obsidian-friendly vault
poetry run kb export-vault
```

## Global Options

| Option | Description |
| --- | --- |
| `--project-root PATH` | Run the CLI against a specific project directory instead of the current working directory. |
| `--verbose` | Enable verbose output. |
| `-h`, `--help` | Show help for any command. |

Example:

```bash
poetry run kb --project-root /path/to/project status
```

## Commands

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

### `kb ingest <source_path>`

Ingest and normalize a source file into the raw corpus.

```bash
poetry run kb ingest path/to/document.pdf
```

What happens:

1. The original file is copied into `raw/sources/`.
2. A canonical markdown or plain-text artifact is created in `raw/normalized/` (for non-text formats).
3. The source is registered in `raw/_manifest.json` with metadata including the content hash, converter used, and timestamps.

Duplicate detection: if you ingest the same file again, it will be detected and skipped.

### `kb compile`

Compile source pages, refresh the wiki index, and update the activity log.

```bash
poetry run kb compile
poetry run kb compile --force    # Rebuild every page, even unchanged ones
```

Reads the normalized artifacts from `raw/normalized/` (or directly from `raw/sources/` for markdown/text files) and generates source pages under `wiki/sources/`. Also updates `wiki/index.md`, `wiki/_index.json`, and `wiki/log.md`.

### `kb search <terms>`

Search the compiled wiki for relevant pages and snippets.

```bash
poetry run kb search "traceability citation"
poetry run kb search --limit 10 "agent architecture"
```

| Option | Default | Description |
| --- | --- | --- |
| `--limit` | 5 | Maximum number of results to return. |

Returns ranked search results with page title, file path, relevance score, and a snippet.

### `kb query <question>`

Answer a question from compiled wiki evidence with citations.

```bash
poetry run kb query "How does the wiki handle stale pages?"
poetry run kb query --limit 5 "What normalization converters are supported?"
```

| Option | Default | Description |
| --- | --- | --- |
| `--limit` | 3 | Maximum number of source pages to use as evidence. |

Returns an answer assembled from the best-matching wiki pages, followed by a Citations section listing which pages were used.

### `kb lint`

Run structural lint checks over the maintained wiki.

```bash
poetry run kb lint
```

Checks for:

- Broken internal links (`[[Missing Target]]` style)
- Orphan pages not referenced by the index
- Missing citations or provenance metadata
- Other structural issues

Exits with code 1 if any errors are found. Warnings and suggestions are printed but don't cause a nonzero exit.

### `kb status`

Show high-level project, corpus, and compile state.

```bash
poetry run kb status
```

Displays:

- `project_root` — current project directory
- `initialized` — whether `kb init` has been run
- `source_count` — number of ingested sources
- `compiled_source_count` — number of compiled source pages
- `concept_page_count` — number of concept pages
- `last_compile_at` — timestamp of the last compile

### `kb diff`

Show a pre-compile preview of source status: new (not yet compiled), changed (source modified since last compile), or up-to-date.

```bash
poetry run kb diff
```

Displays each source with a status tag:

- `[NEW]` — ingested but not yet compiled
- `[CHANGED]` — source content changed since last compile
- `[OK]` — compiled and up-to-date

Followed by summary counts for new, changed, and up-to-date sources.

### `kb export-vault`

Export the compiled wiki into the Obsidian-friendly vault folder.

```bash
poetry run kb export-vault
```

Copies compiled wiki pages into `vault/obsidian/` in a format compatible with [Obsidian](https://obsidian.md/).

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

## Project Layout

After initialization and a few ingested sources, the project directory looks like this:

```
project-root/
├── kb.config.yaml          # Project configuration
├── kb.schema.md            # Compilation schema
├── raw/
│   ├── _manifest.json      # Source metadata, hashes, converter info
│   ├── sources/            # Original ingested files
│   └── normalized/         # Canonical markdown/text artifacts
├── wiki/
│   ├── sources/            # Generated source pages
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

The script runs `help`, `init`, `status`, `ingest`, `diff`, `compile`, `search`, `query`, `lint`, and `export-vault`, writes a consolidated log file under the disposable project root, and exits nonzero if any supported-source ingest fails, lint reports errors, or another command fails unexpectedly.

Unsupported files found under the raw corpus are probed separately to confirm they are rejected cleanly.

This is a manual smoke-test workflow, not a GitHub Actions dependency. The tracked pytest coverage only exercises the smoke-test script against a temporary corpus created inside the test itself.
