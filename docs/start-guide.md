# Start Guide

This guide walks through the first successful `kb` project run. The README is
the command reference; this file is the step-by-step path from a fresh clone to
a searchable, citation-grounded wiki.

## 1. Install the CLI

Requirements:

| Tool | Required | Notes |
| --- | --- | --- |
| Python 3.11.x | Yes | The project is pinned to Python `>=3.11,<3.12`. |
| Poetry | Yes | Installs dependencies and runs the `kb` entrypoint. |
| LLM API key | Yes for `update`, `legacy ask`, `review`, and real GraphRAG index/query jobs | OpenAI, Anthropic, or Gemini. |
| Mistral API key | Required for PDFs, Office docs, images, and HTML OCR | Markdown and plain text do not need it. |
| wkhtmltopdf | Required only for HTML OCR | Must be on `PATH` or configured in `kb.config.yaml`. |

From the repository root:

```powershell
cd LLM-Wiki-Knowledge-Base
poetry install
poetry run kb --help
```

## 2. Create a project

Pick the directory that should hold your knowledge base:

```powershell
$projectRoot = "..\my-kb-project"
poetry run kb --project-root $projectRoot init
```

This creates the project files and directories:

| Path | Purpose |
| --- | --- |
| `kb.config.yaml` | Provider, conversion, and project settings. |
| `kb.schema.md` | Project-specific compilation guidance. |
| `raw/` | Original and normalized source files. |
| `wiki/` | Generated source, concept, analysis, index, and log pages. |
| `graph/` | Legacy search state plus the GraphRAG workspace. |
| `vault/` | Obsidian export output. |

The rest of this guide assumes you stay in the repo checkout and pass
`--project-root $projectRoot` to each command.

## 3. Configure providers

Choose one text provider:

```powershell
poetry run kb --project-root $projectRoot config provider set openai
```

Set the matching API key in your shell:

```powershell
$env:OPENAI_API_KEY = "..."
```

Other supported provider env vars are `ANTHROPIC_API_KEY` and
`GEMINI_API_KEY`. For Mistral-first conversion of PDFs, Office files, images,
and HTML, also set:

```powershell
$env:MISTRAL_API_KEY = "..."
```

Check the setup before adding sources:

```powershell
poetry run kb --project-root $projectRoot doctor
poetry run kb --project-root $projectRoot doctor --strict
```

## 4. Add sources

Add individual files or a folder:

```powershell
poetry run kb --project-root $projectRoot add path\to\paper.pdf
poetry run kb --project-root $projectRoot add path\to\notes.md
poetry run kb --project-root $projectRoot add path\to\research-folder
```

Supported common inputs include Markdown, text, PDF, HTML, DOCX, PPTX, images,
CSV, Excel, EPUB, and notebooks. `kb add` stores the original source and writes
normalized markdown when conversion is needed.

## 5. Build the wiki

Run update after adding sources:

```powershell
poetry run kb --project-root $projectRoot update
```

`kb update` compiles source pages, generates concept pages when useful,
refreshes `wiki/index.md` and `wiki/log.md`, and currently rebuilds the
temporary SQLite FTS5 search index for explicit legacy commands. During the
GraphRAG pivot, GraphRAG is the default retrieval target and retained FTS5
behavior is exposed only through deprecated `kb legacy ...` commands.

If a run is interrupted, resume it:

```powershell
poetry run kb --project-root $projectRoot update --resume
```

If a source changed and you want to refresh generated pages:

```powershell
poetry run kb --project-root $projectRoot update --force
```

## 6. Initialize and sync GraphRAG

After `kb update` has normalized and compiled the corpus, initialize the
GraphRAG workspace and sync the normalized artifacts into it:

```powershell
poetry run kb --project-root $projectRoot graph init
poetry run kb --project-root $projectRoot graph sync
```

This writes `graph/graphrag/input/sources.json` from `raw/_manifest.json` and
`raw/normalized/`, preserving source IDs, hashes, paths, converter metadata, and
the normalized text for GraphRAG indexing. The generated JSON file can contain
local corpus text and stays untracked.

Check readiness before running an index job:

```powershell
poetry run kb --project-root $projectRoot graph status
poetry run kb --project-root $projectRoot graph index --method fast --dry-run
```

Real `kb graph index` runs call the configured GraphRAG model and embedding
provider, so set `GRAPHRAG_API_KEY` or the local GraphRAG `.env` file before
running a non-dry-run index.

## 7. Search and ask

Deprecated legacy search returns matching wiki pages:

```powershell
poetry run kb --project-root $projectRoot legacy find "citation grounding"
poetry run kb --project-root $projectRoot legacy find --limit 10 "agent architecture"
```

Deprecated legacy ask uses source-page chunks as evidence and returns a cited answer:

```powershell
poetry run kb --project-root $projectRoot legacy ask "How does the wiki handle stale pages?"
poetry run kb --project-root $projectRoot legacy ask --show-evidence "What formats are supported?"
```

Save useful answers as analysis pages:

```powershell
poetry run kb --project-root $projectRoot legacy ask --save "What does the update pipeline do?"
poetry run kb --project-root $projectRoot legacy ask --save-as update-pipeline "What does the update pipeline do?"
```

Saved analysis pages are searchable with `kb legacy find`, but later legacy ask
runs do not cite saved answers or generated concept pages as primary evidence.
Top-level `kb find` and `kb ask` are reserved for GraphRAG behavior and now fail
with guidance until graph querying is implemented.

## 8. Check and export

Run structural checks:

```powershell
poetry run kb --project-root $projectRoot lint
```

Run semantic review:

```powershell
poetry run kb --project-root $projectRoot review
```

Inspect project state:

```powershell
poetry run kb --project-root $projectRoot status
poetry run kb --project-root $projectRoot status --changed
```

Export an Obsidian-compatible vault:

```powershell
poetry run kb --project-root $projectRoot export
poetry run kb --project-root $projectRoot export --clean
```

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `Provider is not configured` | Run `kb config provider set openai`, `anthropic`, or `gemini`. |
| Provider authentication errors | Confirm the matching API key environment variable is set in the same shell. |
| PDF, DOCX, PPTX, image, or HTML conversion fails | Set `MISTRAL_API_KEY`; for HTML also install/configure `wkhtmltopdf`. |
| Search returns stale results | Run `kb update` after adding or changing sources. |
| GraphRAG workspace is missing | Run `kb graph init`. |
| GraphRAG input is missing | Run `kb graph sync` after `kb update`. |
| GraphRAG output is missing | Run `kb graph index --method fast --dry-run`, then a real index when provider credentials and cost are acceptable. |
| Generated pages look stale | Run `kb status --changed`, then `kb update --force` if needed. |

## Next Steps

After the first successful run, use the README for the full command reference
and `docs/architecture/` for implementation architecture.
