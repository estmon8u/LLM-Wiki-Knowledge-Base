# Start Guide

This guide walks through the first successful `kb` project run. The README is
the command reference; this file is the step-by-step path from a fresh clone to
a searchable, citation-grounded wiki.

## 1. Install the CLI

Requirements:

| Tool | Required | Notes |
| --- | --- | --- |
| Python 3.11, 3.12, or 3.13 | Yes | The project is pinned to Python `>=3.11,<3.14`. |
| Poetry | Yes | Installs dependencies and runs the `kb` entrypoint. |
| LLM API key | Yes for `kb update`, `kb legacy ask`, `kb review`, and real GraphRAG index/query jobs | Normal `kb update` warns and skips graph indexing if GraphRAG credentials are missing; `kb update --graph-only` requires them. |
| Mistral API key | Required for PDFs, Office docs, images, and HTML OCR | Markdown and plain text do not need it. |
| HTML renderer | Required only for HTML OCR | `wkhtmltopdf` is preferred when installed; bundled `xhtml2pdf` is the pure-Python fallback. |

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
| `graph/` | GraphRAG workspace plus local search and run state. |
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

Keep `MISTRAL_API_KEY` set for real PDF work. PDFs use the paid Mistral OCR API
as the primary converter because OCR quality drives the compiled source pages,
GraphRAG entities, retrieval, citations, and answers. Local PDF converters are
fallbacks for outage or setup recovery, not equal-quality defaults.

GraphRAG runtime settings live in the `graph` section of `kb.config.yaml`.
The default graph provider is OpenAI with `gpt-5.4-nano`,
`text-embedding-3-small`, and the centralized `OPENAI_API_KEY` provider entry.
Set that key before running a real graph index or query job:

```powershell
$env:OPENAI_API_KEY = "..."
```

`kb init` creates the project-local GraphRAG workspace and syncs the managed
provider, model, embedding, and API-key fields into
`graph/graphrag/settings.yaml`. Later `kb init` or `kb update` runs refresh
those managed fields while preserving user-owned GraphRAG tuning such as
chunking, cache, vector-store, and search settings.

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
refreshes `wiki/index.md` and `wiki/log.md`, rebuilds the maintained wiki search
index used by `kb find`, and refreshes the deprecated legacy comparator index.
During the GraphRAG pivot, GraphRAG is the default answer path and deprecated
FTS comparison commands are exposed only through `kb legacy ...`.

If a run is interrupted, resume it:

```powershell
poetry run kb --project-root $projectRoot update --resume
```

If a source changed and you want to refresh generated pages:

```powershell
poetry run kb --project-root $projectRoot update --force
```

## 6. Check GraphRAG state

After `kb update`, GraphRAG state is determined from the planned or synced
input, the latest successful index run, digest-based freshness metadata, and the
active complete output directory. A complete output directory includes the
required GraphRAG Parquet tables and the configured vector store, but status and
ask still require the current input, source hashes, settings, prompts, and
GraphRAG runtime identity to match the last successful index run. The graph step
first plans sync work without mutating workspace files. When graph work proceeds, it writes
`graph/graphrag/input/sources.json`
from `raw/_manifest.json` and `raw/normalized/`, preserving source IDs, hashes,
paths, converter metadata, and the normalized text for GraphRAG indexing. The
generated JSON file can contain local corpus text and stays untracked.
During a normal update, isolated missing normalized artifacts are skipped with a
warning instead of blocking every source from syncing into GraphRAG. Run
`kb lint` after that warning to identify the stale manifest entry and repair or
remove it. `kb update --graph-only` remains strict because there is no wiki-only
work to complete.

The same graph decision checks whether the GraphRAG index needs a full `fast`
rebuild, an incremental `fast-update`, a retry after the latest failed attempt,
or no index job because sources and runtime settings already match the last
successful run.
When complete graph output already exists and indexing is skipped, `kb update`
still refreshes `wiki/graph/` from the active output directory recorded by the
latest successful complete run.

Check readiness after update:

```powershell
poetry run kb --project-root $projectRoot status
```

Real GraphRAG index actions call GraphRAG's installed Python entrypoints through
a signature-aware adapter and use the configured GraphRAG model and embedding
provider, so set the provider API key such as `OPENAI_API_KEY`, or put the same
variable in the local GraphRAG `.env` file, before running graph indexing,
`kb ask`, or `kb update --graph-only`. A normal `kb update` without GraphRAG
credentials still compiles the wiki and reports graph indexing as skipped.
Interactive terminals show a live indexing status spinner while GraphRAG runs.
After indexing, the command prints the active GraphRAG output path; if legacy
SQLite FTS5 refresh is unavailable, it also prints the markdown-scan fallback
warning for `kb find`.
Use `kb update --force` for a full source-page and GraphRAG rebuild after
model/prompt changes or suspected corrupt output. Use `kb update --no-graph`
only when you want to refresh the wiki and legacy index without touching
GraphRAG.

## 7. Search and ask

Search direct GraphRAG entity/relationship artifacts plus the maintained wiki
index for source, analysis, concept, and generated graph pages:

```powershell
poetry run kb --project-root $projectRoot find "citation grounding"
poetry run kb --project-root $projectRoot find --limit 10 "agent architecture"
```

After a real `kb update`, ask through the default GraphRAG controller:

```powershell
poetry run kb --project-root $projectRoot ask "What are the main retrieval design patterns?"
poetry run kb --project-root $projectRoot ask --method global "What are the main retrieval design patterns?"
poetry run kb --project-root $projectRoot ask --method drift --save "Compare RAG, REALM, FiD, Self-RAG, and GraphRAG."
```

Do not use the deprecated top-level `--limit` flag with `kb ask`; GraphRAG
answers reject it because source-page evidence limiting only applies to
`kb legacy ask`.

The default `--method auto` router uses question wording and a capped scan of
known graph terms to choose Basic, Local, Global, or DRIFT. It does not fall
back to FTS5 if the graph is missing or not ready; it fails with the next
GraphRAG setup command to run. Readiness is checked per query method, so a
global question can run from community reports while local/basic/drift questions
still require the vector store and their method-specific tables. Non-streaming
GraphRAG answers are preserved even when the underlying entrypoint returns the
answer instead of printing it.

Use `local` for specific entity, method, or paper questions; `global` for
whole-corpus themes; `drift` for multi-paper comparisons; and `basic` as the
simple vector-RAG baseline. Saved graph answers go to `wiki/analysis/` with
graph metadata, source trace, support level, separate raw stdout/stderr audit
sections, and unique filenames on repeated saves. Blank GraphRAG answers are
not saved.

Export graph artifacts into inspectable wiki pages:

```powershell
poetry run kb --project-root $projectRoot export
```

This creates `wiki/graph/index.md` plus generated pages for GraphRAG documents,
entities, relationships, communities, and text units. Raw document and text-unit
content is fenced so paper-internal markdown is inspectable without creating
wiki lint failures. Large relationship tables are counted in the graph index,
but only the strongest relationship pages are materialized. Existing
`wiki/concepts/` pages stay in place as legacy concept pages.

Deprecated legacy search returns matching wiki pages:

```powershell
poetry run kb --project-root $projectRoot legacy find "citation grounding"
poetry run kb --project-root $projectRoot legacy find --limit 10 "agent architecture"
```

Deprecated legacy search and ask are source-page-only comparators. Legacy ask
uses source-page chunks as evidence and returns a cited answer:

```powershell
poetry run kb --project-root $projectRoot legacy ask "How does the wiki handle stale pages?"
poetry run kb --project-root $projectRoot legacy ask --show-evidence "What formats are supported?"
```

Save useful legacy comparison answers as analysis pages:

```powershell
poetry run kb --project-root $projectRoot legacy ask --save "What does the update pipeline do?"
poetry run kb --project-root $projectRoot legacy ask --save-as update-pipeline "What does the update pipeline do?"
```

Saved analysis pages are searchable with top-level `kb find`, but `kb legacy
find` and later legacy ask runs stay source-only so saved answers, generated
concept pages, and direct graph artifacts are not treated as primary evidence.

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

`--clean` deletes vault markdown only after building the current export set, so
cleanup uses the exact destination paths exported by that run.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `Provider is not configured` | Run `kb config provider set openai`, `anthropic`, or `gemini`. |
| Provider authentication errors | Confirm the matching API key environment variable is set in the same shell. |
| PDF, DOCX, PPTX, image, or HTML conversion fails | Set `MISTRAL_API_KEY`; for HTML confirm `wkhtmltopdf` or bundled `xhtml2pdf` is available. If PDF metadata shows `fallback_used: true`, rerun with Mistral before trusting downstream graph or answer quality. |
| Search returns stale results | Run `kb update` after adding or changing sources. |
| GraphRAG workspace is missing | Run `kb init`. |
| GraphRAG input is missing | Run `kb update`. |
| GraphRAG output or vector store is missing, empty, unreadable, or incompatible | Set graph provider credentials, then run `kb update`; a normal update without them only refreshes the wiki and warns. |
| Update warns about missing normalized graph input | Run `kb lint` to identify the stale manifest entry, then re-add or remove the affected source. |
| Generated pages look stale | Run `kb status --changed`, then `kb update --force` if needed. |

## Next Steps

After the first successful run, use the README for the full command reference
and `docs/architecture/` for implementation architecture.
