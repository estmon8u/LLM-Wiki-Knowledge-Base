# Start Guide

This guide walks through the first successful `kb` project run. The README is
the command reference; this file is the step-by-step path from a fresh clone to
a searchable, citation-grounded wiki.

## 1. Install the CLI

Requirements:

| Tool | Required | Notes |
| --- | --- | --- |
| Python 3.11 or 3.12 | Yes | The project is pinned to Python `>=3.11,<3.13` while Microsoft GraphRAG documents Python 3.10-3.12 support. |
| Poetry | Yes | Installs dependencies and runs the `kb` entrypoint. |
| LLM API key | Yes for `kb update`, `kb ask --engine legacy`, `kb review`, and real GraphRAG index/query jobs | Normal `kb update` warns and skips graph indexing if GraphRAG credentials are missing; `kb update --graph-only` requires them. |
| Mistral API key | Required for PDFs, Office docs, images, and HTML OCR | Markdown and plain text do not need it. |
| HTML renderer | Required only for HTML OCR | `wkhtmltopdf` is preferred when installed; bundled `xhtml2pdf` is the pure-Python fallback. |

From the repository root:

```powershell
cd LLM-Wiki-Knowledge-Base
poetry install --with dev --all-extras
poetry run kb --help
```

For package installs outside the repo, choose the extras you need, such as
`graphwiki-kb[openai]`, `graphwiki-kb[agent]`, `graphwiki-kb[pdf]`,
`graphwiki-kb[wikigraph]`, `graphwiki-kb[wikigraph-eval]`, or
`graphwiki-kb[all]`.

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

OpenAI response storage is disabled by default through
`providers.openai.store_responses: false`; only set it to `true` when you
explicitly want provider-side response retention for your account.

`kb init` creates the project-local GraphRAG workspace and syncs the managed
provider, model, embedding, API-key, chunking, technical extraction, and
GraphRAG input-safety fields into
`graph/graphrag/settings.yaml`. Later `kb init` or `kb update` runs refresh
those managed fields while preserving unrelated user-owned GraphRAG tuning such
as cache, vector-store, and search settings. The default graph extraction entity
types are tuned for technical corpora (`concept`, `technology`, `method`,
`algorithm`, `dataset`, `model`, `benchmark`, `framework`, `component`, `api`,
`paper`, and `claim`), and `graph.input.max_source_bytes` rejects unexpectedly
large normalized sources before GraphRAG input sync reads them.

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
`kb ask` defaults to WikiGraphRAG. Microsoft GraphRAG remains explicit with
`kb ask --engine graphrag`, and deprecated FTS comparison behavior is exposed
only through `kb find/ask --engine legacy`. WikiGraphRAG build defaults live in
the typed `wikigraph` config section, optional NetworkX-backed dependencies load
lazily, and `--export-wikigraph-artifacts` writes generated entity, community,
chunk, and TextUnit cards under `wiki/wikigraph/` in classic mode, or entity,
relation, source-chunk, index, and diagnostics pages in LightRAG mode, without
allowing those generated cards to feed back into the next graph build. Unless
`--no-wikigraph-normalized-text`
or config disables it, WikiGraphRAG creates source-document and TextUnit nodes
from `raw/normalized/`, persists `documents.json` and `text_units.json`, and
reports the included document and TextUnit counts in the update summary.

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
active complete output directory. A complete output directory includes readable
required GraphRAG Parquet tables and the configured vector store in a ready
state, but status and ask still require the current input, source hashes,
settings, prompts, and
GraphRAG runtime identity to match the last successful index run. The graph step
first plans sync work without mutating workspace files. When graph work proceeds, it writes
compact `graph/graphrag/input/sources.json`
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
successful run. Use `--graph-method auto|standard|fast|standard-update|fast-update`
only when you need to override that planner.
When complete graph output already exists and indexing is skipped, `kb update`
still refreshes `wiki/graph/` from the active output directory recorded by the
latest successful complete run.

Check readiness after update:

```powershell
poetry run kb --project-root $projectRoot status
poetry run kb --project-root $projectRoot status --strict
```

`status --strict` exits non-zero unless the wiki sources are compiled and the
GraphRAG index is complete, fresh, and query-ready. Status distinguishes corrupt
run metadata, unreadable or dependency-missing Parquet tables, and vector-store
health instead of collapsing those states into "not ready."

Real GraphRAG index actions call GraphRAG's installed Python entrypoints through
a signature-aware adapter with documented CLI fallback for entrypoint contract
drift, and use the configured GraphRAG model and embedding provider. Set the
provider API key such as `OPENAI_API_KEY`, or put the same variable in the local
GraphRAG `.env` file, before running graph indexing, `kb ask --engine graphrag`, or
`kb update --graph-only`. A normal `kb update` without GraphRAG
credentials still compiles the wiki and reports graph indexing as skipped.
Interactive terminals show a live indexing status spinner while GraphRAG runs.
After indexing, the command prints the active GraphRAG output path; if legacy
SQLite FTS5 refresh is unavailable, it also prints the markdown-scan fallback
warning for `kb find`.
Use `kb update --force` for a full source-page and GraphRAG rebuild after
model/prompt changes or suspected corrupt output. Use `kb update --no-graph`
only when you want to refresh the wiki and legacy index without touching
GraphRAG. Bundled GraphRAG prompt templates are copied only when missing; if a
bundled prompt changes after you tune a workspace prompt, the new default is
written beside it as `*.new` instead of overwriting your tuned prompt.

## 7. Search and ask

Search direct GraphRAG entity/relationship artifacts, WikiGraphRAG contexts,
the maintained wiki index, or the deprecated source-page legacy path:

```powershell
poetry run kb --project-root $projectRoot find "citation grounding"
poetry run kb --project-root $projectRoot find --limit 10 "agent architecture"
poetry run kb --project-root $projectRoot find --engine legacy "citation grounding"
```

After a real `kb update`, ask through the default WikiGraphRAG backend. Use
`--engine graphrag`, comma-separated engines, or `--engine all` when you want a
comparison run:

```powershell
poetry run kb --project-root $projectRoot ask "What are the main retrieval design patterns?"
poetry run kb --project-root $projectRoot ask --method global "What are the main retrieval design patterns?"
poetry run kb --project-root $projectRoot ask --engine graphrag --method drift --save "Compare RAG, REALM, FiD, Self-RAG, and GraphRAG."
poetry run kb --project-root $projectRoot ask --engine all "How does REALM differ from RAG?"
```

Do not use the deprecated top-level `--limit` flag with `kb ask`; answer
backends reject it because source-page evidence limiting only applies to
`kb ask --engine legacy`.

The default `--method auto` router is backend-specific. WikiGraphRAG supports
`basic`, `local`, `global`, and `drift-lite`. WikiGraphRAG auto-routing uses
question intent: comparison words such as `compare`, `differ`, `vs`, or
`contrast` choose `drift-lite`; corpus-wide phrases such as `main themes`,
`across`, or `whole corpus` choose `global`; matched entities choose `local`;
otherwise it falls back to `basic`. Microsoft
GraphRAG supports `basic`, `local`, `global`, and `drift` through
`--engine graphrag`. Neither graph-backed path silently falls back to FTS5; the
legacy comparator must be requested with `--engine legacy`. Saved graph answers
go to `wiki/analysis/` with graph metadata, source trace, support level, and
unique filenames on repeated saves. Multi-engine saves prefix the slug per
engine so outputs do not collide.

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

Deprecated legacy search returns matching source pages through the unified
`find` command:

```powershell
poetry run kb --project-root $projectRoot find --engine legacy "citation grounding"
poetry run kb --project-root $projectRoot find --engine legacy --limit 10 "agent architecture"
```

Deprecated legacy search and ask are source-page-only comparators. Legacy ask
uses source-page chunks as evidence and returns a cited answer through the
unified `ask` command:

```powershell
poetry run kb --project-root $projectRoot ask --engine legacy "How does the wiki handle stale pages?"
poetry run kb --project-root $projectRoot ask --engine legacy --show-source-trace "What formats are supported?"
```

Save useful legacy comparison answers as analysis pages:

```powershell
poetry run kb --project-root $projectRoot ask --engine legacy --save-as update-pipeline "What does the update pipeline do?"
```

Saved analysis pages are searchable with top-level `kb find`, but `kb find --engine legacy` and later `kb ask --engine legacy` runs stay source-only so saved answers, generated
concept pages, and direct graph artifacts are not treated as primary evidence.

Optional natural-language control is available through `kb agent` when the
agent extra and `OPENAI_API_KEY` are installed. It routes through the same typed
services as the direct commands, so local KB answers can use GraphRAG or
WikiGraphRAG through `ask_kb`, searches can use the same fused `find_kb`
engines as `kb find`, and research recommendations are stored separately from
ingestion:

```powershell
poetry run kb --project-root $projectRoot agent "What does my KB say about GraphRAG evaluation?"
poetry run kb --project-root $projectRoot agent "Research current GraphRAG evaluation work and recommend sources"
poetry run kb --project-root $projectRoot agent "show previous recommendations"
poetry run kb --project-root $projectRoot agent --yes "add recommendation 1 and update the KB"
poetry run kb --project-root $projectRoot agent
```

One-shot `kb agent "..."` calls are sessionless unless you pass `--session ID`.
Interactive mode uses a persistent `repl` session. Write tools such as
recommendation ingestion and `update_kb` require an approval prompt or `--yes`;
research never ingests a web source automatically. "Show previous
recommendations" reads the latest saved run that actually has recommendations,
so an empty follow-up research run does not hide the numbered sources a user is
preparing to ingest. When asked to compare Microsoft GraphRAG and WikiGraphRAG,
the agent calls `ask_kb` once per engine instead of substituting one backend for
the other.

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
| WikiGraphRAG reports that NetworkX is missing | Install `graphwiki-kb[wikigraph]` or use the repository development install with all extras. WikiGraphRAG dependencies load lazily, so unrelated non-WikiGraphRAG commands do not need NetworkX. |
| Update warns about missing normalized graph input | Run `kb lint` to identify the stale manifest entry, then re-add or remove the affected source. |
| Generated pages look stale | Run `kb status --changed`, then `kb update --force` if needed. |
| `kb agent` says `openai-agents` is not installed | Install the optional agent extra with `poetry install -E agent` or use the repository development install with all extras. |

## Next Steps

After the first successful run, use the README for the full command reference
and `docs/architecture/` for implementation architecture.
