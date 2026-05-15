# GraphRAG Pivot

Date: 2026-05-11

## 1. Why the pivot is necessary

The current project already has useful research-memory infrastructure: source ingestion, normalized markdown or plain-text artifacts, manifest metadata, provenance-aware source pages, saved analysis pages, structural linting, provider-backed review, vault export, and a rebuildable SQLite FTS5 search index.

The weak point is retrieval. The deprecated `kb legacy find` path is centered on lexical SQLite FTS5 search, and `kb legacy ask` retrieves source-page chunks through that search path before asking a provider to synthesize an answer. That legacy path can work for direct factual questions when the right source chunk is retrieved, but it is not enough for cross-paper comparison, whole-corpus themes, or questions that need synthesis across related methods.

GraphRAG is a better fit for that gap because it indexes raw text into a knowledge graph, detects communities, summarizes those communities, and supports query modes for entity-specific questions, whole-dataset questions, DRIFT-style mixed global/local retrieval, and basic vector-RAG comparison. The pivot keeps the maintainable wiki system, but moves retrieval and synthesis toward GraphRAG.

The project should now be described as:

```text
CLI-first GraphRAG research-memory system for ingesting technical documents,
building a graph-based retrieval index, answering local/global research questions,
and exporting inspectable wiki artifacts with provenance and citations.
```

Keep this sentence in project materials:

```text
The wiki is not the retrieval engine. The wiki is the human-readable artifact layer.
GraphRAG is the retrieval and synthesis engine.
```

## 2. What stays from the old system

- Raw source files remain the source of truth.
- Normalized markdown or plain-text artifacts remain the stable bridge between heterogeneous inputs and downstream processing.
- `raw/_manifest.json` remains the provenance ledger for source IDs, hashes, converter metadata, and freshness checks.
- `kb update` remains responsible for wiki artifact maintenance and may continue refreshing the temporary FTS index while legacy commands exist.
- `wiki/sources/`, legacy `wiki/concepts/`, `wiki/graph/`, `wiki/analysis/`, `wiki/index.md`, and `wiki/log.md` remain inspectable artifacts.
- FTS5 behavior is retained only behind explicit `kb legacy find` and `kb legacy ask` commands with deprecation warnings; both legacy commands stay source-page-only comparators.
- `kb lint`, `kb review`, `kb status`, `kb doctor`, and `kb export` remain maintenance and operations surfaces.
- Provider configuration, conversion routing, and Windows/corporate TLS setup remain part of the current project support story.

## 3. What changes

- Retrieval moves from lexical/wiki-first to GraphRAG-first.
- SQLite FTS5 is not a peer default. It is retained only as an explicit deprecated legacy path for comparison or exact lookup unless it is removed later.
- GraphRAG must not silently fall back to FTS5 when the graph index, output tables, or vector store are missing or stale; it should fail with clear next-step guidance such as `kb update`.
- A dedicated GraphRAG workspace now lives under `graph/graphrag/` and is initialized through GraphRAG's installed Python initialization entrypoint before project settings are patched in.
- Normalized corpus artifacts are synced into GraphRAG JSON input with source metadata attached through `kb update`; the same command now auto-decides whether to run a full `fast` index, an incremental `fast-update`, retry after the latest failed attempt, or skip when sources and runtime settings match the last successful index.
- `kb init`, `kb update`, `kb status`, `kb ask`, and `kb export` now own the GraphRAG lifecycle. The earlier `kb graph` command group was removed in Phase 9 so the default UX is the main command surface.
- GraphRAG indexing creates graph outputs such as text units, entities, relationships, communities, community reports, and the vector store needed by query modes.
- GraphRAG query modes are now first-class CLI behavior under `kb ask --method`:
  - `basic` for simple vector-RAG comparison,
  - `local` for specific entity, paper, or method questions,
  - `global` for whole-corpus themes,
  - `drift` for comparison questions that benefit from global context plus local refinement.
- Top-level `kb ask` is now the default GraphRAG-aware answer controller. It checks graph readiness, chooses `basic`, `local`, `global`, or `drift` with deterministic routing unless `--method` is explicit, passes the active output directory to GraphRAG's Python query entrypoints, runs GraphRAG retrieval, and saves analysis pages with planner, method, index-run, manifest-hash, source-trace, conservative support-level metadata, and parsed `[Data: ...]` references when GraphRAG emits them.
- GraphRAG runtime configuration now lives in the `graph` section of `kb.config.yaml`; `kb init` syncs completion provider/model, embedding provider/model, JSON input settings, prompt paths, and API-key environment variables resolved from the centralized `providers` catalog into `graph/graphrag/settings.yaml`.
- `kb update` and `kb export` convert active complete GraphRAG Parquet outputs into generated markdown pages under `wiki/graph/` for documents, text units, entities, relationships, and communities. Complete output means the required Parquet tables and vector store are present. `kb update` also refreshes graph pages when indexing is skipped because the graph is already current. Raw document/text-unit content is fenced to keep source markdown inspectable without creating wiki lint failures, and relationship-page materialization is capped while full relationship row counts remain visible in the graph index.
- Wiki pages are now the inspection, provenance, export, and maintenance layer over the graph workflow.

## 4. New architecture

Target flow:

```text
User documents
  -> kb add
  -> normalization and manifest metadata
  -> kb update
  -> source wiki artifacts, optional legacy FTS index, and GraphRAG JSON input with provenance metadata
  -> auto GraphRAG index/update/skip decision
  -> text units, entities, relationships, communities, community reports
  -> graph wiki export
  -> wiki/graph pages for graph inspection and export
  -> kb ask
  -> auto-routed or explicit basic, local, global, and drift answers
  -> wiki artifacts for source pages, graph pages, communities, saved answers, and evaluation reports
  -> kb lint / status / doctor freshness checks
```

Layer responsibilities:

| Layer | Responsibility |
| --- | --- |
| `raw/` | Original files, normalized artifacts, manifest metadata, source hashes, converter provenance |
| `wiki/` | Human-readable source pages, legacy concept pages, generated graph pages, analysis pages, index, and activity log |
| `graph/exports/` | Existing compile-run state and optional legacy SQLite FTS5 index |
| `graph/graphrag/` | Initialized GraphRAG workspace: tracked prompts/settings, generated JSON input, ignored `.env`, cache, logs, and output |
| CLI commands | Thin user-facing wrappers over services |
| Services | Deterministic sync, status, export, lint, typed service wiring, direct graph-artifact search, and GraphRAG orchestration |
| Providers | Explicit model-backed compile, review, ask, and GraphRAG runtime configuration boundaries |

The immediate design rule is to wrap GraphRAG rather than reimplement it. The project preserves its strengths in ingestion, provenance, traceability, and maintenance while delegating graph initialization, indexing, updates, and query modes to Microsoft GraphRAG's installed Python entrypoints.

CLI design guardrails:

- `kb ask` is the GraphRAG-aware default controller.
- `kb ask --method local|global|drift|basic` exposes explicit GraphRAG method control.
- `kb find` is graph-aware but non-generative: it searches direct GraphRAG entity/relationship artifacts before falling back to maintained wiki pages.
- `kb export` refreshes graph artifact pages when complete GraphRAG output exists, so inspection pages can be regenerated without rerunning GraphRAG.
- Old FTS5 behavior should not be available through a normal `--retriever lexical` option.
- Old FTS5 behavior is exposed only through source-only `kb legacy find` and `kb legacy ask`.
- Legacy commands should print deprecation warnings to stderr for human output and keep primary answer/search output on stdout.
- `--json` output should stay machine-readable, keep stderr empty, and include retriever metadata such as `retriever: "legacy-fts"`, `deprecated: true`, and a `warning` field.
- If the GraphRAG workspace, synced input, output tables, or vector store are missing, top-level `kb ask`, `kb status`, `kb doctor`, and `kb lint` should report actionable next steps instead of silently falling back to FTS5.
- Use the main command surface for GraphRAG operations and one explicit `legacy` command group for retained FTS5 behavior; avoid hidden aliases and avoid reviving the removed graph command group.
- Command names should stay lowercase, short, and discoverable through Click help.

## 5. Evaluation plan

The pivot demotes the old retrieval path to a legacy comparator instead of a product default. Evaluation may compare:

1. Deprecated SQLite FTS5 retrieval through explicit `kb legacy ...` commands.
2. GraphRAG Basic Search.
3. GraphRAG Local Search.
4. GraphRAG Global Search.
5. GraphRAG DRIFT Search.

Benchmark questions should include local factual questions, exact lookup questions, semantic entity questions, cross-paper comparisons, whole-corpus theme questions, and maintenance/freshness checks.

Phase 8 implements this as a scriptable benchmark under `eval/benchmark.yaml` plus evaluation runners under `scripts/`. The default runner is local-safe: it records deprecated `kb legacy find --json` metrics and deterministic `kb ask` auto-router method fit, then marks provider-backed legacy ask and GraphRAG query rows as skipped unless `--allow-provider-calls` is passed. This keeps cost-bearing model work explicit while still producing repeatable CSV baselines for retrieval and routing.

Minimum metrics:

| Metric | Purpose |
| --- | --- |
| Recall@5 | Confirm expected sources appear in retrieved evidence or source references |
| Multi-source coverage | Check whether comparison questions retrieve both or all required sources |
| Method fit | Check whether local/global/drift/basic routing matches the question type |
| Claim support rate | Measure whether answer claims are grounded in retrieved source material |
| Insufficient-evidence behavior | Confirm the system hedges or refuses when evidence is weak |
| Comprehensiveness | Evaluate global and DRIFT answers for whole-corpus coverage |
| Diversity | Evaluate whether global and DRIFT answers cover multiple themes and source families |
| Latency and cost | Track practical runtime and provider usage |
| Maintenance sensitivity | Verify stale graph inputs, outputs, and wiki artifacts are detected after source changes |

This evaluation story directly explains the pivot: the original wiki workflow remains valuable for provenance and maintenance, while GraphRAG becomes the supported retrieval and synthesis path for comparison, synthesis, and corpus-level reasoning.

Current evaluation outputs:

- `scripts/evaluate_graph_modes.py` runs the full Phase 8 matrix and writes `eval/results/summary.md`, `eval/results/retrieval_metrics.csv`, and `eval/results/answer_metrics.csv`.
- `scripts/evaluate_retrieval.py` focuses on retrieval metrics such as Recall@5, multi-source coverage, method fit, and latency.
- `scripts/evaluate_answers.py` focuses on answer metrics such as claim-support status, insufficient-evidence behavior, comprehensiveness, diversity, and latency.
- `eval/results/artifacts/` stores per-question JSON command captures and is ignored because artifacts can include local corpus text or provider output.

## Source notes

- Current code source of truth: `src/services/search_service.py` maintains the SQLite FTS5 path at `graph/exports/search_index.sqlite3`, `src/services/query_service.py` builds legacy answers from source-page search results, and `src/commands/legacy.py` is the only CLI surface that invokes those retrieval behaviors.
- CLI design source: the Command Line Interface Guidelines emphasize human-first design, composability through stdout/stderr/exit codes, discoverable help, visible system state, human-readable errors, deprecation warnings before breaking changes, and simple lowercase names: <https://clig.dev/>
- Microsoft GraphRAG docs describe GraphRAG as a structured, hierarchical RAG approach that extracts a knowledge graph, builds community hierarchies, generates community summaries, and uses those structures for RAG tasks: <https://microsoft.github.io/graphrag/>
- Microsoft GraphRAG query docs describe Local, Global, DRIFT, and Basic Search modes: <https://microsoft.github.io/graphrag/query/overview/>
- Microsoft GraphRAG indexing docs describe entity, relationship, claim, community, summary, embedding, and Parquet output behavior: <https://microsoft.github.io/graphrag/index/overview/>
- Microsoft GraphRAG CLI docs expose `init`, `index`, `query`, `prompt-tune`, and `update`: <https://microsoft.github.io/graphrag/cli/>
- Microsoft GraphRAG API docs expose Python entrypoints for indexing and Local, Global, DRIFT, and Basic search: <https://microsoft.github.io/graphrag/examples_notebooks/api_overview/>
- Phase 9 local workspace source of truth: `pyproject.toml` declares `graphrag`; `kb.config.yaml` version 6 owns the `graph` provider/model/embedding defaults while API-key environment variables are resolved from `providers` unless graph-specific overrides are set; `kb init` creates and syncs `graph/graphrag/settings.yaml`; `kb update` writes `graph/graphrag/input/sources.json` from `raw/_manifest.json` and `raw/normalized/`, auto-selects full/update/skip index actions, records run metadata, warns and skips isolated missing normalized artifacts during normal mixed wiki/graph updates, prints the active graph output path after indexing, and exports generated graph markdown; `kb status`, `kb doctor`, and `kb lint` include GraphRAG readiness/freshness checks for settings, input, output tables, and vector store, and lint also reports manifest raw/normalized artifact drift; `kb ask --method auto|basic|local|global|drift` is the user-facing GraphRAG query controller and routes the active output directory through GraphRAG's Python query entrypoints; `kb find` searches direct GraphRAG entity/relationship artifacts before maintained wiki pages; `kb export` refreshes graph wiki pages when complete output exists and clean mode removes only vault markdown absent from the current export set; runtime `.env`, generated input, `output`, `cache`, `logs`, and `graph/runs/*.json` files are ignored. The old `kb graph` command group has been removed.
- Provider source notes: Google Gemini exposes embedding models such as `gemini-embedding-2` and `gemini-embedding-001` through the Gemini API; Anthropic does not offer its own embedding model and points embedding users to Voyage AI; Microsoft GraphRAG uses LiteLLM for model calls and supports non-OpenAI providers, while documenting OpenAI GPT-4-series models as its most tested path.
- Phase 8 local workspace source of truth: `eval/benchmark.yaml` contains the 12-question comparison benchmark; `scripts/evaluation_lib.py`, `scripts/evaluate_graph_modes.py`, `scripts/evaluate_retrieval.py`, and `scripts/evaluate_answers.py` generate retrieval and answer metrics; `eval/results/summary.md`, `eval/results/retrieval_metrics.csv`, and `eval/results/answer_metrics.csv` are the report outputs; `eval/results/artifacts/` is ignored because run captures may include source snippets or model output.
