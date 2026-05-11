# GraphRAG Pivot

Date: 2026-05-11

## 1. Why the pivot is necessary

The current project already has useful research-memory infrastructure: source ingestion, normalized markdown or plain-text artifacts, manifest metadata, provenance-aware source pages, saved analysis pages, structural linting, provider-backed review, vault export, and a rebuildable SQLite FTS5 search index.

The weak point is retrieval. `kb find` is centered on lexical SQLite FTS5 search, and `kb ask` retrieves source-page chunks through that search path before asking a provider to synthesize an answer. That baseline works for direct factual questions when the right source chunk is retrieved, but it is not enough for cross-paper comparison, whole-corpus themes, or questions that need synthesis across related methods.

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
- `kb update` remains responsible for wiki artifact maintenance and the lexical baseline index.
- `wiki/sources/`, `wiki/concepts/`, `wiki/analysis/`, `wiki/index.md`, and `wiki/log.md` remain inspectable artifacts.
- `kb find` and the current source-grounded `kb ask` path remain available as the lexical baseline.
- `kb lint`, `kb review`, `kb status`, `kb doctor`, and `kb export` remain maintenance and operations surfaces.
- Provider configuration, conversion routing, and Windows/corporate TLS setup remain part of the current project support story.

## 3. What changes

- Retrieval moves from lexical/wiki-first to GraphRAG-first.
- The old SQLite FTS5 index becomes the baseline for comparison, exact lookup, and regression checks.
- A dedicated GraphRAG workspace will live under `graph/graphrag/`.
- Normalized corpus artifacts will be synced into GraphRAG input with source metadata attached.
- GraphRAG indexing will create graph outputs such as text units, entities, relationships, communities, and community reports.
- GraphRAG query modes will become first-class CLI behavior:
  - `basic` for simple vector-RAG comparison,
  - `local` for specific entity, paper, or method questions,
  - `global` for whole-corpus themes,
  - `drift` for comparison questions that benefit from global context plus local refinement.
- Wiki pages become the inspection, provenance, export, and maintenance layer over the graph workflow.

## 4. New architecture

Target flow:

```text
User documents
  -> kb add
  -> normalization and manifest metadata
  -> kb update
  -> source wiki artifacts and lexical baseline index
  -> kb graph sync
  -> GraphRAG JSON input with provenance metadata
  -> kb graph index
  -> text units, entities, relationships, communities, community reports
  -> kb graph ask / kb ask
  -> basic, local, global, and drift answers
  -> wiki artifacts for source pages, graph pages, communities, saved answers, and evaluation reports
  -> kb lint / status / doctor freshness checks
```

Layer responsibilities:

| Layer | Responsibility |
| --- | --- |
| `raw/` | Original files, normalized artifacts, manifest metadata, source hashes, converter provenance |
| `wiki/` | Human-readable source pages, concept pages, analysis pages, index, activity log, later graph artifacts |
| `graph/exports/` | Existing compile-run state and SQLite FTS5 lexical baseline index |
| `graph/graphrag/` | Planned GraphRAG workspace: input, prompts, settings, cache, output |
| CLI commands | Thin user-facing wrappers over services |
| Services | Deterministic sync, status, export, lint, and GraphRAG orchestration |
| Providers | Explicit model-backed compile, review, ask, and GraphRAG runtime configuration boundaries |

The immediate design rule is to wrap GraphRAG rather than reimplement it. The project should preserve its strengths in ingestion, provenance, traceability, and maintenance while delegating graph indexing and graph query modes to Microsoft GraphRAG.

## 5. Evaluation plan

The pivot turns the old retrieval path into a useful baseline instead of a dead end. Evaluation should compare:

1. SQLite FTS5 lexical baseline.
2. GraphRAG Basic Search.
3. GraphRAG Local Search.
4. GraphRAG Global Search.
5. GraphRAG DRIFT Search.

Benchmark questions should include local factual questions, exact lookup questions, semantic entity questions, cross-paper comparisons, whole-corpus theme questions, and maintenance/freshness checks.

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

This evaluation story directly explains the pivot: the original wiki workflow remains valuable for provenance and maintenance, while GraphRAG is introduced to improve comparison, synthesis, and corpus-level reasoning.

## Source notes

- Current code source of truth: `src/services/search_service.py` maintains the SQLite FTS5 baseline at `graph/exports/search_index.sqlite3`, and `src/services/query_service.py` builds `kb ask` answers from source-page search results.
- Microsoft GraphRAG docs describe GraphRAG as a structured, hierarchical RAG approach that extracts a knowledge graph, builds community hierarchies, generates community summaries, and uses those structures for RAG tasks: <https://microsoft.github.io/graphrag/>
- Microsoft GraphRAG query docs describe Local, Global, DRIFT, and Basic Search modes: <https://microsoft.github.io/graphrag/query/overview/>
- Microsoft GraphRAG indexing docs describe entity, relationship, claim, community, summary, embedding, and Parquet output behavior: <https://microsoft.github.io/graphrag/index/overview/>
- Microsoft GraphRAG CLI docs expose `init`, `index`, `query`, `prompt-tune`, and `update`: <https://microsoft.github.io/graphrag/cli/>
