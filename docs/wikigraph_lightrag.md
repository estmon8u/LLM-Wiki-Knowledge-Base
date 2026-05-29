# WikiGraphRAG — LightRAG-style backend

GraphWiki KB ships **two** implementations of the custom `WikiGraphRAG`
retrieval engine, selected by `wikigraph.mode` in `kb.config.yaml`:

| `wikigraph.mode` | Engine | Built from | Retrieval |
|---|---|---|---|
| `classic` (default) | wiki-page-first graph + Louvain communities | compiled wiki pages + normalized TextUnits | basic / local / global / drift-lite |
| `lightrag` | source-chunk-first entity/relation graph | normalized source text (`raw/normalized/*`) | local / global / hybrid / basic / auto |

The public engine name stays `WikiGraphRAG` (`kb ask --engine wikigraph`).
Microsoft GraphRAG (`--engine graphrag`) and the deprecated FTS path
(`--engine legacy`) are unchanged.

## Why LightRAG

LightRAG's contribution is a *lighter* entity/relation graph index plus
*dual-level* retrieval (specific entity queries vs. abstract theme queries),
which is cheaper than community-traversal GraphRAG while staying graph-grounded.
In GraphWiki terms:

```
normalized source chunks
  → LLM-extracted entities/relations (cached, structured)
  → deduped canonical profiles (acronym/alias/fuzzy merge, provenance preserved)
  → vector retrieval over entity & relation profiles (+ chunk BM25)
  → source-chunk citations
  → exported, inspectable wiki graph cards
```

## Pipeline & on-disk layout

```
graph/wikigraph/lightrag/
├─ index.json              # metadata: tier, source hashes, prompt hash, counts
├─ chunks.json             # token-aware LightChunks (verbatim source text)
├─ entities.json           # deduped EntityProfiles
├─ relations.json          # deduped RelationProfiles
├─ entity_vectors.json     # entity embedding vectors (omitted in BM25 fallback)
├─ relation_vectors.json   # relation embedding vectors (omitted in BM25 fallback)
├─ source_contributions.json
├─ build_manifest.json     # freshness digest + missing-source flags
└─ extraction_cache/<key>.json
```

Generated, human-readable cards are written under `wiki/wikigraph/`
(`index.md`, `entities/`, `relations/`, `sources/`, `diagnostics/`) and flow
into the Obsidian vault export like any other wiki page.

## Configuration (`kb.config.yaml`, schema v9)

```yaml
wikigraph:
  mode: classic            # classic | lightrag
  lightrag:
    chunk_token_size: 1200
    chunk_overlap_tokens: 100
    entity_extract_max_gleaning: 1
    extraction:
      extractor: deterministic   # deterministic (default, no LLM cost) | llm
    entity_types: [MODEL, METHOD, DATASET, METRIC, TASK, PAPER, TOOL,
                   ORGANIZATION, PERSON, CLAIM]
    relation_types: [USES, EVALUATES_ON, IMPROVES_OVER, COMPARES_TO,
                     INTRODUCES, DEPENDS_ON, TRADEOFF_WITH, CONTRADICTS, SUPPORTS]
    retrieval:
      default_method: hybrid
      top_k_entities: 12
      top_k_relations: 16
      top_k_chunks: 8
      max_entity_tokens: 6000
      max_relation_tokens: 8000
      max_chunk_tokens: 8000
      max_total_tokens: 24000
    embeddings:
      required_for_strict_lightrag: true
      local_fallback: bm25

embeddings:
  provider: openai
  model: text-embedding-3-large
  dimension: 3072
```

## Execution tiers

- **Tier A — strict LightRAG**: an LLM provider extracts entities/relations and
  an embedding provider builds entity/relation vectors. Used for the headline
  backend comparison.
- **Tier B — cached LightRAG**: reuses the extraction cache + persisted vectors;
  no provider needed for retrieval, optional provider for answer synthesis.
- **Tier C — fallback diagnostic mode**: no provider. Deterministic heuristic
  extraction + BM25 retrieval. Runs are labeled `bm25-fallback`; this is **not**
  strict LightRAG and is intended for local-safe use and tests.

The build report and `kb status` surface the active tier (e.g.
`provider+embedded`, `fallback+bm25`).

## CLI

```bash
# Build / refresh the LightRAG index
kb update --wikigraph-mode lightrag

# Inspect freshness, counts, tier
kb status --json   # -> wikigraph_status block

# Retrieve (dual-level)
kb find "How does RAG use retrieval?"   --engine wikigraph --method local  --json
kb find "Main retrieval themes?"        --engine wikigraph --method global --json
kb find "Compare REALM and RAG"         --engine wikigraph --method hybrid --json

# Citation-grounded answer
kb ask "How does REALM differ from RAG?" --engine wikigraph --method hybrid \
    --show-source-trace

# Export inspectable graph cards into the wiki (+ vault)
kb export
```

Every answer claim must cite a source chunk/TextUnit (entity/relation profiles
are retrieval scaffolding, not standalone evidence). Missing sources are flagged
for review (`wiki/wikigraph/diagnostics/stale-sources.md`, `kb lint`), never
silently deleted.

## Evaluation

`scripts/evaluate_backends.py` adds the full LightRAG ablation matrix:

```bash
# Retrieval-only (provider-free / BM25 fallback is safe)
python scripts/evaluate_backends.py --retrieval-only \
  --backends wikigraph-light wikigraph-light-local wikigraph-light-global \
             wikigraph-light-hybrid wikigraph-light-basic \
             wikigraph-light-no-vectors-bm25 graphrag legacy

# Provider-backed (strict tier; costs tokens)
python scripts/evaluate_backends.py --allow-provider-calls \
  --backends wikigraph-light graphrag
```

LightRAG ablation backends require an index built with
`kb update --wikigraph-mode lightrag`.


## Extraction tier is opt-in

`wikigraph.lightrag.extraction.extractor` controls extraction:

- `deterministic` (default) — provider-free heuristic extraction. `kb update
  --wikigraph-mode lightrag` makes **no LLM calls** by default, so a routine
  update never incurs surprise token cost.
- `llm` — provider-backed structured extraction (Tier A strict). Opt in
  explicitly when you want the higher-fidelity entity/relation graph.

## Incremental updates

Re-running `kb update --wikigraph-mode lightrag` is a **source-level incremental
update**: unchanged sources reuse their previous chunks (no re-chunk) and their
previous embedding vectors (no re-embed); only new/changed sources are
re-extracted and re-embedded. The build report surfaces `reused_source_count` /
`reprocessed_source_count`, and sources that disappear from the manifest are
flagged `missing` / `requires_review` in `source_contributions.json` (and
`kb lint`) rather than being silently deleted.

The query engine is cached in `WikiGraphQueryService` keyed by the index
`built_at`, so repeated `kb find` / `kb ask` calls within a long-lived process
(agent loop, evaluator) skip reloading the index and re-fitting BM25.

## Cross-corpus synthesis benchmark

`eval/benchmark_synthesis.yaml` complements `eval/benchmark.yaml`: every question
requires 3+ source papers, which is the multi-hop workload dual-level retrieval
targets. Run both and compare deltas across the `wikigraph-light*` ablation
backends.
