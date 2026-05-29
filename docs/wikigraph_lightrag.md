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

Backends (including `wikigraph` with `--wikigraph-method local|global|hybrid`) are compared with the research-grounded RAG evaluation harness:

```bash
python scripts/evaluate_rag.py --retrieval-only \
    --methods direct legacy graphrag wikigraph
python scripts/evaluate_rag.py --allow-provider-calls --ragas --judge \
    --methods direct legacy graphrag wikigraph
```

See [docs/rag_evaluation.md](rag_evaluation.md) for metrics (rank-aware retrieval, RAGAS, anti-gaming generation, bias-mitigated judge) and the fairness/contamination methodology.
