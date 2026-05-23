# LightRAG-Style WikiGraphRAG Backend

`wikigraph.mode: lightrag` reimplements the custom WikiGraphRAG backend
in the style of [LightRAG (EMNLP 2025)][lightrag], while preserving the
project's invariants: normalized-source provenance, wiki as the
inspectable artifact layer, citation-grounded answers, unified `kb ask`
/ `kb find` CLI surface, stale-state detection, and backend comparison.

This document describes the high-level architecture, file layout, and
execution tiers. For the full migration plan and rationale, see the PR
that introduced this mode.

## Architecture

```
GraphWiki KB
├─ raw/ + manifest + normalized artifacts     source of truth
├─ wiki/                                      human-readable artifact layer
├─ graph/graphrag/                            Microsoft GraphRAG backend
├─ graph/wikigraph/                           classic WikiGraphRAG artifacts
└─ graph/wikigraph/lightrag/                  LightRAG-style backend
```

The new `lightrag` mode is **source-chunk-first** instead of
**wiki-page-first**:

```
normalized source chunk
  → LLM (or deterministic) entity / relation extraction
  → canonical entity & relation profiles (dedupe + alias merge)
  → vector retrieval over profile embedding texts
  → answer prompts cite source chunk anchors
  → exported wiki graph cards under wiki/wikigraph/* (optional)
```

## Modules

All new code lives under `src/graphwiki_kb/wikigraph/light_*.py`:

| Module                       | Responsibility                                                         |
|------------------------------|------------------------------------------------------------------------|
| `light_models.py`            | Pydantic models (`LightChunk`, `EntityProfile`, `RelationProfile`, …). |
| `light_chunker.py`           | Token-aware chunking of normalized artifacts (~1200-token windows).     |
| `light_extractor.py`         | Provider-protocol + deterministic offline extractor + extraction cache.|
| `light_deduper.py`           | Canonicalize entities (alias/acronym/fuzzy) and relations (inverse).   |
| `light_embeddings.py`        | `EmbeddingProvider` protocol + BM25 / hashing fallbacks.               |
| `light_vector_store.py`      | Cosine top-K vector store (NumPy when available, pure-Python else).    |
| `light_graph_store.py`       | JSON-backed persistence under `graph/wikigraph/lightrag/`.             |
| `light_index_builder.py`     | End-to-end builder + incremental update + source contributions.        |
| `light_keywords.py`          | Low- / high-level keyword extraction (rule-based fallback).            |
| `light_context_builder.py`   | Dual-level retrieval (`local`, `global`, `hybrid`, `basic`, `auto`).   |
| `light_query_service.py`     | High-level `LightGraphQueryEngine` + `LightAnswerService`.             |

`WikiGraphIndexService.build()` and `WikiGraphQueryService.find()` /
`.ask()` dispatch on `wikigraph.mode`, so callers (CLI, tests, scripts)
never need to branch.

## Execution Tiers

| Tier | Requires                                   | Use case                              |
|------|--------------------------------------------|---------------------------------------|
| A    | LLM provider + embedding provider          | Strict LightRAG comparison runs.      |
| B    | Existing extraction cache + saved vectors  | Local ask after a one-time index run. |
| C    | None (deterministic extractor + BM25)      | CI, tests, offline diagnostic runs.   |

Always label Tier C output as **fallback diagnostic**, never as a strict
LightRAG run.

## Config

`kb.config.yaml` (schema version 9) adds:

```yaml
wikigraph:
  mode: classic   # or lightrag
  lightrag:
    chunk_token_size: 1200
    overlap_tokens: 100
    min_chunk_tokens: 30
    fuzzy_match_threshold: 88
    max_description_chars: 600
    embed_chunks: false
    extraction_min_occurrences: 1
    entity_types: [MODEL, METHOD, DATASET, METRIC, TASK, PAPER,
                   TOOL, ORGANIZATION, PERSON, CLAIM]
    relation_types: [USES, EVALUATES_ON, IMPROVES_OVER, COMPARES_TO,
                     INTRODUCES, DEPENDS_ON, TRADEOFF_WITH,
                     CONTRADICTS, SUPPORTS]
    retrieval:
      default_method: hybrid
      top_k_entities: 12
      top_k_relations: 16
      top_k_chunks: 8
      max_total_tokens: 24000
      rrf_k: 60
    embeddings:
      provider: bm25
      model: bm25-fallback
      dimension: 0
      local_fallback: bm25
```

The migration from v8 leaves existing configs at `mode: classic` so the
live behavior is unchanged until the user opts in.

## CLI surface

```bash
# Build the LightRAG-style index (one-off override).
kb update --wikigraph-mode lightrag

# Query with dual-level retrieval.
kb find "Compare RAG and DPR" --engine wikigraph --json
kb ask  "Compare RAG and DPR" --engine wikigraph --method drift-lite

# Inspect status.
kb status --json
```

When `wikigraph.mode == lightrag`, the legacy `kb ask --engine wikigraph`
flow transparently dispatches to the LightRAG query engine; the
`--method drift-lite` option maps to LightRAG `hybrid` (the closest
classic literal). The retrieval bundle is rendered into the existing
`WikiGraphFindResult` / `WikiGraphAnswer` shapes so downstream tools
keep working.

## Status block

`kb status --json` reports a `wikigraph.lightrag` block with:

```json
{
  "initialized": true,
  "built_at": "...",
  "chunk_count": 850,
  "entity_count": 420,
  "relation_count": 760,
  "source_count": 30,
  "missing_source_count": 0,
  "extractor": "deterministic",
  "embedding_provider": "bm25-sparse",
  "embedding_model": "bm25-sparse",
  "stale_reasons": []
}
```

Missing sources (present in the previous build but absent from the
current manifest) are flagged with `status: "missing"` and
`requires_review: true` rather than silently dropped — per project
recommendation §11 / §22.

[lightrag]: https://github.com/HKUDS/LightRAG
