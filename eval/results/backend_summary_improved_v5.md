# Backend evaluation summary

Branch note: commit `1db294e` records the conservative Phase 4 result here:
section-title overlap boost and drift-lite additive cross-bundle bonus remained,
while aggressive RRF plus alias-expanded BM25 was not kept in public retrieval.

## Retrieval metrics (per backend, averaged)

| Backend | Method | Effective Recall@8 | Questions w/ Ground Truth | Method Fit | Avg Latency (s) | Errors |
|---|---|---|---|---|---|---|
| graphrag | auto | 0.827 | 13/14 | 0.00 (14) | 1.348 | 0 |
| wikigraph | auto | 0.865 | 13/14 | 1.00 (14) | 0.075 | 0 |

