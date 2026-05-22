# Backend evaluation summary

Branch note: commit `1db294e` added this baseline-vs-improved retrieval
artifact for the Phase 4 ablation. This file is the WGR baseline with
`wikigraph.retrieval_improvements_enabled=false`.

## Retrieval metrics (per backend, averaged)

| Backend | Method | Effective Recall@8 | Questions w/ Ground Truth | Method Fit | Avg Latency (s) | Errors |
|---|---|---|---|---|---|---|
| graphrag | auto | 0.827 | 13/14 | 0.00 (14) | 1.353 | 0 |
| wikigraph | auto | 0.865 | 13/14 | 1.00 (14) | 0.075 | 0 |

