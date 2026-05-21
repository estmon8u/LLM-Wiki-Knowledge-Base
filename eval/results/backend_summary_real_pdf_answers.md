# Backend evaluation summary

## Retrieval metrics (per backend, averaged)

| Backend | Method | Effective Recall@8 | Questions w/ Ground Truth | Method Fit | Avg Latency (s) | Errors |
|---|---|---|---|---|---|---|
| graphrag | auto | 0.827 | 13/14 | 0.00 (14) | 1.365 | 0 |
| wikigraph | auto | 0.865 | 13/14 | 1.00 (14) | 0.144 | 0 |

## Answer metrics (per backend, averaged)

| Backend | Method | Provider Modes | Quality Score | Grounded Entity Rate | Grounded Term Rate | Has Supported Citations | Avg Citations | Insufficient-Evidence Match | Citation Ref Valid (loose) | Citation Ref Valid (strict) |
|---|---|---|---|---|---|---|---|---|---|---|
| graphrag | auto | provider-backed | **0.938** | 0.929 | 0.881 | 0.93 | 5.21 | 0.93 | 0.929 | 0.929 |
| wikigraph | auto | provider | **0.838** | 0.518 | 0.571 | 1.00 | 4.71 | 0.64 | 1.000 | 1.000 |

