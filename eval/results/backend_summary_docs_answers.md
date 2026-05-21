# Backend evaluation summary

## Retrieval metrics (per backend, averaged)

| Backend | Method | Effective Recall@5 | Questions w/ Ground Truth | Avg Latency (s) | Errors |
|---|---|---|---|---|---|
| graphrag | auto | 0.350 | 5/6 | 0.549 | 0 |
| wikigraph | auto | 0.767 | 5/6 | 0.033 | 0 |

## Answer metrics (per backend, averaged)

| Backend | Method | Provider Modes | Quality Score | Grounded Entity Rate | Grounded Term Rate | Avg Citations | Insufficient-Evidence Match | Citation Ref Valid Rate |
|---|---|---|---|---|---|---|---|---|
| graphrag | auto | provider-backed | **0.830** | 1.000 | 0.833 | 2.67 | 0.83 | 1.000 |
| wikigraph | auto | provider | **0.907** | 0.833 | 0.833 | 4.00 | 1.00 | 1.000 |

