# Backend evaluation summary

This provider-backed real-PDF answer summary includes colon-prefix alias
behavior, so short paper names in questions can resolve to colon-formatted wiki
page titles before answer synthesis.

## Retrieval metrics (per backend, averaged)

| Backend | Method | Effective Recall@8 | Questions w/ Ground Truth | Method Fit | Avg Latency (s) | Errors |
|---|---|---|---|---|---|---|
| graphrag | auto | 0.827 | 13/14 | 0.00 (14) | 1.370 | 0 |
| wikigraph | auto | 0.827 | 13/14 | 1.00 (14) | 0.112 | 0 |

## Answer metrics (per backend, averaged)

| Backend | Method | Provider Modes | Quality Score | Grounded Entity Rate | Grounded Term Rate | Has Supported Citations | Avg Citations | Insufficient-Evidence Match | Citation Ref Valid (loose) | Citation Ref Valid (strict) |
|---|---|---|---|---|---|---|---|---|---|---|
| graphrag | auto | provider-backed | **0.874** | 0.857 | 0.881 | 0.79 | 4.86 | 0.93 | 0.786 | 0.786 |
| wikigraph | auto | provider | **0.839** | 0.554 | 0.548 | 1.00 | 5.21 | 0.64 | 1.000 | 1.000 |

