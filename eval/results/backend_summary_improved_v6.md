# Backend evaluation summary

This v6 retrieval artifact includes colon-prefix title aliases in the
WikiGraphRAG entity catalog. It shows the tradeoff: entity matching improves for
short paper names such as `REALM`, while recall can shift when local expansion
fans out from the newly matched entity.

## Retrieval metrics (per backend, averaged)

| Backend | Method | Effective Recall@8 | Questions w/ Ground Truth | Method Fit | Avg Latency (s) | Errors |
|---|---|---|---|---|---|---|
| graphrag | auto | 0.827 | 13/14 | 0.00 (14) | 1.361 | 0 |
| wikigraph | auto | 0.827 | 13/14 | 1.00 (14) | 0.079 | 0 |

