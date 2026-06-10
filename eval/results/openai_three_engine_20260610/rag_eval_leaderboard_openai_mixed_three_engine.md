# RAG evaluation leaderboard

> Contamination caveat: the corpus papers are public, so the `direct` (no-retrieval) baseline may benefit from pretraining memorization; weigh grounded/faithfulness/citation metrics over raw correctness.

| Backend | recall_at_k | precision_at_k | ndcg_at_k | mrr | hit_at_k | citation_validity | grounded | refusal_correct | token_f1 | rouge_l | entity_coverage | latency_seconds | answer_token_length |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| legacy | 1.000 [1.00,1.00] (n=5) | 0.325 [0.17,0.53] (n=5) | 0.815 [0.63,0.98] (n=5) | 0.767 [0.50,1.00] (n=5) | 1.000 [1.00,1.00] (n=5) | 0.800 [0.40,1.00] (n=5) | 0.200 [0.00,0.60] (n=5) | 0.200 [0.00,0.60] (n=5) | 0.464 [0.46,0.46] (n=1) | 0.261 [0.26,0.26] (n=1) | 1.000 [1.00,1.00] (n=1) | 5.323 [2.76,8.31] (n=5) | 126.400 [73.60,181.00] (n=5) |
| wikigraph-classic | 0.950 [0.90,1.00] (n=5) | 0.750 [0.53,0.93] (n=5) | 0.992 [0.98,1.00] (n=5) | 1.000 [1.00,1.00] (n=5) | 1.000 [1.00,1.00] (n=5) | 1.000 [1.00,1.00] (n=5) | 1.000 [1.00,1.00] (n=5) | 1.000 [1.00,1.00] (n=5) | 0.224 [0.15,0.32] (n=5) | 0.145 [0.11,0.18] (n=5) | 0.733 [0.47,1.00] (n=5) | 5.349 [3.43,7.93] (n=5) | 154.400 [94.40,246.40] (n=5) |
| wikigraph-lightrag | 0.550 [0.25,0.85] (n=5) | 0.475 [0.20,0.70] (n=5) | 0.703 [0.31,0.99] (n=5) | 0.650 [0.25,1.00] (n=5) | 0.800 [0.40,1.00] (n=5) | 1.000 [1.00,1.00] (n=5) | 1.000 [1.00,1.00] (n=5) | 1.000 [1.00,1.00] (n=5) | 0.257 [0.19,0.34] (n=5) | 0.181 [0.13,0.24] (n=5) | 0.800 [0.40,1.00] (n=5) | 9.373 [8.38,10.30] (n=5) | 133.000 [55.40,252.20] (n=5) |
