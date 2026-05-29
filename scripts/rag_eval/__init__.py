"""Research-grounded RAG evaluation harness.

A from-scratch, fair, non-gameable, rigorous evaluation harness that compares
all four answering methods (direct / legacy / graphrag / wikigraph) using:

* deterministic **rank-aware retrieval metrics** (recall@k, precision@k, MRR,
  nDCG@k, hit@k) against ground-truth source ids;
* **RAGAS** answer/context-quality metrics (faithfulness, answer relevancy,
  context precision, context recall) via the real ``ragas`` library;
* deterministic **anti-gaming generation metrics** (citation validity, grounded
  rate, refusal correctness, lexical overlap, verbosity guard);
* a **bias-mitigated LLM judge** (blinded identity, order-swap, cross-family);
* **bootstrap confidence intervals** so within-noise differences are not
  reported as wins.

Provider/LLM-judge/RAGAS calls are gated behind an explicit opt-in so the
deterministic retrieval layer stays offline and reproducible.
"""

from __future__ import annotations

from scripts.rag_eval import _compat

__all__ = ["_compat"]
