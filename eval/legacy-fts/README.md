# Legacy FTS Capture

`realm_vs_rag_find.json` was captured with `kb find --engine legacy --json` against the real-PDF validation project.

Provider-backed `kb ask --engine legacy` artifacts are pending explicit approval because those commands send retrieved local source snippets to the configured external model provider.

Phase 8 keeps these captures as historical comparison data. The current benchmark and scripts under `eval/benchmark.yaml` and `scripts/evaluate_graph_modes.py` can run `kb find --engine legacy --json` safely by default, but `kb ask --engine legacy` is skipped unless `--allow-provider-calls --include-legacy-ask` is passed.

For branch comparisons after commit `c28a62e`, legacy FTS retrieval rows use the
same fair metric columns as the graph backends: `effective_recall_at_5` ignores
questions without ground-truth sources, while answer scoring separates raw
entity mentions from grounded entity hits so refusal text does not receive
entity-match credit.
After commit `b77b1cc`, evaluator subprocesses call the unified legacy surface
directly with `kb ask --engine legacy` and `kb find --engine legacy`; old
`kb legacy` invocations should not appear in new captures.
