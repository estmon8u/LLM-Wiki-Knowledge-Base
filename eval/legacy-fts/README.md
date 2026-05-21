# Legacy FTS Capture

`realm_vs_rag_find.json` was captured with `kb find --engine legacy --json` against the real-PDF validation project.

Provider-backed `kb ask --engine legacy` artifacts are pending explicit approval because those commands send retrieved local source snippets to the configured external model provider.

Phase 8 keeps these captures as historical comparison data. The current benchmark and scripts under `eval/benchmark.yaml` and `scripts/evaluate_graph_modes.py` can run `kb find --engine legacy --json` safely by default, but `kb ask --engine legacy` is skipped unless `--allow-provider-calls --include-legacy-ask` is passed.
