# Legacy FTS Capture

`realm_vs_rag_find.json` was captured with `kb legacy find --json` against the real-PDF validation project.

Provider-backed `kb legacy ask` artifacts are pending explicit approval because those commands send retrieved local source snippets to the configured external model provider.

Phase 8 keeps these captures as historical comparison data. The current benchmark and scripts under `eval/benchmark.yaml` and `scripts/evaluate_graph_modes.py` can run `kb legacy find --json` safely by default, but `kb legacy ask` is skipped unless `--allow-provider-calls --include-legacy-ask` is passed.
