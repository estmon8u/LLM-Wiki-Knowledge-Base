# Command surface end-to-end smoke test

This report was first added in commit `1031eb8` as the branch's persisted
command-surface verification artifact. It exercises the unified post-pivot CLI
surface, including `find --engine ...`, `ask --engine ...`, `agent`, `export`,
and `update --no-graph`.

- Project root: `/tmp/wikigraph-verify-project`
- Commands exercised: 47
- OK: 47
- Expected failures: 0
- Unexpected failures: 0
- Full log: `/tmp/wikigraph-command-surface-e2e.log`

| Command | Exit | Status | Seconds |
|---|---:|---|---:|
| `--help` | 0 | ok | 2.08 |
| `add --help` | 0 | ok | 1.15 |
| `agent --help` | 0 | ok | 1.99 |
| `ask --help` | 0 | ok | 1.27 |
| `config --help` | 0 | ok | 1.23 |
| `doctor --help` | 0 | ok | 1.23 |
| `export --help` | 0 | ok | 1.24 |
| `find --help` | 0 | ok | 1.21 |
| `init --help` | 0 | ok | 1.13 |
| `lint --help` | 0 | ok | 1.14 |
| `review --help` | 0 | ok | 1.16 |
| `sources --help` | 0 | ok | 1.17 |
| `status --help` | 0 | ok | 1.19 |
| `update --help` | 0 | ok | 1.18 |
| `--project-root /tmp/wikigraph-verify-project status` | 0 | ok | 1.32 |
| `--project-root /tmp/wikigraph-verify-project status --json` | 0 | ok | 1.32 |
| `--project-root /tmp/wikigraph-verify-project status --changed` | 0 | ok | 1.37 |
| `--project-root /tmp/wikigraph-verify-project status --strict` | 0 | ok | 1.39 |
| `--project-root /tmp/wikigraph-verify-project config show` | 0 | ok | 1.31 |
| `--project-root /tmp/wikigraph-verify-project config provider clear` | 0 | ok | 1.29 |
| `--project-root /tmp/wikigraph-verify-project config provider set openai --model gpt-4.1-mini` | 0 | ok | 1.32 |
| `--project-root /tmp/wikigraph-verify-project sources list` | 0 | ok | 1.29 |
| `--project-root /tmp/wikigraph-verify-project sources list --json` | 0 | ok | 1.22 |
| `--project-root /tmp/wikigraph-verify-project sources show graphwiki-kb` | 0 | ok | 1.24 |
| `--project-root /tmp/wikigraph-verify-project find GraphRAG provider config` | 0 | ok | 1.78 |
| `--project-root /tmp/wikigraph-verify-project find --engine wiki GraphRAG` | 0 | ok | 1.41 |
| `--project-root /tmp/wikigraph-verify-project find --engine wikigraph GraphRAG` | 0 | ok | 1.44 |
| `--project-root /tmp/wikigraph-verify-project find --engine graphrag GraphRAG` | 0 | ok | 1.57 |
| `--project-root /tmp/wikigraph-verify-project find --engine legacy GraphRAG` | 0 | ok | 1.51 |
| `--project-root /tmp/wikigraph-verify-project find --json GraphRAG` | 0 | ok | 1.72 |
| `--project-root /tmp/wikigraph-verify-project ask --engine wikigraph --json Where is GraphRAG configured?` | 0 | ok | 13.93 |
| `--project-root /tmp/wikigraph-verify-project ask --engine wikigraph --show-source-trace Where is GraphRAG configured?` | 0 | ok | 13.77 |
| `--project-root /tmp/wikigraph-verify-project ask --engine graphrag --json Where is GraphRAG configured?` | 0 | ok | 18.43 |
| `--project-root /tmp/wikigraph-verify-project ask --engine legacy --json Where is GraphRAG configured?` | 0 | ok | 4.5 |
| `--project-root /tmp/wikigraph-verify-project ask --engine all --json Where is GraphRAG configured?` | 0 | ok | 30.7 |
| `--project-root /tmp/wikigraph-verify-project lint` | 0 | ok | 7.32 |
| `--project-root /tmp/wikigraph-verify-project lint --json` | 0 | ok | 6.94 |
| `--project-root /tmp/wikigraph-verify-project doctor` | 0 | ok | 5.76 |
| `--project-root /tmp/wikigraph-verify-project doctor --json` | 0 | ok | 5.19 |
| `--project-root /tmp/wikigraph-verify-project doctor --strict` | 0 | ok | 5.3 |
| `--project-root /tmp/wikigraph-verify-project review` | 0 | ok | 10.05 |
| `--project-root /tmp/wikigraph-verify-project review --json` | 0 | ok | 7.35 |
| `--project-root /tmp/wikigraph-verify-project agent --json Report KB status` | 0 | ok | 7.95 |
| `--project-root /tmp/wikigraph-verify-project export` | 0 | ok | 6.0 |
| `--project-root /tmp/wikigraph-verify-project export --clean` | 0 | ok | 6.07 |
| `--project-root /tmp/wikigraph-verify-project add /tmp/wikigraph-extra-note.md` | 0 | ok | 1.27 |
| `--project-root /tmp/wikigraph-verify-project update --no-graph` | 0 | ok | 7.39 |
