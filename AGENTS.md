# AGENTS.md

## Cursor Cloud specific instructions

This is a Python CLI application ("GraphWiki KB") managed with **Poetry**. There is no web server, database server, or Docker dependency — it is a single-package CLI tool.

### Quick reference

| Task | Command |
|---|---|
| Install deps | `poetry install --with dev --all-extras` |
| Format check | `poetry run black --check src tests` |
| Lint | `poetry run ruff check src tests` |
| Type check | `poetry run mypy src/graphwiki_kb` |
| Run tests | `poetry run pytest tests -q` |
| CLI smoke test | `poetry run kb --help` |
| Build wheel | `poetry build` |

See `README.md` for the full command reference and `docs/start-guide.md` for the first-run walkthrough.

### Gotchas

- **System deps for `poetry install`**: The `docling` optional extra transitively requires `pycairo`, which needs `libcairo2-dev`, `pkg-config`, and `python3-dev` system packages on Ubuntu. These are already installed in the Cloud Agent VM snapshot.
- **`poetry` PATH**: Poetry is installed at `~/.local/bin`. Ensure `PATH` includes `$HOME/.local/bin` (already configured in the Cloud Agent VM).
- **No API keys needed for tests**: The test suite uses stub/mock providers. All 888 tests pass without any API keys. Real `kb update`, `kb ask`, `kb review` commands require `OPENAI_API_KEY` (or another provider key) set as an environment variable.
- **`kb update --no-graph` still requires a provider**: Even wiki-only updates need a configured provider for compile summaries. Commands that work without a provider: `kb init`, `kb add`, `kb status`, `kb doctor`, `kb lint`, `kb sources`, `kb find`, `kb export`.
- **GraphRAG lock files are runtime state**: Commit `1c2276c` added `graph/graphrag/.workspace.state.lock` to `.gitignore` after it was accidentally committed. Do not stage that file; it belongs to local GraphRAG workspace locking only.
- **Rich table width**: The Rich `Console()` uses the actual terminal width (often 80 in CI/VM). CLI tests that assert full file paths in table output should use partial matches or `--json` output to avoid Rich table truncation.
