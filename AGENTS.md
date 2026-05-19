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
- **Two test files have pre-existing collection errors** (`tests/test_evaluation_scripts.py` and `tests/test_phase9_graph_cli.py`) due to missing `__init__.py` files in `scripts/` and `tests/`. CI may handle this differently. You can skip them with `--ignore` flags if needed.
- **Pre-existing lint issues**: `ruff check` reports ~31 errors (mostly `RUF043` and `RUF059` rules from newer ruff versions). `mypy` reports 4 errors in `file_lock.py` for Windows-only `msvcrt` module attributes. These are pre-existing in `main`.
- **No API keys needed for tests**: The test suite uses stub/mock providers. All 849+ tests pass without any API keys. Real `kb update`, `kb ask`, `kb review` commands require `OPENAI_API_KEY` (or another provider key) set as an environment variable.
- **`kb update --no-graph` still requires a provider**: Even wiki-only updates need a configured provider for compile summaries. Commands that work without a provider: `kb init`, `kb add`, `kb status`, `kb doctor`, `kb lint`, `kb sources`, `kb find`, `kb export`.
