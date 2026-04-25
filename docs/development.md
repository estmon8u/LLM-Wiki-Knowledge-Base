# Developer Guide

This guide is for contributors working on `kb` itself. For user-facing
documentation (installation, commands, provider configuration) see the
[README](../README.md).

## Prerequisites

| Tool | Version | Notes |
| --- | --- | --- |
| Python | 3.11.x | Constrained in pyproject.toml |
| Poetry | Latest | Dependency management and virtualenv |
| Git | Any recent | Branch workflow below |
| wkhtmltopdf | Optional | Only needed if HTML conversion is enabled |

Install dependencies:

```bash
poetry install
```

## Running tests

```bash
poetry run pytest tests -q
```

Coverage must stay at or above **97%** (enforced by `--cov-fail-under=97` in
pyproject.toml). Every PR that drops below this threshold will fail.

### Format code

```bash
poetry run black src tests
```

### Coverage report

```bash
poetry run pytest tests --cov=src --cov-report=term-missing
```

### Real corpus smoke test

Exercise the full CLI against a real source corpus:

```bash
poetry run python scripts/run_real_corpus_smoke.py \
    --raw-root path/to/raw-corpus \
    --project-root path/to/disposable-project
```

This runs every command (`init`, `add`, `update`, `find`, `ask`, `lint`,
`review`, `export`, etc.) against actual files and writes a consolidated log.
It is a manual workflow, not a CI dependency.

## Architecture overview

Detailed docs live in `docs/architecture/`:

- [high-level.md](architecture/high-level.md) — product boundaries, data
  domains, runtime flow
- [mid-level.md](architecture/mid-level.md) — package map, command-to-service
  mapping, data movement
- [low-level.md](architecture/low-level.md) — file-level responsibilities

The rest of this guide covers the patterns you need to know when making
changes.

## Project layout

```
src/
  cli.py                    # Click entrypoint, builds runtime context
  engine/
    command_registry.py     # Maps command names → module paths
  commands/                 # Thin Click wrappers (one file per command)
  services/                 # Business logic (one service per domain)
  models/                   # Shared dataclasses (CommandContext, wiki models, etc.)
  providers/                # LLM provider abstraction + implementations
  storage/                  # Persistent stores (compile runs, FTS5 search index)
  data/                     # Bundled data files (e.g. english_stopwords.txt)
  schemas/                  # (reserved, currently empty)
tests/
  conftest.py               # Shared fixtures: TestProject, _StubProvider
  test_*.py                 # One test file per domain
  golden_markdown/          # Expected markdown output for golden-file tests
docs/
  architecture/             # Layered architecture docs
  proposal/                 # Capstone proposal documents
```

## Layering rules

### Commands are thin

Files in `src/commands/` are Click command definitions only. They validate
arguments, call a service, and format output. No business logic belongs here.

A typical command file:

```python
# src/commands/example.py
def build_spec(_: CommandContext = None) -> CommandSpec:
    return CommandSpec(name="example", summary=SUMMARY)

def create_command() -> click.Command:
    @click.command(name="example", help=SUMMARY)
    @click.pass_obj
    @require_initialized
    def cmd(ctx: CommandContext, ...) -> None:
        service = ctx.services["example"]
        result = service.run(...)
        echo_section("Result", result.summary)
    return cmd
```

### Services own the logic

Files in `src/services/` contain all business logic. Each service receives
`ProjectPaths` and any dependencies (provider, other services) through its
constructor. Services do not import Click or Rich.

### Storage is separate

`src/storage/` holds persistent state classes (`CompileRunStore`,
`SearchIndexStore`). Services depend on storage, not the other way around.

## Adding a command

1. Create `src/commands/yourcommand.py` with `build_spec()` and
   `create_command()`.
2. Register it in `src/engine/command_registry.py` by adding an entry to
   `FLAT_COMMAND_MODULES`:
   ```python
   FLAT_COMMAND_MODULES = {
       ...
       "yourcommand": "src.commands.yourcommand",
   }
   ```
3. Create the backing service in `src/services/` if one doesn't exist.
4. Wire the service in `src/services/__init__.py` inside `build_services()`.
5. Add tests. Update the README command reference.

## Adding a provider

All providers implement the `TextProvider` interface from
`src/providers/base.py`:

```python
class TextProvider:
    name: str = "base"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        raise NotImplementedError
```

`ProviderRequest` carries the prompt, optional system prompt, max tokens, an
optional JSON response schema hint, and per-request reasoning effort.
`ProviderResponse` returns the text, model name, token counts, and finish
reason.

To add a provider:

1. Create `src/providers/yourprovider.py` implementing `TextProvider`.
2. Apply the `@provider_retry()` decorator from `src/providers/retry.py` to
   your `generate()` method (3 attempts, exponential backoff, transient-only).
3. Register it in `build_provider()` inside `src/providers/__init__.py`.
4. Add a catalog entry to `_FALLBACK_PROVIDER_CATALOG` with default model,
   API key env var, and reasoning settings.
5. Add tests.

## Structured output parsing

Provider responses often come back as JSON wrapped in prose or fenced code
blocks. Use the shared parser from `src/providers/structured.py`:

```python
from src.providers.structured import parse_model_payload

result = parse_model_payload(response.text, YourPydanticModel, label="review")
```

`parse_model_payload` handles direct JSON, fenced JSON, and prose-prefaced
JSON. It validates against a Pydantic model and raises
`StructuredOutputError` on failure.

For raw JSON without Pydantic validation, use `parse_json_payload`.

## Shared modules

These exist so you don't reinvent them:

| Module | What it provides |
| --- | --- |
| `services/markdown_document.py` | `parse_document()`, `parse_frontmatter()`, `plain_text()`, `headings()`, `sections()`, `links()` — AST-based markdown parsing via markdown-it-py |
| `services/stopwords.py` | `STOPWORDS` frozenset loaded from `src/data/english_stopwords.txt` — 198 NLTK English stopwords, no runtime download needed |
| `services/citation_cleanup.py` | `clean_citation_refs()` — strips raw `[wiki/sources/page.md#chunk-0]` markers from provider answers |
| `services/project_service.py` | `ProjectPaths`, `build_project_paths()`, `slugify()`, `atomic_write_text()`, `utc_now_iso()` |

## Data files

Bundled data lives in `src/data/`. Currently this contains
`english_stopwords.txt` (198 words, one per line). This convention exists
because NLTK corpus data is runtime data, not a pip package — bundling it
avoids requiring `nltk.download()` at runtime.

If you add a new data file, use the same pattern: plain text, loaded via
`Path(__file__).resolve().parent.parent / "data" / "filename"` from a service
module.

## NLTK dependencies

NLTK is used for:

- `SnowballStemmer("english")` — term stemming in concept and review services
- `punkt_tab` tokenizer — sentence splitting in normalization
- Bigram/trigram collocation scoring — deterministic concept topic extraction

The `punkt_tab` tokenizer data must be downloaded once:

```bash
poetry run python -c "import nltk; nltk.download('punkt_tab')"
```

Stopwords are **not** loaded from NLTK at runtime — they are bundled in
`src/data/english_stopwords.txt`.

## Testing conventions

### Test file naming

One test file per service/domain:

| Test file | Tests for |
| --- | --- |
| `test_compile_and_lint.py` | CompileService and LintService |
| `test_concept_service.py` | ConceptService |
| `test_review_service.py` | ReviewService |
| `test_search_query_export_status.py` | SearchService, QueryService, ExportService, StatusService |
| `test_cli.py` | CLI integration (Click runner) |
| `test_normalization_service.py` | NormalizationService |
| `test_manifest_and_ingest.py` | ManifestService and IngestService |
| `test_project_and_config.py` | ProjectService and ConfigService |
| `test_provider_integration.py` | Provider abstraction, factory, catalog |
| `test_golden_markdown.py` | Golden-file markdown output comparison |

### Fixtures

`tests/conftest.py` provides:

- `test_project` — a fully initialized project in `tmp_path` with a
  `_StubProvider` that returns deterministic summaries
- `uninitialized_project` — same but without `kb init`

The `_StubProvider` returns `"Stub summary of the document."` for compile
requests and `'{"issues": []}'` for review requests. Tests that need
different provider behavior should create their own provider class.

### Golden files

`tests/golden_markdown/` contains expected markdown output (source pages,
index, analysis pages). `test_golden_markdown.py` compares actual generated
output against these files. Update the golden files when intentional output
changes are made.

### When to add vs. update tests

- **New feature or command**: add a new test function or test file.
- **Changed behavior**: update existing assertions to match the new behavior.
  Don't just delete tests that fail.
- **Bug fix**: add a regression test that would have caught the bug.

## Branch workflow

- `main` is the stable branch.
- Feature work happens on named branches (e.g. `quality-layer`,
  `simplify`).
- Commits should be focused and incremental with descriptive messages.
- Run `poetry run black src tests` and `poetry run pytest tests -q` before
  pushing.
- The git identity for this repo is `estmon8u@users.noreply.github.com`.

## Data domains

The project organizes data into four directories inside a project root:

| Directory | Ownership | Contents |
| --- | --- | --- |
| `raw/` | User + ingest | Original source files, normalized markdown, `_manifest.json` |
| `wiki/` | Generated | Source pages, concept pages, analysis pages, index, log |
| `vault/` | Export | Obsidian-compatible vault copy |
| `graph/` | Machine | SQLite search index, compile run state, future operational state |

Commands read from earlier domains and write to later ones. `raw/` is
user-owned input. `wiki/` is the generated knowledge base. `vault/` is the
export target. `graph/` holds operational state that should not be
hand-edited.
