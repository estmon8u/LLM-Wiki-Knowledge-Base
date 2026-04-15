# Low-Level Architecture

## Key Entry And Registry Files

| File | Responsibility |
| --- | --- |
| `src/cli.py` | Builds the CLI entrypoint and runtime context |
| `src/engine/command_registry.py` | Registers the available CLI commands |
| `src/engine/tool_registry.py` | Holds the internal tool boundary for future agent-style actions |
| `src/providers/base.py` | Defines the provider abstraction: `ProviderRequest`, `ProviderResponse`, `TextProvider` |
| `src/providers/__init__.py` | Factory `build_provider(config)` — lazy-imports the right provider by name |
| `src/providers/openai_provider.py` | OpenAI chat-completions provider (`gpt-5.4-mini`); `reasoning_effort="high"` |
| `src/providers/anthropic_provider.py` | Anthropic messages provider (`claude-sonnet-4-6`); extended thinking enabled (10 000-token budget) |
| `src/providers/gemini_provider.py` | Google Gemini provider (`gemini-3.1-flash-lite-preview`); `thinking_level="high"` |

## Current Command Files

| File | Responsibility |
| --- | --- |
| `src/commands/common.py` | Shared command helpers |
| `src/commands/init.py` | Project initialization behavior |
| `src/commands/ingest.py` | Source ingest command |
| `src/commands/compile.py` | Wiki compilation command |
| `src/commands/diff.py` | Pre-compile source diff command |
| `src/commands/search.py` | Search command |
| `src/commands/query.py` | Query command; forwards `--self-consistency N` into `QueryService` |
| `src/commands/review.py` | Semantic review command |
| `src/commands/lint.py` | Lint command |
| `src/commands/status.py` | Status command |
| `src/commands/export_vault.py` | Vault export command |

## Current Service Files

| File | Responsibility |
| --- | --- |
| `src/services/project_service.py` | Project layout and initialization helpers |
| `src/services/config_service.py` | Config loading and defaults |
| `src/services/manifest_service.py` | Raw-source manifest read/write behavior |
| `src/services/normalization_service.py` | Document-type normalization routing for direct text inputs, Docling-backed PDFs, and bounded MarkItDown-backed born-digital converters |
| `src/services/ingest_service.py` | Raw-source copy, normalized-artifact write, duplicate detection, and source registration |
| `src/services/compile_service.py` | Derived wiki generation |
| `src/services/diff_service.py` | Pre-compile source diff reporting |
| `src/services/search_service.py` | Search over compiled artifacts |
| `src/services/query_service.py` | Query answer assembly from maintained wiki context; optional provider synthesis; self-consistency sampling, claim normalization, deterministic merge, and optional save-to-wiki for analysis pages |
| `src/services/review_service.py` | Semantic review checks: topic overlap, terminology variants, future contradiction detection |
| `src/services/lint_service.py` | Structural validation for wiki links, markdown links, fragments, headings, titles, typed frontmatter, empty pages, and maintenance findings |
| `src/services/export_service.py` | Vault export generation |
| `src/services/status_service.py` | Project and corpus status reporting |

## Current Model Files

| File | Responsibility |
| --- | --- |
| `src/models/command_models.py` | Command-facing dataclasses and result types |
| `src/models/source_models.py` | Source metadata models |
| `src/models/tool_models.py` | Tool-facing data structures |
| `src/models/wiki_models.py` | Wiki-oriented dataclasses |

## Schema Files (Pydantic)

| File | Responsibility |
| --- | --- |
| `src/schemas/__init__.py` | Re-exports all schema types |
| `src/schemas/claims.py` | `EvidenceItem`, `EvidenceBundle` (with deterministic `context_hash`), `Claim`, `CandidateAnswer`, `MergedAnswer` |
| `src/schemas/review.py` | `Verdict` enum, `ReviewFinding` |
| `src/schemas/runs.py` | `RunRecord` — full deliberation artifact with auto-generated run ID and timestamp |

## Storage Files

| File | Responsibility |
| --- | --- |
| `src/storage/__init__.py` | Re-exports `RunStore` |
| `src/storage/run_store.py` | SQLite-backed persistence at `graph/exports/run_artifacts.sqlite3`: `runs` table (full record JSON + indexed columns), `run_citations` table (normalized claim→page index) |

## Supporting Project Files

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | Dependency pins, including Docling and MarkItDown, plus CLI entrypoint, Black config, pytest and coverage settings |
| `.github/workflows/tests.yml` | CI for Poetry install, Black, pytest, and coverage artifact upload |
| `tests/` | Unit, CLI, and regression coverage for the current command/service surface |

## Low-Level Guardrails

- Keep file additions aligned with the current layer split instead of mixing CLI, service, and model logic.
- Prefer extending existing services over adding duplicate helper modules.
- Treat CI and formatter config as part of the architecture because they enforce the supported workflow.
- Keep converter-backed normalization in a dedicated service instead of mixing converter logic directly into command handlers or compile.
