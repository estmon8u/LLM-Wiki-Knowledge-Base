from __future__ import annotations

"""Service construction helpers for the capstone CLI."""

from typing import Any

from src.providers import build_provider
from src.services.compile_service import CompileService
from src.services.concept_service import ConceptService
from src.services.config_service import ConfigService
from src.services.diff_service import DiffService
from src.services.doctor_service import DoctorService
from src.services.export_service import ExportService
from src.services.ingest_service import IngestService
from src.services.lint_service import LintService
from src.services.manifest_service import ManifestService
from src.services.model_registry_service import ModelRegistryService
from src.services.project_service import ProjectPaths, ProjectService
from src.services.query_service import QueryService
from src.services.review_service import ReviewService
from src.services.search_service import SearchService
from src.services.status_service import StatusService
from src.storage.compile_run_store import CompileRunStore
from src.storage.run_store import RunStore


def _resolve_provider(
    config: dict[str, Any], registry: ModelRegistryService, task: str
) -> Any:
    """Resolve and build a provider for a specific task using the registry."""
    provider_cfg = config.get("provider") or {}
    if not provider_cfg.get("name"):
        return build_provider(config)  # returns None when no provider configured

    runtime = config.get("_runtime") or {}
    try:
        resolved = registry.resolve(
            config=config,
            tier=runtime.get("tier"),
            model=runtime.get("model"),
            task=task,
        )
    except ValueError:
        # Unknown or test-stub provider — fall back to unresolved construction.
        return build_provider(config)
    return build_provider(config, resolved=resolved)


def _maybe_trace_provider(provider: Any, config: dict[str, Any], task: str) -> Any:
    """Wrap *provider* in a LangSmith tracing decorator when enabled."""
    if provider is None:
        return provider

    ecosystem = config.get("ecosystem") or {}
    observability = ecosystem.get("observability") or {}

    if not observability.get("enabled"):
        return provider

    if observability.get("backend") != "langsmith":
        return provider

    from src.observability.langsmith_provider import LangSmithTracingProvider

    return LangSmithTracingProvider(
        provider,
        task=task,
        project_name=observability.get("project"),
    )


def build_services(paths: ProjectPaths, config: dict[str, Any]) -> dict[str, Any]:
    config_service = ConfigService(paths)
    manifest_service = ManifestService(paths)
    search_service = SearchService(paths)
    registry = ModelRegistryService()
    compile_provider = _maybe_trace_provider(
        _resolve_provider(config, registry, "update"), config, "update"
    )
    query_provider = _maybe_trace_provider(
        _resolve_provider(config, registry, "ask"), config, "ask"
    )
    review_provider = _maybe_trace_provider(
        _resolve_provider(config, registry, "review"), config, "review"
    )
    run_store = RunStore(paths.graph_exports_dir / "run_artifacts.sqlite3")
    compile_run_store = CompileRunStore(paths.graph_exports_dir / "compile_runs.json")

    ecosystem = config.get("ecosystem") or {}
    workflows = ecosystem.get("workflows") or {}
    query_backend = workflows.get("query_backend", "python")
    review_backend = workflows.get("review_backend", "python")

    return {
        "project": ProjectService(paths),
        "config": config_service,
        "manifest": manifest_service,
        "ingest": IngestService(paths, manifest_service),
        "compile": CompileService(
            paths,
            config,
            manifest_service,
            provider=compile_provider,
            compile_run_store=compile_run_store,
        ),
        "concepts": ConceptService(paths),
        "diff": DiffService(paths, manifest_service),
        "doctor": DoctorService(paths, config, provider=compile_provider),
        "lint": LintService(paths, config, manifest_service),
        "search": search_service,
        "status": StatusService(paths, manifest_service, config=config),
        "query": QueryService(
            paths,
            search_service,
            provider=query_provider,
            run_store=run_store,
            workflow_backend=query_backend,
        ),
        "export": ExportService(paths),
        "review": ReviewService(
            paths,
            provider=review_provider,
            run_store=run_store,
            workflow_backend=review_backend,
        ),
        "run_store": run_store,
        "compile_run_store": compile_run_store,
    }
