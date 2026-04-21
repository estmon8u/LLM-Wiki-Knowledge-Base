"""Phase 0 contract tests for the LangChain ecosystem transition.

These tests lock down the current provider, query, review, and service-factory
seams so that later phases (LangSmith, LangGraph, LangChain providers, hybrid
retrieval) can be added without silently breaking existing behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.providers.base import ProviderRequest, ProviderResponse, TextProvider
from src.services import build_services
from src.services.config_service import (
    CURRENT_CONFIG_VERSION,
    DEFAULT_CONFIG,
    ConfigService,
    _apply_config_migrations,
)
from src.services.doctor_service import DoctorService
from src.services.query_service import QueryAnswer, QueryService
from src.services.review_service import ReviewService
from src.models.wiki_models import ReviewReport
from src.storage.run_store import RunStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ContractStubProvider(TextProvider):
    """Deterministic provider that returns controlled text and model name."""

    name = "contract-stub"

    def __init__(self, text: str = "Stub answer.", model_name: str = "stub-1") -> None:
        self._text = text
        self._model_name = model_name

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(text=self._text, model_name=self._model_name)


class _ExplodingProvider(TextProvider):
    """Provider that raises on generate() to test error propagation."""

    name = "exploding"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Config version 3 and ecosystem defaults
# ---------------------------------------------------------------------------


def test_current_config_version_is_three() -> None:
    assert CURRENT_CONFIG_VERSION == 3


def test_default_config_includes_ecosystem_section() -> None:
    eco = DEFAULT_CONFIG.get("ecosystem")
    assert eco is not None
    assert eco["observability"]["backend"] == "none"
    assert eco["observability"]["enabled"] is False
    assert eco["workflows"]["query_backend"] == "python"
    assert eco["workflows"]["review_backend"] == "python"
    assert eco["providers"]["backend"] == "direct"
    assert eco["retrieval"]["mode"] == "lexical"


def test_config_migration_v2_to_v3_adds_ecosystem() -> None:
    v2 = {
        "version": 2,
        "project": {"name": "Test"},
        "storage": {"raw_dir": "raw/sources", "raw_normalized_dir": "raw/normalized"},
        "provider": {"name": "openai"},
    }
    migrated, changed = _apply_config_migrations(v2)

    assert changed is True
    assert migrated["version"] == 3
    assert "ecosystem" in migrated
    assert migrated["ecosystem"]["observability"]["backend"] == "none"
    assert migrated["ecosystem"]["providers"]["backend"] == "direct"
    assert migrated["provider"]["name"] == "openai"


def test_config_migration_v1_to_v3_chains() -> None:
    v1 = {
        "version": 1,
        "project": {"name": "Old"},
        "compile": {"summary_paragraph_limit": 2, "excerpt_character_limit": 200},
    }
    migrated, changed = _apply_config_migrations(v1)

    assert changed is True
    assert migrated["version"] == 3
    assert "summary_paragraph_limit" not in migrated["compile"]
    assert "ecosystem" in migrated


def test_config_service_loads_ecosystem_defaults(test_project) -> None:
    loaded = ConfigService(test_project.paths).load()
    eco = loaded.get("ecosystem")
    assert eco is not None
    assert eco["observability"]["enabled"] is False
    assert eco["retrieval"]["mode"] == "lexical"


def test_config_service_migrates_v2_file_to_v3(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 2\n"
        "project:\n"
        "  name: V2 Project\n"
        "provider:\n"
        "  name: openai\n",
        encoding="utf-8",
    )
    loaded = ConfigService(test_project.paths).load()
    assert loaded["version"] == 3
    assert "ecosystem" in loaded

    persisted = yaml.safe_load(
        test_project.paths.config_file.read_text(encoding="utf-8")
    )
    assert persisted["version"] == 3
    assert "ecosystem" in persisted


def test_config_preserves_custom_ecosystem_overrides(test_project) -> None:
    test_project.paths.config_file.write_text(
        "version: 3\n"
        "ecosystem:\n"
        "  observability:\n"
        "    backend: langsmith\n"
        "    enabled: true\n"
        "  retrieval:\n"
        "    mode: hybrid\n",
        encoding="utf-8",
    )
    loaded = ConfigService(test_project.paths).load()
    assert loaded["ecosystem"]["observability"]["backend"] == "langsmith"
    assert loaded["ecosystem"]["observability"]["enabled"] is True
    assert loaded["ecosystem"]["retrieval"]["mode"] == "hybrid"
    # Defaults still merge for unspecified sub-keys
    assert loaded["ecosystem"]["workflows"]["query_backend"] == "python"
    assert loaded["ecosystem"]["providers"]["backend"] == "direct"


# ---------------------------------------------------------------------------
# Doctor ecosystem check
# ---------------------------------------------------------------------------


def test_doctor_ecosystem_valid_defaults(test_project) -> None:
    report = test_project.services["doctor"].diagnose()
    eco_check = next(c for c in report.checks if c.name == "ecosystem_config")
    assert eco_check.ok is True
    assert eco_check.severity == "ok"


def test_doctor_ecosystem_warns_on_invalid_backend(test_project) -> None:
    config = dict(test_project.config)
    config["ecosystem"] = {"observability": {"backend": "bogus"}}
    doctor = DoctorService(test_project.paths, config)
    report = doctor.diagnose()
    eco_check = next(c for c in report.checks if c.name == "ecosystem_config")
    assert eco_check.ok is False
    assert eco_check.severity == "warning"
    assert "bogus" in eco_check.detail


def test_doctor_ecosystem_warns_on_invalid_workflow_backend(test_project) -> None:
    config = dict(test_project.config)
    config["ecosystem"] = {"workflows": {"query_backend": "invalid"}}
    doctor = DoctorService(test_project.paths, config)
    report = doctor.diagnose()
    eco_check = next(c for c in report.checks if c.name == "ecosystem_config")
    assert eco_check.ok is False
    assert "invalid" in eco_check.detail


def test_doctor_ecosystem_warns_on_invalid_retrieval_mode(test_project) -> None:
    config = dict(test_project.config)
    config["ecosystem"] = {"retrieval": {"mode": "magic"}}
    doctor = DoctorService(test_project.paths, config)
    report = doctor.diagnose()
    eco_check = next(c for c in report.checks if c.name == "ecosystem_config")
    assert eco_check.ok is False
    assert "magic" in eco_check.detail


def test_doctor_ecosystem_warns_on_invalid_provider_backend(test_project) -> None:
    config = dict(test_project.config)
    config["ecosystem"] = {"providers": {"backend": "mystery"}}
    doctor = DoctorService(test_project.paths, config)
    report = doctor.diagnose()
    eco_check = next(c for c in report.checks if c.name == "ecosystem_config")
    assert eco_check.ok is False
    assert "mystery" in eco_check.detail


def test_doctor_ecosystem_accepts_all_valid_options(test_project) -> None:
    config = dict(test_project.config)
    config["ecosystem"] = {
        "observability": {"backend": "langsmith", "enabled": True},
        "workflows": {"query_backend": "langgraph", "review_backend": "langgraph"},
        "providers": {"backend": "langchain"},
        "retrieval": {"mode": "semantic"},
    }
    doctor = DoctorService(test_project.paths, config)
    report = doctor.diagnose()
    eco_check = next(c for c in report.checks if c.name == "ecosystem_config")
    assert eco_check.ok is True


# ---------------------------------------------------------------------------
# Provider contract
# ---------------------------------------------------------------------------


def test_provider_contract_generate_returns_provider_response() -> None:
    provider = _ContractStubProvider(text="hello", model_name="test-model")
    request = ProviderRequest(prompt="say hi", system_prompt="be nice")
    response = provider.generate(request)

    assert isinstance(response, ProviderResponse)
    assert response.text == "hello"
    assert response.model_name == "test-model"


def test_provider_contract_propagates_exceptions() -> None:
    provider = _ExplodingProvider()
    request = ProviderRequest(prompt="fail")
    with pytest.raises(RuntimeError, match="boom"):
        provider.generate(request)


# ---------------------------------------------------------------------------
# Query service contract
# ---------------------------------------------------------------------------


def test_query_contract_single_answer_calls_provider_once(test_project) -> None:
    call_count = 0
    original_text = "Stub answer. [Source One]"

    class _CountingProvider(TextProvider):
        name = "counting"

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            nonlocal call_count
            call_count += 1
            return ProviderResponse(text=original_text, model_name="c-1")

    from tests.conftest import _StubProvider

    # Set up a source so search returns something
    from tests.test_compile_and_lint import _ingest_source

    _ingest_source(test_project, "notes/a.md", "# A\n\nBody.\n")
    test_project.services["compile"].compile()
    test_project.services["search"].refresh(force=True)

    qs = QueryService(
        test_project.paths,
        test_project.services["search"],
        provider=_CountingProvider(),
        run_store=None,
    )
    result = qs.answer_question("What is A?", limit=3, self_consistency=1)

    assert isinstance(result, QueryAnswer)
    assert call_count == 1
    assert result.answer == original_text


def test_query_contract_no_match_returns_without_provider_call(test_project) -> None:
    provider = _ContractStubProvider()
    qs = QueryService(
        test_project.paths,
        test_project.services["search"],
        provider=provider,
        run_store=None,
    )
    result = qs.answer_question("nonexistent topic", limit=3, self_consistency=1)

    assert isinstance(result, QueryAnswer)
    assert result.mode == "no-matches"


def test_query_contract_self_consistency_saves_run_record(test_project) -> None:
    from tests.test_compile_and_lint import _ingest_source

    _ingest_source(test_project, "notes/b.md", "# B\n\nContent about topic B.\n")
    test_project.services["compile"].compile()
    test_project.services["search"].refresh(force=True)

    run_store = RunStore(test_project.paths.graph_exports_dir / "runs_contract.sqlite3")
    provider = _ContractStubProvider(text="Claim about B. [B]", model_name="stub-1")
    qs = QueryService(
        test_project.paths,
        test_project.services["search"],
        provider=provider,
        run_store=run_store,
    )
    result = qs.answer_question("What is B?", limit=3, self_consistency=2)

    assert isinstance(result, QueryAnswer)
    assert result.run_id is not None

    runs = run_store.list_runs()
    assert len(runs) == 1
    assert runs[0].command == "query"


# ---------------------------------------------------------------------------
# Review service contract
# ---------------------------------------------------------------------------


def test_review_contract_basic_review_returns_report(test_project) -> None:
    from tests.test_compile_and_lint import _ingest_source

    _ingest_source(test_project, "notes/r.md", "# R\n\nReview content.\n")
    test_project.services["compile"].compile()

    rs = ReviewService(
        test_project.paths,
        provider=_ContractStubProvider(text="NO_CLAIMS"),
        run_store=None,
    )
    report = rs.review(adversarial=False)

    assert isinstance(report, ReviewReport)


def test_review_contract_deep_review_saves_run_record(test_project) -> None:
    from tests.test_compile_and_lint import _ingest_source

    _ingest_source(test_project, "notes/s.md", "# S\n\nDeep content.\n")
    _ingest_source(test_project, "notes/t.md", "# T\n\nAnother deep piece.\n")
    test_project.services["compile"].compile()

    run_store = RunStore(
        test_project.paths.graph_exports_dir / "runs_review_contract.sqlite3"
    )
    rs = ReviewService(
        test_project.paths,
        provider=_ContractStubProvider(text="NO_CLAIMS"),
        run_store=run_store,
    )
    report = rs.review(adversarial=True)

    assert isinstance(report, ReviewReport)
    assert report.run_id is not None

    runs = run_store.list_runs()
    assert len(runs) == 1
    assert runs[0].command == "review"


# ---------------------------------------------------------------------------
# Service factory contract
# ---------------------------------------------------------------------------


def test_build_services_returns_expected_keys(test_project) -> None:
    services = build_services(test_project.paths, test_project.config)
    expected = {
        "project",
        "config",
        "manifest",
        "ingest",
        "compile",
        "concepts",
        "diff",
        "doctor",
        "lint",
        "review",
        "search",
        "status",
        "query",
        "export",
        "run_store",
        "compile_run_store",
    }
    assert set(services) == expected


def test_build_services_deterministic_commands_without_provider(test_project) -> None:
    config = dict(test_project.config)
    config.pop("provider", None)
    services = build_services(test_project.paths, config)

    # Deterministic services should still be functional
    assert services["lint"] is not None
    assert services["search"] is not None
    assert services["status"] is not None
    assert services["export"] is not None


def test_build_services_query_has_run_store(test_project) -> None:
    services = build_services(test_project.paths, test_project.config)
    qs = services["query"]
    assert isinstance(qs, QueryService)
    assert qs.run_store is not None


def test_build_services_review_has_run_store(test_project) -> None:
    services = build_services(test_project.paths, test_project.config)
    rs = services["review"]
    assert isinstance(rs, ReviewService)
    assert rs.run_store is not None
