"""Phase 1 tests — LangSmith observability wrapper and service-factory wiring."""

from __future__ import annotations

import pytest

from src.observability.langsmith_provider import LangSmithTracingProvider
from src.providers.base import ProviderRequest, ProviderResponse, TextProvider
from src.services import _maybe_trace_provider, build_services


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubProvider(TextProvider):
    name = "stub"

    def __init__(self, text: str = "ok", model_name: str = "s-1") -> None:
        self._text = text
        self._model_name = model_name

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(text=self._text, model_name=self._model_name)


class _BoomProvider(TextProvider):
    name = "boom"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        raise RuntimeError("provider exploded")


# ---------------------------------------------------------------------------
# LangSmithTracingProvider unit tests
# ---------------------------------------------------------------------------


def test_wrapper_delegates_to_wrapped_provider() -> None:
    inner = _StubProvider(text="hello world", model_name="test-m")
    wrapper = LangSmithTracingProvider(inner, task="ask")
    response = wrapper.generate(ProviderRequest(prompt="hi"))

    assert response.text == "hello world"
    assert response.model_name == "test-m"


def test_wrapper_preserves_provider_name() -> None:
    inner = _StubProvider()
    wrapper = LangSmithTracingProvider(inner, task="update")
    assert wrapper.name == "stub"


def test_wrapper_preserves_task_and_project() -> None:
    inner = _StubProvider()
    wrapper = LangSmithTracingProvider(inner, task="review", project_name="my-proj")
    assert wrapper.task == "review"
    assert wrapper.project_name == "my-proj"


def test_wrapper_propagates_provider_exceptions() -> None:
    inner = _BoomProvider()
    wrapper = LangSmithTracingProvider(inner, task="ask")
    with pytest.raises(RuntimeError, match="provider exploded"):
        wrapper.generate(ProviderRequest(prompt="fail"))


def test_wrapper_returns_provider_response_type() -> None:
    inner = _StubProvider(text="typed", model_name="t-1")
    wrapper = LangSmithTracingProvider(inner, task="ask")
    result = wrapper.generate(ProviderRequest(prompt="test"))
    assert isinstance(result, ProviderResponse)


# ---------------------------------------------------------------------------
# _maybe_trace_provider unit tests
# ---------------------------------------------------------------------------


def test_maybe_trace_returns_none_when_provider_is_none() -> None:
    result = _maybe_trace_provider(None, {}, "ask")
    assert result is None


def test_maybe_trace_returns_unwrapped_when_disabled() -> None:
    inner = _StubProvider()
    config = {
        "ecosystem": {"observability": {"enabled": False, "backend": "langsmith"}}
    }
    result = _maybe_trace_provider(inner, config, "ask")
    assert result is inner
    assert not isinstance(result, LangSmithTracingProvider)


def test_maybe_trace_returns_unwrapped_when_backend_not_langsmith() -> None:
    inner = _StubProvider()
    config = {"ecosystem": {"observability": {"enabled": True, "backend": "none"}}}
    result = _maybe_trace_provider(inner, config, "ask")
    assert result is inner


def test_maybe_trace_wraps_when_enabled_and_langsmith() -> None:
    inner = _StubProvider()
    config = {
        "ecosystem": {
            "observability": {
                "enabled": True,
                "backend": "langsmith",
                "project": "test-proj",
            }
        }
    }
    result = _maybe_trace_provider(inner, config, "review")
    assert isinstance(result, LangSmithTracingProvider)
    assert result.wrapped is inner
    assert result.task == "review"
    assert result.project_name == "test-proj"


def test_maybe_trace_returns_unwrapped_when_no_ecosystem_key() -> None:
    inner = _StubProvider()
    result = _maybe_trace_provider(inner, {}, "ask")
    assert result is inner


def test_maybe_trace_returns_unwrapped_when_ecosystem_empty() -> None:
    inner = _StubProvider()
    result = _maybe_trace_provider(inner, {"ecosystem": {}}, "ask")
    assert result is inner


# ---------------------------------------------------------------------------
# build_services integration: observability wiring
# ---------------------------------------------------------------------------


def test_build_services_does_not_wrap_when_observability_disabled(test_project) -> None:
    config = dict(test_project.config)
    config["ecosystem"] = {"observability": {"enabled": False, "backend": "langsmith"}}
    services = build_services(test_project.paths, config)

    # query and review providers should NOT be wrapped
    assert not isinstance(services["query"].provider, LangSmithTracingProvider)
    assert not isinstance(services["review"].provider, LangSmithTracingProvider)


def test_build_services_wraps_providers_when_observability_enabled(
    test_project,
) -> None:
    config = dict(test_project.config)
    config["provider"] = {"name": "openai"}
    config["ecosystem"] = {
        "observability": {"enabled": True, "backend": "langsmith", "project": "ci"}
    }
    services = build_services(test_project.paths, config)

    # All three provider-bearing services should be wrapped
    assert isinstance(services["query"].provider, LangSmithTracingProvider)
    assert isinstance(services["review"].provider, LangSmithTracingProvider)
    assert isinstance(services["compile"].provider, LangSmithTracingProvider)

    # The wrapped provider should still delegate correctly
    assert services["query"].provider.task == "ask"
    assert services["review"].provider.task == "review"
    assert services["compile"].provider.task == "update"
