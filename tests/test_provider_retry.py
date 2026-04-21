"""Tests for provider retry policy (tenacity-backed)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.providers.base import ProviderRequest, ProviderResponse
from src.providers.retry import TRANSIENT_EXCEPTIONS, provider_retry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubProvider:
    """Minimal provider with a retryable generate()."""

    name = "stub"

    def __init__(self, mock_call: MagicMock) -> None:
        self._mock = mock_call

    @provider_retry()
    def generate(self, request: ProviderRequest) -> ProviderResponse:
        return self._mock(request)


# ---------------------------------------------------------------------------
# TRANSIENT_EXCEPTIONS tuple
# ---------------------------------------------------------------------------


def test_transient_exceptions_includes_stdlib_types() -> None:
    assert ConnectionError in TRANSIENT_EXCEPTIONS
    assert TimeoutError in TRANSIENT_EXCEPTIONS


def test_transient_exceptions_includes_openai_types() -> None:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    assert APITimeoutError in TRANSIENT_EXCEPTIONS
    assert APIConnectionError in TRANSIENT_EXCEPTIONS
    assert RateLimitError in TRANSIENT_EXCEPTIONS
    assert InternalServerError in TRANSIENT_EXCEPTIONS


def test_transient_exceptions_includes_anthropic_types() -> None:
    from anthropic import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )

    assert APITimeoutError in TRANSIENT_EXCEPTIONS
    assert APIConnectionError in TRANSIENT_EXCEPTIONS
    assert RateLimitError in TRANSIENT_EXCEPTIONS
    assert InternalServerError in TRANSIENT_EXCEPTIONS


# ---------------------------------------------------------------------------
# Retry behavior — success after transient failures
# ---------------------------------------------------------------------------


def test_retry_succeeds_after_transient_errors() -> None:
    mock = MagicMock(
        side_effect=[
            ConnectionError("reset"),
            TimeoutError("timed out"),
            ProviderResponse(text="ok", model_name="stub"),
        ]
    )
    provider = _StubProvider(mock)
    result = provider.generate(ProviderRequest(prompt="test"))
    assert result.text == "ok"
    assert mock.call_count == 3


# ---------------------------------------------------------------------------
# Retry exhaustion — propagates after max attempts
# ---------------------------------------------------------------------------


def test_retry_exhaustion_propagates_error() -> None:
    mock = MagicMock(
        side_effect=[
            ConnectionError("1"),
            ConnectionError("2"),
            ConnectionError("3"),
        ]
    )
    provider = _StubProvider(mock)
    with pytest.raises(ConnectionError):
        provider.generate(ProviderRequest(prompt="test"))
    assert mock.call_count == 3


# ---------------------------------------------------------------------------
# Non-retriable errors — propagate immediately
# ---------------------------------------------------------------------------


def test_non_retriable_error_propagates_immediately() -> None:
    mock = MagicMock(side_effect=ValueError("bad prompt"))
    provider = _StubProvider(mock)
    with pytest.raises(ValueError, match="bad prompt"):
        provider.generate(ProviderRequest(prompt="test"))
    assert mock.call_count == 1


def test_non_retriable_runtime_error_propagates_immediately() -> None:
    mock = MagicMock(side_effect=RuntimeError("something unexpected"))
    provider = _StubProvider(mock)
    with pytest.raises(RuntimeError, match="something unexpected"):
        provider.generate(ProviderRequest(prompt="test"))
    assert mock.call_count == 1


# ---------------------------------------------------------------------------
# Decorator factory returns independent decorators
# ---------------------------------------------------------------------------


def test_provider_retry_returns_new_decorator_each_call() -> None:
    d1 = provider_retry()
    d2 = provider_retry()
    assert d1 is not d2
