"""Tests for provider factory, provider classes, and provider-backed services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.providers import (
    ProviderConfigurationError,
    ProviderExecutionError,
    build_provider,
)
from src.providers.base import ProviderRequest, ProviderResponse, TextProvider
from src.services.query_service import QueryAnswer, QueryService
from src.services.review_service import ReviewService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeProvider(TextProvider):
    """In-process provider that returns a canned response."""

    name = "fake"

    def __init__(self, text: str = "LLM says hello", model: str = "fake-v1") -> None:
        self._text = text
        self._model = model

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(text=self._text, model_name=self._model)


class FailingProvider(TextProvider):
    """Provider that always raises."""

    name = "failing"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        raise RuntimeError("API is down")


# ---------------------------------------------------------------------------
# build_provider factory tests
# ---------------------------------------------------------------------------


def test_build_provider_returns_none_when_no_provider_section() -> None:
    assert build_provider({}) is None


def test_build_provider_returns_none_when_provider_name_empty() -> None:
    assert build_provider({"provider": {"name": ""}}) is None


def test_build_provider_returns_unavailable_provider_for_unknown_provider() -> None:
    provider = build_provider({"provider": {"name": "unknown-llm"}})

    assert provider is not None
    with pytest.raises(ProviderConfigurationError, match="Unknown provider name"):
        provider.generate(ProviderRequest(prompt="hi"))


def test_build_provider_returns_unavailable_provider_when_api_key_missing() -> None:
    with patch.dict("os.environ", {}, clear=True):
        provider = build_provider({"provider": {"name": "openai"}})
    assert provider is not None
    with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
        provider.generate(ProviderRequest(prompt="hi"))


def test_build_provider_creates_openai_provider() -> None:
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"}):
        with patch("src.providers.openai_provider.OpenAI"):
            provider = build_provider({"provider": {"name": "openai"}})
    assert provider is not None
    assert provider.name == "openai"
    assert provider.model == "gpt-5.4-mini"


def test_build_provider_creates_openai_provider_with_custom_model() -> None:
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        with patch("src.providers.openai_provider.OpenAI"):
            provider = build_provider(
                {"provider": {"name": "openai", "model": "gpt-5.4"}}
            )
    assert provider is not None
    assert provider.model == "gpt-5.4"


def test_build_provider_creates_anthropic_provider() -> None:
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-456"}):
        with patch("src.providers.anthropic_provider.Anthropic"):
            provider = build_provider({"provider": {"name": "anthropic"}})
    assert provider is not None
    assert provider.name == "anthropic"
    assert provider.model == "claude-sonnet-4-6"


def test_build_provider_creates_anthropic_provider_with_custom_model() -> None:
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("src.providers.anthropic_provider.Anthropic"):
            provider = build_provider(
                {"provider": {"name": "anthropic", "model": "claude-opus-4-6"}}
            )
    assert provider is not None
    assert provider.model == "claude-opus-4-6"


def test_build_provider_creates_gemini_provider() -> None:
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key-789"}):
        with patch("src.providers.gemini_provider.genai") as mock_genai:
            provider = build_provider({"provider": {"name": "gemini"}})
    assert provider is not None
    assert provider.name == "gemini"
    assert provider.model == "gemini-3.1-flash-lite-preview"


def test_build_provider_creates_gemini_provider_with_custom_model() -> None:
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        with patch("src.providers.gemini_provider.genai"):
            provider = build_provider(
                {"provider": {"name": "gemini", "model": "gemini-3.1-pro-preview"}}
            )
    assert provider is not None
    assert provider.model == "gemini-3.1-pro-preview"


def test_build_provider_respects_custom_api_key_env() -> None:
    with patch.dict("os.environ", {"MY_KEY": "test-key-custom"}):
        with patch("src.providers.openai_provider.OpenAI"):
            provider = build_provider(
                {"provider": {"name": "openai", "api_key_env": "MY_KEY"}}
            )
    assert provider is not None


# ---------------------------------------------------------------------------
# Individual provider generate() tests (mocked SDK calls)
# ---------------------------------------------------------------------------


def test_openai_provider_generate() -> None:
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
        with patch("src.providers.openai_provider.OpenAI") as MockClient:
            from src.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider(model="gpt-5.4-mini")
            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock()]
            mock_completion.choices[0].message.content = " Hello world "
            MockClient.return_value.chat.completions.create.return_value = (
                mock_completion
            )

            result = provider.generate(
                ProviderRequest(prompt="test", system_prompt="sys")
            )

    assert result.text == "Hello world"
    assert result.model_name == "gpt-5.4-mini"


def test_openai_provider_generate_without_system_prompt() -> None:
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
        with patch("src.providers.openai_provider.OpenAI") as MockClient:
            from src.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider()
            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock()]
            mock_completion.choices[0].message.content = "response"
            MockClient.return_value.chat.completions.create.return_value = (
                mock_completion
            )

            result = provider.generate(ProviderRequest(prompt="test"))

    call_kwargs = MockClient.return_value.chat.completions.create.call_args
    messages = call_kwargs.kwargs.get("messages", call_kwargs[1].get("messages", []))
    assert all(m["role"] != "developer" for m in messages)


def test_openai_provider_missing_key_raises() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            from src.providers.openai_provider import OpenAIProvider

            OpenAIProvider()


def test_anthropic_provider_generate() -> None:
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        with patch("src.providers.anthropic_provider.Anthropic") as MockClient:
            from src.providers.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(model="claude-sonnet-4-6")
            mock_message = MagicMock()
            mock_block = MagicMock()
            mock_block.type = "text"
            mock_block.text = " Claude says hi "
            mock_message.content = [mock_block]
            MockClient.return_value.messages.create.return_value = mock_message

            result = provider.generate(
                ProviderRequest(prompt="test", system_prompt="sys")
            )

    assert result.text == "Claude says hi"
    assert result.model_name == "claude-sonnet-4-6"


def test_anthropic_provider_generate_without_system_prompt() -> None:
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        with patch("src.providers.anthropic_provider.Anthropic") as MockClient:
            from src.providers.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider()
            mock_message = MagicMock()
            mock_block = MagicMock()
            mock_block.type = "text"
            mock_block.text = "response"
            mock_message.content = [mock_block]
            MockClient.return_value.messages.create.return_value = mock_message

            result = provider.generate(ProviderRequest(prompt="test"))

    call_kwargs = MockClient.return_value.messages.create.call_args
    assert "system" not in call_kwargs.kwargs


def test_anthropic_provider_missing_key_raises() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            from src.providers.anthropic_provider import AnthropicProvider

            AnthropicProvider()


def test_gemini_provider_generate() -> None:
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
        with patch("src.providers.gemini_provider.genai") as mock_genai:
            from src.providers.gemini_provider import GeminiProvider

            provider = GeminiProvider(model="gemini-3.1-flash-lite-preview")
            mock_response = MagicMock()
            mock_response.text = " Gemini says hi "
            mock_genai.Client.return_value.models.generate_content.return_value = (
                mock_response
            )

            result = provider.generate(
                ProviderRequest(prompt="test", system_prompt="sys")
            )

    assert result.text == "Gemini says hi"
    assert result.model_name == "gemini-3.1-flash-lite-preview"


def test_gemini_provider_generate_without_system_prompt() -> None:
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
        with patch("src.providers.gemini_provider.genai") as mock_genai:
            from src.providers.gemini_provider import GeminiProvider

            provider = GeminiProvider()
            mock_response = MagicMock()
            mock_response.text = "response"
            mock_genai.Client.return_value.models.generate_content.return_value = (
                mock_response
            )

            result = provider.generate(ProviderRequest(prompt="test"))

    assert result.text == "response"


def test_gemini_provider_missing_key_raises() -> None:
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            from src.providers.gemini_provider import GeminiProvider

            GeminiProvider()


# ---------------------------------------------------------------------------
# QueryService with provider
# ---------------------------------------------------------------------------


def test_query_service_uses_provider_when_available(test_project) -> None:
    test_project.write_file(
        "wiki/sources/alpha.md",
        "---\ntitle: Alpha\n---\n# Alpha\n\nKnowledge base traceability.\n",
    )
    provider = FakeProvider(text="The wiki tracks provenance via raw hashes.")
    from src.services.search_service import SearchService

    query_service = QueryService(
        test_project.paths,
        SearchService(test_project.paths),
        provider=provider,
    )

    answer = query_service.answer_question("How does traceability work?")

    assert "provenance" in answer.answer
    assert answer.mode.startswith("provider:")
    assert len(answer.citations) >= 1


def test_query_service_raises_without_provider(test_project) -> None:
    test_project.write_file(
        "wiki/sources/alpha.md",
        "---\ntitle: Alpha\n---\n# Alpha\n\nKnowledge base traceability.\n",
    )
    from src.services.search_service import SearchService

    query_service = QueryService(
        test_project.paths,
        SearchService(test_project.paths),
        provider=None,
    )

    with pytest.raises(ProviderConfigurationError, match="kb ask requires"):
        query_service.answer_question("knowledge base")


def test_query_service_raises_on_provider_error(test_project) -> None:
    test_project.write_file(
        "wiki/sources/alpha.md",
        "---\ntitle: Alpha\n---\n# Alpha\n\nKnowledge base traceability.\n",
    )
    provider = FailingProvider()
    from src.services.search_service import SearchService

    query_service = QueryService(
        test_project.paths,
        SearchService(test_project.paths),
        provider=provider,
    )

    with pytest.raises(ProviderExecutionError, match="API is down"):
        query_service.answer_question("knowledge base")


def test_query_service_no_matches_returns_fallback_regardless_of_provider(
    test_project,
) -> None:
    provider = FakeProvider()
    from src.services.search_service import SearchService

    query_service = QueryService(
        test_project.paths,
        SearchService(test_project.paths),
        provider=provider,
    )

    answer = query_service.answer_question("nonexistent garbage xyzzy")
    assert "No compiled wiki pages" in answer.answer
    assert len(answer.citations) == 0
    assert answer.mode == "no-matches"


# ---------------------------------------------------------------------------
# ReviewService with provider
# ---------------------------------------------------------------------------


def test_review_service_uses_provider_when_available(test_project) -> None:
    test_project.write_file(
        "wiki/sources/alpha.md",
        "Knowledge base markdown traceability.",
    )
    provider = FakeProvider(
        text="ISSUE|warning|stale-claim|wiki/sources/alpha.md|Claim may be outdated."
    )
    review_service = ReviewService(test_project.paths, provider=provider)

    report = review_service.review()

    assert report.mode == "provider:fake-v1"
    provider_issues = [i for i in report.issues if i.code == "stale-claim"]
    assert len(provider_issues) == 1
    assert provider_issues[0].severity == "warning"


def test_review_service_requires_provider(test_project) -> None:
    from src.providers import ProviderConfigurationError

    test_project.write_file("wiki/sources/alpha.md", "Some content here.")
    review_service = ReviewService(test_project.paths, provider=None)

    with pytest.raises(
        ProviderConfigurationError, match="requires a configured provider"
    ):
        review_service.review()


def test_review_service_falls_back_on_provider_error(test_project) -> None:
    test_project.write_file("wiki/sources/alpha.md", "Some content here.")
    provider = FailingProvider()
    review_service = ReviewService(test_project.paths, provider=provider)

    with pytest.raises(ProviderExecutionError, match="API is down"):
        review_service.review()


def test_review_service_provider_no_issues_response(test_project) -> None:
    test_project.write_file("wiki/sources/alpha.md", "Some content here.")
    provider = FakeProvider(text="NO_ISSUES")
    review_service = ReviewService(test_project.paths, provider=provider)

    report = review_service.review()

    provider_issues = [
        i
        for i in report.issues
        if i.code not in ("overlapping-topics", "terminology-variant")
    ]
    assert len(provider_issues) == 0


def test_review_service_provider_multiple_issues(test_project) -> None:
    test_project.write_file("wiki/sources/a.md", "Page A content.")
    test_project.write_file("wiki/sources/b.md", "Page B content.")
    provider = FakeProvider(
        text=(
            "ISSUE|error|contradiction|wiki/sources/a.md, wiki/sources/b.md|Conflicting claims about X.\n"
            "ISSUE|suggestion|redundant-content|wiki/sources/b.md|Repeats content from A."
        )
    )
    review_service = ReviewService(test_project.paths, provider=provider)

    report = review_service.review()

    contradiction = [i for i in report.issues if i.code == "contradiction"]
    redundant = [i for i in report.issues if i.code == "redundant-content"]
    assert len(contradiction) == 1
    assert contradiction[0].severity == "error"
    assert len(contradiction[0].pages) == 2
    assert len(redundant) == 1


def test_review_service_provider_skips_malformed_lines(test_project) -> None:
    test_project.write_file("wiki/sources/a.md", "Content.")
    provider = FakeProvider(
        text=(
            "ISSUE|warning|valid-code|wiki/sources/a.md|Valid issue.\n"
            "ISSUE|too-few-parts\n"
            "Some random text\n"
            "ISSUE|suggestion|another|wiki/sources/a.md|Another issue."
        )
    )
    review_service = ReviewService(test_project.paths, provider=provider)

    report = review_service.review()

    provider_codes = [
        i.code
        for i in report.issues
        if i.code not in ("overlapping-topics", "terminology-variant")
    ]
    assert "valid-code" in provider_codes
    assert "another" in provider_codes
    assert len(provider_codes) == 2


def test_review_service_provider_normalizes_unknown_severity(test_project) -> None:
    test_project.write_file("wiki/sources/a.md", "Content.")
    provider = FakeProvider(
        text="ISSUE|critical|bad-severity|wiki/sources/a.md|Unknown severity level."
    )
    review_service = ReviewService(test_project.paths, provider=provider)

    report = review_service.review()

    bad_sev = [i for i in report.issues if i.code == "bad-severity"]
    assert len(bad_sev) == 1
    assert bad_sev[0].severity == "suggestion"


# ---------------------------------------------------------------------------
# _parse_provider_issues unit tests
# ---------------------------------------------------------------------------


def test_parse_provider_issues_valid_line() -> None:
    raw = "ISSUE|error|contradiction|a.md, b.md|Pages disagree."
    issues = ReviewService._parse_provider_issues(raw)
    assert len(issues) == 1
    assert issues[0].code == "contradiction"
    assert issues[0].pages == ["a.md", "b.md"]


def test_parse_provider_issues_no_issues() -> None:
    assert ReviewService._parse_provider_issues("NO_ISSUES") == []


def test_parse_provider_issues_empty_string() -> None:
    assert ReviewService._parse_provider_issues("") == []


def test_parse_provider_issues_skips_non_issue_lines() -> None:
    raw = "Some preamble\nISSUE|warning|x|a.md|msg\nMore text"
    issues = ReviewService._parse_provider_issues(raw)
    assert len(issues) == 1


# ---------------------------------------------------------------------------
# Provider base class updated fields
# ---------------------------------------------------------------------------


def test_provider_request_max_tokens_default() -> None:
    req = ProviderRequest(prompt="hi")
    assert req.max_tokens == 1024


def test_provider_request_custom_max_tokens() -> None:
    req = ProviderRequest(prompt="hi", max_tokens=512)
    assert req.max_tokens == 512
