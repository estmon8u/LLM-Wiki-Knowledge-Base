"""Tests for provider factory, provider classes, and provider-backed services."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from graphwiki_kb.providers import (
    ProviderConfigurationError,
    ProviderExecutionError,
    build_provider,
)
from graphwiki_kb.providers.base import ProviderRequest, ProviderResponse, TextProvider
from graphwiki_kb.services.query_service import QueryService
from graphwiki_kb.services.review_service import ReviewService

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeProvider(TextProvider):
    """In-process provider that returns a canned response."""

    name = "fake"

    def __init__(self, text: str = "LLM says hello", model: str = "fake-v1") -> None:
        """Initializes the instance.

        Args:
            text: Text content being processed.
            model: Model value used by the operation.
        """
        self._text = text
        self._model = model

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate.

        Args:
            request: Request value used by the operation.

        Returns:
            ProviderResponse produced by the operation.
        """
        return ProviderResponse(text=self._text, model_name=self._model)


class FailingProvider(TextProvider):
    """Provider that always raises."""

    name = "failing"

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate.

        Args:
            request: Request value used by the operation.

        Returns:
            ProviderResponse produced by the operation.
        """
        raise RuntimeError("API is down")


# ---------------------------------------------------------------------------
# build_provider factory tests
# ---------------------------------------------------------------------------


def test_build_provider_returns_none_when_no_provider_section() -> None:
    """Verifies that build provider returns none when no provider section."""
    assert build_provider({}) is None


def test_build_provider_returns_none_when_provider_name_empty() -> None:
    """Verifies that build provider returns none when provider name empty."""
    assert build_provider({"provider": {"name": ""}}) is None


def test_build_provider_returns_unavailable_provider_for_unknown_provider() -> None:
    """Verifies that build provider returns unavailable provider for unknown provider."""
    provider = build_provider({"provider": {"name": "unknown-llm"}})

    assert provider is not None
    with pytest.raises(ProviderConfigurationError, match="Unknown provider name"):
        provider.generate(ProviderRequest(prompt="hi"))


def test_build_provider_returns_unavailable_provider_when_api_key_missing() -> None:
    """Verifies that build provider returns unavailable provider when api key missing."""
    with patch.dict("os.environ", {}, clear=True):
        provider = build_provider({"provider": {"name": "openai"}})
    assert provider is not None
    with pytest.raises(ProviderConfigurationError, match="OPENAI_API_KEY"):
        provider.generate(ProviderRequest(prompt="hi"))


def test_build_provider_creates_openai_provider() -> None:
    """Verifies that build provider creates openai provider."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"}):
        with patch("graphwiki_kb.providers.openai_provider.OpenAI"):
            provider = build_provider({"provider": {"name": "openai"}})
    assert provider is not None
    assert provider.name == "openai"
    assert provider.model == "gpt-5.4-nano"


def test_build_provider_uses_catalog_defaults_for_openai() -> None:
    """Verifies that build provider uses catalog defaults for openai."""
    config = {
        "provider": {"name": "openai"},
        "providers": {
            "openai": {
                "model": "gpt-5.4",
                "api_key_env": "OPENAI_ALT_KEY",
                "reasoning_effort": "medium",
            },
            "anthropic": {
                "model": "claude-sonnet-4-6",
                "api_key_env": "ANTHROPIC_API_KEY",
                "thinking_budget": 10_000,
            },
            "gemini": {
                "model": "gemini-2.5-flash",
                "api_key_env": "GEMINI_API_KEY",
                "reasoning_effort": "high",
            },
        },
    }

    with patch.dict("os.environ", {"OPENAI_ALT_KEY": "test-key"}):
        with patch("graphwiki_kb.providers.openai_provider.OpenAI"):
            provider = build_provider(config)

    assert provider is not None
    assert provider.model == "gpt-5.4"
    assert provider._reasoning_effort == "medium"


def test_build_provider_creates_openai_provider_with_custom_model() -> None:
    """Verifies that build provider creates openai provider with custom model."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        with patch("graphwiki_kb.providers.openai_provider.OpenAI"):
            provider = build_provider(
                {"provider": {"name": "openai", "model": "gpt-5.4"}}
            )
    assert provider is not None
    assert provider.model == "gpt-5.4"


def test_build_provider_creates_anthropic_provider() -> None:
    """Verifies that build provider creates anthropic provider."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key-456"}):
        with patch("graphwiki_kb.providers.anthropic_provider.Anthropic"):
            provider = build_provider({"provider": {"name": "anthropic"}})
    assert provider is not None
    assert provider.name == "anthropic"
    assert provider.model == "claude-sonnet-4-6"


def test_build_provider_uses_catalog_thinking_budget_for_anthropic() -> None:
    """Verifies that build provider uses catalog thinking budget for anthropic."""
    config = {
        "provider": {"name": "anthropic"},
        "providers": {
            "openai": {
                "model": "gpt-5.4-nano",
                "api_key_env": "OPENAI_API_KEY",
                "reasoning_effort": "high",
            },
            "anthropic": {
                "model": "claude-sonnet-4-6",
                "api_key_env": "ANTHROPIC_ALT_KEY",
                "thinking_budget": 2048,
            },
            "gemini": {
                "model": "gemini-2.5-flash",
                "api_key_env": "GEMINI_API_KEY",
                "reasoning_effort": "high",
            },
        },
    }

    with patch.dict("os.environ", {"ANTHROPIC_ALT_KEY": "test-key"}):
        with patch("graphwiki_kb.providers.anthropic_provider.Anthropic"):
            provider = build_provider(config)

    assert provider is not None
    assert provider.model == "claude-sonnet-4-6"
    assert provider._thinking_budget == 2048


def test_build_provider_creates_anthropic_provider_with_custom_model() -> None:
    """Verifies that build provider creates anthropic provider with custom model."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        with patch("graphwiki_kb.providers.anthropic_provider.Anthropic"):
            provider = build_provider(
                {"provider": {"name": "anthropic", "model": "claude-opus-4-6"}}
            )
    assert provider is not None
    assert provider.model == "claude-opus-4-6"


def test_build_provider_creates_gemini_provider() -> None:
    """Verifies that build provider creates gemini provider."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key-789"}):
        with patch("graphwiki_kb.providers.gemini_provider.genai"):
            provider = build_provider({"provider": {"name": "gemini"}})
    assert provider is not None
    assert provider.name == "gemini"
    assert provider.model == "gemini-2.5-flash"


def test_build_provider_creates_gemini_provider_with_custom_model() -> None:
    """Verifies that build provider creates gemini provider with custom model."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        with patch("graphwiki_kb.providers.gemini_provider.genai"):
            provider = build_provider(
                {"provider": {"name": "gemini", "model": "gemini-3.1-pro-preview"}}
            )
    assert provider is not None
    assert provider.model == "gemini-3.1-pro-preview"


def test_build_provider_respects_custom_api_key_env() -> None:
    """Verifies that build provider respects custom api key env."""
    with patch.dict("os.environ", {"MY_KEY": "test-key-custom"}):
        with patch("graphwiki_kb.providers.openai_provider.OpenAI"):
            provider = build_provider(
                {"provider": {"name": "openai", "api_key_env": "MY_KEY"}}
            )
    assert provider is not None


def test_build_provider_ignores_ambiguous_active_provider_overrides_in_v3() -> None:
    """Verifies that build provider ignores ambiguous active provider overrides in v3."""
    config = {
        "version": 3,
        "provider": {
            "name": "openai",
            "model": "should-not-win",
            "api_key_env": "WRONG_KEY",
        },
        "providers": {
            "openai": {
                "model": "gpt-5.4",
                "api_key_env": "OPENAI_RIGHT_KEY",
                "reasoning_effort": "medium",
            },
            "anthropic": {
                "model": "claude-sonnet-4-6",
                "api_key_env": "ANTHROPIC_API_KEY",
                "thinking_budget": 10_000,
            },
            "gemini": {
                "model": "gemini-2.5-flash",
                "api_key_env": "GEMINI_API_KEY",
                "reasoning_effort": "high",
            },
        },
    }

    with patch.dict("os.environ", {"OPENAI_RIGHT_KEY": "test-key"}):
        with patch("graphwiki_kb.providers.openai_provider.OpenAI"):
            provider = build_provider(config)

    assert provider is not None
    assert provider.model == "gpt-5.4"
    assert provider._reasoning_effort == "medium"


# ---------------------------------------------------------------------------
# Individual provider generate() tests (mocked SDK calls)
# ---------------------------------------------------------------------------


def test_openai_provider_generate() -> None:
    """Verifies that openai provider generate."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
        with patch("graphwiki_kb.providers.openai_provider.OpenAI") as MockClient:
            from graphwiki_kb.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider(model="gpt-5.4-nano")
            mock_response = MagicMock()
            mock_response.output_text = " Hello world "
            mock_response.model = "gpt-5.4-nano"
            MockClient.return_value.responses.create.return_value = mock_response

            result = provider.generate(
                ProviderRequest(prompt="test", system_prompt="sys")
            )

    assert result.text == "Hello world"
    assert result.model_name == "gpt-5.4-nano"


def test_openai_provider_uses_request_reasoning_effort_override() -> None:
    """Verifies that openai provider uses request reasoning effort override."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
        with patch("graphwiki_kb.providers.openai_provider.OpenAI") as MockClient:
            from graphwiki_kb.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider(reasoning_effort="high")
            mock_response = MagicMock()
            mock_response.output_text = "response"
            MockClient.return_value.responses.create.return_value = mock_response

            provider.generate(ProviderRequest(prompt="test", reasoning_effort="low"))

    call_kwargs = MockClient.return_value.responses.create.call_args.kwargs
    assert call_kwargs["reasoning"] == {"effort": "low"}


def test_openai_provider_does_not_send_reasoning_to_non_reasoning_model() -> None:
    """Regression: non-reasoning OpenAI models should not receive reasoning args."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
        with patch("graphwiki_kb.providers.openai_provider.OpenAI") as MockClient:
            from graphwiki_kb.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider(model="gpt-4o-mini")
            mock_response = MagicMock()
            mock_response.output_text = "response"
            MockClient.return_value.responses.create.return_value = mock_response

            provider.generate(ProviderRequest(prompt="test"))

    call_kwargs = MockClient.return_value.responses.create.call_args.kwargs
    assert "reasoning" not in call_kwargs


def test_openai_provider_validates_api_mode_and_reasoning_effort() -> None:
    """Verifies provider config fails before unsupported OpenAI kwargs are sent."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
        with patch("graphwiki_kb.providers.openai_provider.OpenAI"):
            from graphwiki_kb.providers.openai_provider import OpenAIProvider

            with pytest.raises(ValueError, match="Unsupported OpenAI API mode"):
                OpenAIProvider(api="legacy")  # type: ignore[arg-type]
            with pytest.raises(ValueError, match="Unsupported OpenAI reasoning"):
                OpenAIProvider(reasoning_effort="extreme")


def test_openai_provider_uses_responses_json_schema_contract() -> None:
    """Verifies Responses API structured-output payload shape."""
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
        with patch("graphwiki_kb.providers.openai_provider.OpenAI") as MockClient:
            from graphwiki_kb.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider()
            mock_response = MagicMock()
            mock_response.output_text = '{"answer": "ok"}'
            MockClient.return_value.responses.create.return_value = mock_response

            provider.generate(
                ProviderRequest(
                    prompt="test",
                    response_schema=schema,
                    response_schema_name="answer_payload",
                )
            )

    call_kwargs = MockClient.return_value.responses.create.call_args.kwargs
    assert call_kwargs["text"] == {
        "format": {
            "type": "json_schema",
            "name": "answer_payload",
            "schema": schema,
            "strict": True,
        }
    }


def test_openai_provider_generate_without_system_prompt() -> None:
    """Verifies that openai provider generate without system prompt."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
        with patch("graphwiki_kb.providers.openai_provider.OpenAI") as MockClient:
            from graphwiki_kb.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider()
            mock_response = MagicMock()
            mock_response.output_text = "response"
            MockClient.return_value.responses.create.return_value = mock_response

            result = provider.generate(ProviderRequest(prompt="test"))

    assert result.text == "response"
    call_kwargs = MockClient.return_value.responses.create.call_args.kwargs
    assert "instructions" not in call_kwargs


def test_openai_responses_helpers_parse_nested_output_items() -> None:
    """Verifies Responses API helper fallback parsing for SDK-like objects."""
    from graphwiki_kb.providers.openai_provider import (
        _response_finish_reason,
        _response_output_text,
    )

    message = MagicMock()
    message.type = "message"
    message.content = [
        MagicMock(text="Part one."),
        {"text": "Part two."},
    ]
    response = MagicMock()
    response.output_text = None
    response.status = ""
    response.output = [
        message,
        {"type": "message", "content": [{"text": "Part three."}]},
    ]

    assert _response_output_text(response) == "Part one.\nPart two.\nPart three."

    status_response = MagicMock()
    status_response.status = "completed"
    status_response.output = []
    assert _response_finish_reason(status_response) == "completed"

    finish_response = MagicMock()
    finish_response.status = None
    finish_response.output = [{"finish_reason": "stop"}]
    assert _response_finish_reason(finish_response) == "stop"


def test_openai_responses_helpers_handle_direct_output_and_empty_values() -> None:
    """Verifies Responses API helpers handle dict output and missing text."""
    from graphwiki_kb.providers.openai_provider import (
        _response_finish_reason,
        _response_output_text,
    )

    direct_response = MagicMock()
    direct_response.output_text = None
    direct_response.output = [{"type": "output_text", "text": "Direct text."}]
    assert _response_output_text(direct_response) == "Direct text."

    empty_response = MagicMock()
    empty_response.output_text = 123
    empty_response.output = None
    assert _response_output_text(empty_response) == ""
    assert _response_finish_reason(empty_response) is None


def test_openai_provider_keeps_chat_completions_fallback() -> None:
    """Verifies that OpenAI can still use Chat Completions as an explicit fallback."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
        with patch("graphwiki_kb.providers.openai_provider.OpenAI") as MockClient:
            from graphwiki_kb.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider(api="chat_completions")
            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock()]
            mock_completion.choices[0].message.content = " fallback "
            MockClient.return_value.chat.completions.create.return_value = (
                mock_completion
            )

            result = provider.generate(ProviderRequest(prompt="test"))

    assert result.text == "fallback"
    assert MockClient.return_value.chat.completions.create.called


def test_openai_chat_fallback_omits_reasoning_for_non_reasoning_model() -> None:
    """Verifies chat fallback also gates reasoning_effort by model family."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
        with patch("graphwiki_kb.providers.openai_provider.OpenAI") as MockClient:
            from graphwiki_kb.providers.openai_provider import OpenAIProvider

            provider = OpenAIProvider(model="gpt-4o-mini", api="chat_completions")
            mock_completion = MagicMock()
            mock_completion.choices = [MagicMock()]
            mock_completion.choices[0].message.content = " fallback "
            MockClient.return_value.chat.completions.create.return_value = (
                mock_completion
            )

            provider.generate(ProviderRequest(prompt="test"))

    call_kwargs = MockClient.return_value.chat.completions.create.call_args.kwargs
    assert "reasoning_effort" not in call_kwargs


def test_openai_provider_missing_key_raises() -> None:
    """Verifies that openai provider missing key raises."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            from graphwiki_kb.providers.openai_provider import OpenAIProvider

            OpenAIProvider()


def test_anthropic_provider_generate() -> None:
    """Verifies that anthropic provider generate."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        with patch("graphwiki_kb.providers.anthropic_provider.Anthropic") as MockClient:
            from graphwiki_kb.providers.anthropic_provider import AnthropicProvider

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
    call_kwargs = MockClient.return_value.messages.create.call_args.kwargs
    assert call_kwargs["thinking"] == {"type": "adaptive"}
    assert call_kwargs["output_config"] == {"effort": "medium"}


def test_anthropic_provider_adaptive_thinking_supports_new_claude_4_models() -> None:
    """Verifies future Claude 4 minor releases are not excluded by a hardcoded list."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        with patch("graphwiki_kb.providers.anthropic_provider.Anthropic") as MockClient:
            from graphwiki_kb.providers.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(model="claude-sonnet-4-7")
            mock_message = MagicMock()
            mock_block = MagicMock()
            mock_block.type = "text"
            mock_block.text = "response"
            mock_message.content = [mock_block]
            MockClient.return_value.messages.create.return_value = mock_message

            provider.generate(ProviderRequest(prompt="test", reasoning_effort="low"))

    call_kwargs = MockClient.return_value.messages.create.call_args.kwargs
    assert call_kwargs["thinking"] == {"type": "adaptive"}
    assert call_kwargs["output_config"] == {"effort": "low"}


def test_anthropic_provider_validates_adaptive_thinking_effort() -> None:
    """Verifies unsupported adaptive-thinking efforts fail before SDK calls."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        with patch("graphwiki_kb.providers.anthropic_provider.Anthropic"):
            from graphwiki_kb.providers.anthropic_provider import AnthropicProvider

            with pytest.raises(ValueError, match="Unsupported Anthropic thinking"):
                AnthropicProvider(thinking_effort="extreme")


def test_anthropic_provider_keeps_manual_thinking_for_older_models() -> None:
    """Verifies that older Claude models retain manual thinking budgets."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        with patch("graphwiki_kb.providers.anthropic_provider.Anthropic") as MockClient:
            from graphwiki_kb.providers.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(
                model="claude-3-7-sonnet-latest",
                thinking_budget=2048,
            )
            mock_message = MagicMock()
            mock_block = MagicMock()
            mock_block.type = "text"
            mock_block.text = "response"
            mock_message.content = [mock_block]
            MockClient.return_value.messages.create.return_value = mock_message

            provider.generate(ProviderRequest(prompt="test"))

    call_kwargs = MockClient.return_value.messages.create.call_args.kwargs
    assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "output_config" not in call_kwargs


def test_anthropic_provider_does_not_send_adaptive_thinking_to_older_claude() -> None:
    """Verifies unsupported older Claude models do not receive adaptive thinking."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        with patch("graphwiki_kb.providers.anthropic_provider.Anthropic") as MockClient:
            from graphwiki_kb.providers.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(model="claude-sonnet-4-5")
            mock_message = MagicMock()
            mock_block = MagicMock()
            mock_block.type = "text"
            mock_block.text = "response"
            mock_message.content = [mock_block]
            MockClient.return_value.messages.create.return_value = mock_message

            provider.generate(ProviderRequest(prompt="test"))

    call_kwargs = MockClient.return_value.messages.create.call_args.kwargs
    assert "thinking" not in call_kwargs
    assert "output_config" not in call_kwargs


def test_anthropic_provider_uses_output_config_format_for_schema() -> None:
    """Anthropic structured output should be API-enforced, not prompt-only."""
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        with patch("graphwiki_kb.providers.anthropic_provider.Anthropic") as MockClient:
            from graphwiki_kb.providers.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider(model="claude-sonnet-4-6")
            mock_message = MagicMock()
            mock_block = MagicMock()
            mock_block.type = "text"
            mock_block.text = '{"answer":"ok"}'
            mock_message.content = [mock_block]
            MockClient.return_value.messages.create.return_value = mock_message

            provider.generate(ProviderRequest(prompt="test", response_schema=schema))

    call_kwargs = MockClient.return_value.messages.create.call_args.kwargs
    assert call_kwargs["output_config"]["effort"] == "medium"
    assert call_kwargs["output_config"]["format"] == {
        "type": "json_schema",
        "schema": schema,
    }
    assert "properties" not in call_kwargs["system"]


def test_anthropic_provider_generate_without_system_prompt() -> None:
    """Verifies that anthropic provider generate without system prompt."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        with patch("graphwiki_kb.providers.anthropic_provider.Anthropic") as MockClient:
            from graphwiki_kb.providers.anthropic_provider import AnthropicProvider

            provider = AnthropicProvider()
            mock_message = MagicMock()
            mock_block = MagicMock()
            mock_block.type = "text"
            mock_block.text = "response"
            mock_message.content = [mock_block]
            MockClient.return_value.messages.create.return_value = mock_message

            provider.generate(ProviderRequest(prompt="test"))

    call_kwargs = MockClient.return_value.messages.create.call_args
    assert "system" not in call_kwargs.kwargs


def test_anthropic_provider_missing_key_raises() -> None:
    """Verifies that anthropic provider missing key raises."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            from graphwiki_kb.providers.anthropic_provider import AnthropicProvider

            AnthropicProvider()


def test_gemini_provider_generate() -> None:
    """Verifies that gemini provider generate."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
        with patch("graphwiki_kb.providers.gemini_provider.genai") as mock_genai:
            from graphwiki_kb.providers.gemini_provider import GeminiProvider

            provider = GeminiProvider(model="gemini-2.5-flash")
            mock_response = MagicMock()
            mock_response.text = " Gemini says hi "
            mock_genai.Client.return_value.models.generate_content.return_value = (
                mock_response
            )

            result = provider.generate(
                ProviderRequest(prompt="test", system_prompt="sys")
            )

    assert result.text == "Gemini says hi"
    assert result.model_name == "gemini-2.5-flash"
    call_kwargs = (
        mock_genai.Client.return_value.models.generate_content.call_args.kwargs
    )
    assert call_kwargs["config"].thinking_config.thinking_budget == 24576
    assert call_kwargs["config"].thinking_config.thinking_level is None


def test_gemini_provider_uses_thinking_level_for_gemini_3_models() -> None:
    """Verifies Gemini 3 models receive thinking_level instead of thinking_budget."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
        with patch("graphwiki_kb.providers.gemini_provider.genai") as mock_genai:
            from graphwiki_kb.providers.gemini_provider import GeminiProvider

            provider = GeminiProvider(model="gemini-3-flash-preview")
            mock_response = MagicMock()
            mock_response.text = "response"
            mock_genai.Client.return_value.models.generate_content.return_value = (
                mock_response
            )

            provider.generate(ProviderRequest(prompt="test", reasoning_effort="low"))

    call_kwargs = (
        mock_genai.Client.return_value.models.generate_content.call_args.kwargs
    )
    assert call_kwargs["config"].thinking_config.thinking_level.value == "LOW"
    assert call_kwargs["config"].thinking_config.thinking_budget is None


def test_gemini_response_schema_preserves_additional_properties() -> None:
    """Verifies current Gemini JSON Schema support preserves object strictness."""
    from graphwiki_kb.providers.gemini_provider import (
        _gemini_response_schema,
        _gemini_response_schema_with_report,
    )

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"name": {"type": "string"}},
                },
            }
        },
    }

    converted = _gemini_response_schema(schema)
    converted_with_report, report = _gemini_response_schema_with_report(schema)

    assert converted == schema
    assert converted_with_report == schema
    assert report.removed_keywords == ()
    assert report.weakened is False


def test_gemini_response_schema_removes_unsupported_keywords() -> None:
    """Gemini receives the supported JSON Schema subset with weakening reported."""
    from graphwiki_kb.providers.gemini_provider import (
        _gemini_response_schema_with_report,
    )

    schema = {
        "$defs": {"item": {"type": "string", "pattern": "^[A-Z]+$"}},
        "type": "object",
        "properties": {
            "name": {"$ref": "#/$defs/item", "default": "RAG"},
            "maybe": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "oneOf": [{"required": ["name"]}],
    }

    converted, report = _gemini_response_schema_with_report(schema)

    assert converted == {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "maybe": {"type": ["string", "null"]},
        },
    }
    assert report.weakened is True
    assert "$.$defs" in report.removed_keywords
    assert "$.oneOf" in report.removed_keywords
    assert "$.properties.name.default" in report.removed_keywords
    assert "$.properties.name.pattern" in report.removed_keywords


def test_gemini_provider_sends_strict_schema_without_downgrade_warning(caplog) -> None:
    """Verifies Gemini receives current strict JSON Schema payloads."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
        with patch("graphwiki_kb.providers.gemini_provider.genai") as mock_genai:
            from graphwiki_kb.providers.gemini_provider import GeminiProvider

            provider = GeminiProvider(model="gemini-2.5-flash")
            mock_response = MagicMock()
            mock_response.text = "{}"
            mock_genai.Client.return_value.models.generate_content.return_value = (
                mock_response
            )

            provider.generate(
                ProviderRequest(
                    prompt="test",
                    response_schema={
                        "type": "object",
                        "additionalProperties": False,
                    },
                )
            )

    call_kwargs = (
        mock_genai.Client.return_value.models.generate_content.call_args.kwargs
    )
    assert call_kwargs["config"].response_schema == {
        "type": "object",
        "additionalProperties": False,
    }
    assert "weakened" not in caplog.text


def test_gemini_provider_generate_without_system_prompt() -> None:
    """Verifies that gemini provider generate without system prompt."""
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}):
        with patch("graphwiki_kb.providers.gemini_provider.genai") as mock_genai:
            from graphwiki_kb.providers.gemini_provider import GeminiProvider

            provider = GeminiProvider()
            mock_response = MagicMock()
            mock_response.text = "response"
            mock_genai.Client.return_value.models.generate_content.return_value = (
                mock_response
            )

            result = provider.generate(ProviderRequest(prompt="test"))

    assert result.text == "response"


def test_gemini_provider_missing_key_raises() -> None:
    """Verifies that gemini provider missing key raises."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            from graphwiki_kb.providers.gemini_provider import GeminiProvider

            GeminiProvider()


# ---------------------------------------------------------------------------
# QueryService with provider
# ---------------------------------------------------------------------------


def test_query_service_uses_provider_when_available(test_project) -> None:
    """Verifies that query service uses provider when available.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file(
        "wiki/sources/alpha.md",
        "---\ntitle: Alpha\n---\n# Alpha\n\nKnowledge base traceability.\n",
    )
    provider = FakeProvider(
        text=json.dumps(
            {
                "answer_markdown": "The wiki tracks provenance via raw hashes.",
                "claims": [
                    {
                        "text": "The wiki tracks provenance.",
                        "citation_refs": ["wiki/sources/alpha.md#chunk-0"],
                    }
                ],
                "citations": [
                    {"ref": "wiki/sources/alpha.md#chunk-0", "title": "Alpha"}
                ],
                "insufficient_evidence": False,
            }
        )
    )
    from graphwiki_kb.services.search_service import SearchService

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
    """Verifies that query service raises without provider.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file(
        "wiki/sources/alpha.md",
        "---\ntitle: Alpha\n---\n# Alpha\n\nKnowledge base traceability.\n",
    )
    from graphwiki_kb.services.search_service import SearchService

    query_service = QueryService(
        test_project.paths,
        SearchService(test_project.paths),
        provider=None,
    )

    with pytest.raises(ProviderConfigurationError, match="kb legacy ask requires"):
        query_service.answer_question("knowledge base")


def test_query_service_raises_on_provider_error(test_project) -> None:
    """Verifies that query service raises on provider error.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file(
        "wiki/sources/alpha.md",
        "---\ntitle: Alpha\n---\n# Alpha\n\nKnowledge base traceability.\n",
    )
    provider = FailingProvider()
    from graphwiki_kb.services.search_service import SearchService

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
    """Verifies that query service no matches returns fallback regardless of provider.

    Args:
        test_project: Test project value used by the operation.
    """
    provider = FakeProvider()
    from graphwiki_kb.services.search_service import SearchService

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
    """Verifies that review service uses provider when available.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file(
        "wiki/sources/alpha.md",
        "Knowledge base markdown traceability.",
    )
    provider = FakeProvider(
        text=json.dumps(
            {
                "issues": [
                    {
                        "severity": "warning",
                        "code": "stale-claim",
                        "pages": ["wiki/sources/alpha.md"],
                        "message": "Claim may be outdated.",
                    }
                ]
            }
        )
    )
    review_service = ReviewService(test_project.paths, provider=provider)

    report = review_service.review()

    assert report.mode == "provider:fake-v1"
    provider_issues = [i for i in report.issues if i.code == "stale-claim"]
    assert len(provider_issues) == 1
    assert provider_issues[0].severity == "warning"


def test_review_service_requires_provider(test_project) -> None:
    """Verifies that review service requires provider.

    Args:
        test_project: Test project value used by the operation.
    """
    from graphwiki_kb.providers import ProviderConfigurationError

    test_project.write_file("wiki/sources/alpha.md", "Some content here.")
    review_service = ReviewService(test_project.paths, provider=None)

    with pytest.raises(
        ProviderConfigurationError, match="requires a configured provider"
    ):
        review_service.review()


def test_review_service_falls_back_on_provider_error(test_project) -> None:
    """Verifies that review service falls back on provider error.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file("wiki/sources/alpha.md", "Some content here.")
    provider = FailingProvider()
    review_service = ReviewService(test_project.paths, provider=provider)

    with pytest.raises(ProviderExecutionError, match="API is down"):
        review_service.review()


def test_review_service_provider_no_issues_response(test_project) -> None:
    """Verifies that review service provider no issues response.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file("wiki/sources/alpha.md", "Some content here.")
    provider = FakeProvider(text=json.dumps({"issues": []}))
    review_service = ReviewService(test_project.paths, provider=provider)

    report = review_service.review()

    provider_issues = [
        i
        for i in report.issues
        if i.code not in ("overlapping-topics", "terminology-variant")
    ]
    assert len(provider_issues) == 0


def test_review_service_provider_multiple_issues(test_project) -> None:
    """Verifies that review service provider multiple issues.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file("wiki/sources/a.md", "Page A content.")
    test_project.write_file("wiki/sources/b.md", "Page B content.")
    provider = FakeProvider(
        text=json.dumps(
            {
                "issues": [
                    {
                        "severity": "error",
                        "code": "contradiction",
                        "pages": ["wiki/sources/a.md", "wiki/sources/b.md"],
                        "message": "Conflicting claims about X.",
                    },
                    {
                        "severity": "suggestion",
                        "code": "redundant-content",
                        "pages": ["wiki/sources/b.md"],
                        "message": "Repeats content from A.",
                    },
                ]
            }
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


def test_review_service_provider_rejects_malformed_json(test_project) -> None:
    """Verifies that review service provider rejects malformed json.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file("wiki/sources/a.md", "Content.")
    provider = FakeProvider(text="Some random text")
    review_service = ReviewService(test_project.paths, provider=provider)

    with pytest.raises(ProviderExecutionError, match="structured JSON schema"):
        review_service.review()


def test_review_service_provider_rejects_unknown_severity(test_project) -> None:
    """Verifies that review service provider rejects unknown severity.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file("wiki/sources/a.md", "Content.")
    provider = FakeProvider(
        text=json.dumps(
            {
                "issues": [
                    {
                        "severity": "critical",
                        "code": "bad-severity",
                        "pages": ["wiki/sources/a.md"],
                        "message": "Unknown severity level.",
                    }
                ]
            }
        )
    )
    review_service = ReviewService(test_project.paths, provider=provider)

    with pytest.raises(ProviderExecutionError, match="structured JSON schema"):
        review_service.review()


# ---------------------------------------------------------------------------
# _parse_provider_issues unit tests
# ---------------------------------------------------------------------------


def test_parse_provider_issues_valid_json() -> None:
    """Verifies that parse provider issues valid json."""
    raw = json.dumps(
        {
            "issues": [
                {
                    "severity": "error",
                    "code": "contradiction",
                    "pages": ["a.md", "b.md"],
                    "message": "Pages disagree.",
                }
            ]
        }
    )
    issues = ReviewService._parse_provider_issues(raw)
    assert len(issues) == 1
    assert issues[0].code == "contradiction"
    assert issues[0].pages == ["a.md", "b.md"]


def test_parse_provider_issues_no_issues() -> None:
    """Verifies that parse provider issues no issues."""
    assert ReviewService._parse_provider_issues("NO_ISSUES") == []


def test_parse_provider_issues_empty_string() -> None:
    """Verifies that parse provider issues empty string."""
    with pytest.raises(ValueError, match="structured JSON schema"):
        ReviewService._parse_provider_issues("")


def test_parse_provider_issues_reads_fenced_json() -> None:
    """Verifies that parse provider issues reads fenced json."""
    raw = '```json\n{"issues": []}\n```'

    assert ReviewService._parse_provider_issues(raw) == []


def test_parse_provider_issues_rejects_non_json() -> None:
    """Verifies that parse provider issues rejects non json."""
    raw = "Some preamble\nISSUE|warning|x|a.md|msg\nMore text"
    with pytest.raises(ValueError, match="structured JSON schema"):
        ReviewService._parse_provider_issues(raw)


# ---------------------------------------------------------------------------
# Provider base class updated fields
# ---------------------------------------------------------------------------


def test_provider_request_max_tokens_default() -> None:
    """Verifies that provider request max tokens default."""
    req = ProviderRequest(prompt="hi")
    assert req.max_tokens == 1024


def test_provider_request_custom_max_tokens() -> None:
    """Verifies that provider request custom max tokens."""
    req = ProviderRequest(prompt="hi", max_tokens=512)
    assert req.max_tokens == 512


def test_provider_request_reasoning_effort_override() -> None:
    """Verifies that provider request reasoning effort override."""
    req = ProviderRequest(prompt="hi", reasoning_effort="low")
    assert req.reasoning_effort == "low"
