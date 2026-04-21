"""Phase 3 tests — Optional LangChain Provider Adapter.

Validates that the LangChainProvider adapter correctly wraps LangChain chat
models behind the TextProvider interface, that build_provider routes to it
when ``ecosystem.providers.backend`` is ``"langchain"``, and that fallback
to UnavailableProvider works when packages or API keys are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.providers import (
    UnavailableProvider,
    build_provider,
)
from src.providers.base import ProviderRequest, ProviderResponse, TextProvider
from src.providers.langchain_provider import LangChainProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_config(
    provider_name: str = "openai",
    backend: str = "langchain",
    model: str | None = None,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "provider": {"name": provider_name},
        "ecosystem": {"providers": {"backend": backend}},
    }
    if model:
        cfg["provider"]["model"] = model
    return cfg


class _FakeAIMessage:
    """Minimal stand-in for langchain_core.messages.AIMessage."""

    def __init__(self, content: Any) -> None:
        self.content = content


# ---------------------------------------------------------------------------
# LangChainProvider._extract_text unit tests
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_string_content(self) -> None:
        msg = _FakeAIMessage("Hello world")
        assert LangChainProvider._extract_text(msg) == "Hello world"

    def test_list_of_text_blocks(self) -> None:
        msg = _FakeAIMessage(
            [
                {"type": "thinking", "thinking": "reasoning..."},
                {"type": "text", "text": "Final answer."},
            ]
        )
        assert LangChainProvider._extract_text(msg) == "Final answer."

    def test_multiple_text_blocks_joined(self) -> None:
        msg = _FakeAIMessage(
            [
                {"type": "text", "text": "Part one."},
                {"type": "text", "text": "Part two."},
            ]
        )
        assert LangChainProvider._extract_text(msg) == "Part one.\nPart two."

    def test_list_of_strings(self) -> None:
        msg = _FakeAIMessage(["chunk1", "chunk2"])
        assert LangChainProvider._extract_text(msg) == "chunk1\nchunk2"

    def test_empty_list_returns_empty(self) -> None:
        msg = _FakeAIMessage([])
        assert LangChainProvider._extract_text(msg) == ""

    def test_non_standard_content_falls_back_to_str(self) -> None:
        msg = _FakeAIMessage(42)
        assert LangChainProvider._extract_text(msg) == "42"

    def test_thinking_blocks_excluded(self) -> None:
        msg = _FakeAIMessage(
            [
                {"type": "reasoning", "reasoning": "step 1"},
                {"type": "text", "text": "Answer."},
            ]
        )
        assert LangChainProvider._extract_text(msg) == "Answer."


# ---------------------------------------------------------------------------
# LangChainProvider constructor and generate()
# ---------------------------------------------------------------------------


class TestLangChainProviderOpenAI:
    """Test the OpenAI path through LangChainProvider."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("src.providers.langchain_provider.LangChainProvider._build_chat_model")
    def test_generate_openai_system_and_human(self, mock_build: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.invoke.return_value = _FakeAIMessage("LangChain answer")
        mock_build.return_value = mock_model

        provider = LangChainProvider(
            provider_name="openai",
            model="gpt-5.4-mini",
            api_key_env="OPENAI_API_KEY",
        )
        request = ProviderRequest(
            prompt="What is Python?",
            system_prompt="You are helpful.",
            max_tokens=512,
        )
        response = provider.generate(request)

        assert response.text == "LangChain answer"
        assert response.model_name == "gpt-5.4-mini"
        call_args = mock_model.invoke.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0].__class__.__name__ == "SystemMessage"
        assert messages[1].__class__.__name__ == "HumanMessage"
        assert call_args[1]["max_tokens"] == 512

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("src.providers.langchain_provider.LangChainProvider._build_chat_model")
    def test_generate_openai_no_system_prompt(self, mock_build: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.invoke.return_value = _FakeAIMessage("Answer only")
        mock_build.return_value = mock_model

        provider = LangChainProvider(
            provider_name="openai",
            model="gpt-5.4-mini",
            api_key_env="OPENAI_API_KEY",
        )
        request = ProviderRequest(prompt="Hello")
        response = provider.generate(request)

        assert response.text == "Answer only"
        messages = mock_model.invoke.call_args[0][0]
        assert len(messages) == 1
        assert messages[0].__class__.__name__ == "HumanMessage"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("src.providers.langchain_provider.LangChainProvider._build_chat_model")
    def test_name_matches_provider(self, mock_build: MagicMock) -> None:
        mock_build.return_value = MagicMock()
        provider = LangChainProvider(
            provider_name="openai",
            model="gpt-5.4-mini",
            api_key_env="OPENAI_API_KEY",
        )
        assert provider.name == "openai"


class TestLangChainProviderAnthropic:
    """Test the Anthropic path through LangChainProvider."""

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"})
    @patch("src.providers.langchain_provider.LangChainProvider._build_chat_model")
    def test_anthropic_thinking_budget_adjusts_max_tokens(
        self, mock_build: MagicMock
    ) -> None:
        mock_model = MagicMock()
        mock_model.invoke.return_value = _FakeAIMessage("Thought answer")
        mock_build.return_value = mock_model

        provider = LangChainProvider(
            provider_name="anthropic",
            model="claude-sonnet-4-6",
            api_key_env="ANTHROPIC_API_KEY",
            thinking_budget=10_000,
        )
        request = ProviderRequest(prompt="Explain.", max_tokens=512)
        provider.generate(request)

        call_kwargs = mock_model.invoke.call_args[1]
        assert call_kwargs["max_tokens"] == 10_000 + 4096

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"})
    @patch("src.providers.langchain_provider.LangChainProvider._build_chat_model")
    def test_anthropic_no_thinking_uses_request_max_tokens(
        self, mock_build: MagicMock
    ) -> None:
        mock_model = MagicMock()
        mock_model.invoke.return_value = _FakeAIMessage("Simple answer")
        mock_build.return_value = mock_model

        provider = LangChainProvider(
            provider_name="anthropic",
            model="claude-sonnet-4-6",
            api_key_env="ANTHROPIC_API_KEY",
            thinking_budget=0,
        )
        request = ProviderRequest(prompt="Hi", max_tokens=256)
        provider.generate(request)

        call_kwargs = mock_model.invoke.call_args[1]
        assert call_kwargs["max_tokens"] == 256


class TestLangChainProviderGemini:
    """Test the Gemini path through LangChainProvider."""

    @patch.dict("os.environ", {"GEMINI_API_KEY": "gem-test"})
    @patch("src.providers.langchain_provider.LangChainProvider._build_chat_model")
    def test_gemini_name_and_generate(self, mock_build: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.invoke.return_value = _FakeAIMessage("Gemini answer")
        mock_build.return_value = mock_model

        provider = LangChainProvider(
            provider_name="gemini",
            model="gemini-3.1-flash-lite-preview",
            api_key_env="GEMINI_API_KEY",
            reasoning_effort="medium",
        )
        assert provider.name == "gemini"
        response = provider.generate(ProviderRequest(prompt="Test"))
        assert response.text == "Gemini answer"
        assert response.model_name == "gemini-3.1-flash-lite-preview"


# ---------------------------------------------------------------------------
# Constructor error paths
# ---------------------------------------------------------------------------


class TestLangChainProviderErrors:
    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            LangChainProvider(
                provider_name="openai",
                model="gpt-5.4-mini",
                api_key_env="OPENAI_API_KEY",
            )

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("src.providers.langchain_provider.LangChainProvider._build_chat_model")
    def test_unsupported_provider_in_build_chat_model(
        self, mock_build: MagicMock
    ) -> None:
        """Simulate _build_chat_model raising for an unknown provider."""
        mock_build.side_effect = ValueError("Unsupported provider")
        with pytest.raises(ValueError, match="Unsupported"):
            LangChainProvider(
                provider_name="unknown",
                model="m",
                api_key_env="OPENAI_API_KEY",
            )


# ---------------------------------------------------------------------------
# _build_chat_model dispatch (real import, mocked constructors)
# ---------------------------------------------------------------------------


class TestBuildChatModel:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("langchain_openai.ChatOpenAI")
    def test_openai_dispatch(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value = MagicMock()
        provider = LangChainProvider.__new__(LangChainProvider)
        provider._reasoning_effort = "medium"
        provider._thinking_budget = 0
        result = provider._build_chat_model("openai", "gpt-5.4-mini", "sk-test")
        mock_cls.assert_called_once_with(
            model="gpt-5.4-mini",
            api_key="sk-test",
            reasoning_effort="medium",
        )

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant"})
    @patch("langchain_anthropic.ChatAnthropic")
    def test_anthropic_dispatch_with_thinking(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value = MagicMock()
        provider = LangChainProvider.__new__(LangChainProvider)
        provider._reasoning_effort = "high"
        provider._thinking_budget = 8000
        result = provider._build_chat_model("anthropic", "claude-sonnet-4-6", "sk-ant")
        mock_cls.assert_called_once_with(
            model="claude-sonnet-4-6",
            api_key="sk-ant",
            thinking={"type": "enabled", "budget_tokens": 8000},
        )

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant"})
    @patch("langchain_anthropic.ChatAnthropic")
    def test_anthropic_dispatch_without_thinking(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value = MagicMock()
        provider = LangChainProvider.__new__(LangChainProvider)
        provider._reasoning_effort = "high"
        provider._thinking_budget = 0
        result = provider._build_chat_model("anthropic", "claude-sonnet-4-6", "sk-ant")
        mock_cls.assert_called_once_with(
            model="claude-sonnet-4-6",
            api_key="sk-ant",
        )

    @patch.dict("os.environ", {"GEMINI_API_KEY": "gem"})
    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_gemini_dispatch(self, mock_cls: MagicMock) -> None:
        mock_cls.return_value = MagicMock()
        provider = LangChainProvider.__new__(LangChainProvider)
        provider._reasoning_effort = "low"
        provider._thinking_budget = 0
        result = provider._build_chat_model(
            "gemini", "gemini-3.1-flash-lite-preview", "gem"
        )
        mock_cls.assert_called_once_with(
            model="gemini-3.1-flash-lite-preview",
            google_api_key="gem",
            thinking_level="low",
        )

    def test_unknown_provider_raises(self) -> None:
        provider = LangChainProvider.__new__(LangChainProvider)
        provider._reasoning_effort = "high"
        provider._thinking_budget = 0
        with pytest.raises(ValueError, match="Unsupported provider"):
            provider._build_chat_model("cohere", "cmd-r", "key")


# ---------------------------------------------------------------------------
# build_provider routing — langchain vs direct
# ---------------------------------------------------------------------------


class TestBuildProviderRouting:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("src.providers.langchain_provider.LangChainProvider._build_chat_model")
    def test_langchain_backend_returns_langchain_provider(
        self, mock_build: MagicMock
    ) -> None:
        mock_build.return_value = MagicMock()
        cfg = _base_config(provider_name="openai", backend="langchain")
        provider = build_provider(cfg)
        assert isinstance(provider, LangChainProvider)
        assert provider.name == "openai"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    def test_direct_backend_returns_direct_provider(self) -> None:
        cfg = _base_config(provider_name="openai", backend="direct")
        provider = build_provider(cfg)
        assert provider is not None
        assert type(provider).__name__ == "OpenAIProvider"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    def test_default_backend_returns_direct_provider(self) -> None:
        cfg: dict[str, Any] = {"provider": {"name": "openai"}}
        provider = build_provider(cfg)
        assert provider is not None
        assert type(provider).__name__ == "OpenAIProvider"

    @patch.dict("os.environ", {}, clear=True)
    def test_langchain_missing_key_returns_unavailable(self) -> None:
        cfg = _base_config(provider_name="openai", backend="langchain")
        provider = build_provider(cfg)
        assert isinstance(provider, UnavailableProvider)

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    def test_langchain_import_error_returns_unavailable(self) -> None:
        cfg = _base_config(provider_name="openai", backend="langchain")
        with patch(
            "src.providers.langchain_provider.LangChainProvider._build_chat_model",
            side_effect=ImportError("no module"),
        ):
            provider = build_provider(cfg)
        assert isinstance(provider, UnavailableProvider)

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant"})
    @patch("src.providers.langchain_provider.LangChainProvider._build_chat_model")
    def test_langchain_anthropic_backend(self, mock_build: MagicMock) -> None:
        mock_build.return_value = MagicMock()
        cfg = _base_config(provider_name="anthropic", backend="langchain")
        provider = build_provider(cfg)
        assert isinstance(provider, LangChainProvider)
        assert provider.name == "anthropic"

    @patch.dict("os.environ", {"GEMINI_API_KEY": "gem"})
    @patch("src.providers.langchain_provider.LangChainProvider._build_chat_model")
    def test_langchain_gemini_backend(self, mock_build: MagicMock) -> None:
        mock_build.return_value = MagicMock()
        cfg = _base_config(provider_name="gemini", backend="langchain")
        provider = build_provider(cfg)
        assert isinstance(provider, LangChainProvider)
        assert provider.name == "gemini"

    def test_no_provider_name_returns_none(self) -> None:
        cfg = _base_config(backend="langchain")
        cfg["provider"]["name"] = ""
        assert build_provider(cfg) is None


# ---------------------------------------------------------------------------
# LangChainProvider is a proper TextProvider subclass
# ---------------------------------------------------------------------------


def test_langchain_provider_is_text_provider_subclass() -> None:
    assert issubclass(LangChainProvider, TextProvider)


# ---------------------------------------------------------------------------
# Response stripping
# ---------------------------------------------------------------------------


class TestResponseStripping:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("src.providers.langchain_provider.LangChainProvider._build_chat_model")
    def test_whitespace_stripped(self, mock_build: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.invoke.return_value = _FakeAIMessage("  padded  \n")
        mock_build.return_value = mock_model

        provider = LangChainProvider(
            provider_name="openai",
            model="gpt-5.4-mini",
            api_key_env="OPENAI_API_KEY",
        )
        response = provider.generate(ProviderRequest(prompt="test"))
        assert response.text == "padded"


# ---------------------------------------------------------------------------
# build_provider with resolved config (model registry path)
# ---------------------------------------------------------------------------


class TestBuildProviderWithResolved:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"})
    @patch("src.providers.langchain_provider.LangChainProvider._build_chat_model")
    def test_resolved_overrides_config_defaults(self, mock_build: MagicMock) -> None:
        from src.models.provider_models import ResolvedProviderConfig

        mock_build.return_value = MagicMock()
        resolved = ResolvedProviderConfig(
            provider_name="openai",
            model="gpt-5.4",
            tier="high",
            api_key_env="OPENAI_API_KEY",
            reasoning_effort="low",
            thinking_budget=0,
        )
        cfg = _base_config(provider_name="openai", backend="langchain")
        provider = build_provider(cfg, resolved=resolved)
        assert isinstance(provider, LangChainProvider)
        assert provider.model == "gpt-5.4"
