"""Token-aware tokenizer abstraction for the LightRAG chunker.

LightRAG segments documents into ~1200-token chunks. To stay deterministic and
import-safe on a base install, this module prefers :mod:`tiktoken`
(``cl100k_base``) when the ``wikigraph`` extra is installed, and otherwise falls
back to a pure-python regex word tokenizer. Both implementations expose the same
:class:`Tokenizer` protocol so the chunker is agnostic to the backend.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

_WORD_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@runtime_checkable
class Tokenizer(Protocol):
    """Minimal tokenizer interface used by the LightRAG chunker."""

    name: str

    def encode(self, text: str) -> list[int]:
        """Return token ids for ``text``."""

    def decode(self, tokens: list[int]) -> str:
        """Return text for a list of token ids."""

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""


class TiktokenTokenizer:
    """A :mod:`tiktoken`-backed tokenizer (default when available)."""

    name = "tiktoken:cl100k_base"

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken

        self._encoding = tiktoken.get_encoding(encoding_name)
        self.name = f"tiktoken:{encoding_name}"

    def encode(self, text: str) -> list[int]:
        """Return tiktoken ids for ``text``."""
        return list(self._encoding.encode(text))

    def decode(self, tokens: list[int]) -> str:
        """Return text decoded from tiktoken ids."""
        return self._encoding.decode(tokens)

    def count(self, text: str) -> int:
        """Return the tiktoken token count for ``text``."""
        return len(self._encoding.encode(text))


class RegexWordTokenizer:
    """A deterministic pure-python fallback tokenizer.

    Tokens are word/punctuation runs joined by single spaces on decode. This is
    intentionally simple: it keeps chunking deterministic and dependency-free,
    at the cost of decoded text not being byte-identical to the input. The
    chunker therefore uses character offsets (not decoded tokens) to slice the
    original text, so the fallback only affects *where* boundaries land.
    """

    name = "regex-word"

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._inverse: dict[int, str] = {}

    def _intern(self, token: str) -> int:
        token_id = self._vocab.get(token)
        if token_id is None:
            token_id = len(self._vocab)
            self._vocab[token] = token_id
            self._inverse[token_id] = token
        return token_id

    def encode(self, text: str) -> list[int]:
        """Return interned ids for each word/punctuation token in ``text``."""
        return [self._intern(token) for token in _WORD_PATTERN.findall(text)]

    def decode(self, tokens: list[int]) -> str:
        """Return a space-joined approximation of the original text."""
        return " ".join(self._inverse.get(token_id, "") for token_id in tokens)

    def count(self, text: str) -> int:
        """Return the number of word/punctuation tokens in ``text``."""
        return len(_WORD_PATTERN.findall(text))


def get_default_tokenizer() -> Tokenizer:
    """Return the best available tokenizer (tiktoken, else regex fallback)."""
    try:
        return TiktokenTokenizer()
    except Exception:  # pragma: no cover - exercised via monkeypatch in tests
        return RegexWordTokenizer()
