"""Provider integration helpers for structured.

This module belongs to `graphwiki_kb.providers.structured` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError


class StructuredOutputError(ValueError):
    """Raised when a provider response cannot be parsed as structured output."""


_FENCED_JSON_PATTERN = re.compile(
    r"```(?:json)?\s*(?P<body>.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_JSON_DECODER = json.JSONDecoder()
_ModelT = TypeVar("_ModelT", bound=BaseModel)


def parse_json_payload(raw: str, *, label: str = "Provider response") -> Any:
    """Parse provider JSON even when a model wraps it in common prose/fences."""
    for payload in iter_json_payloads(raw, label=label):
        return payload

    raise StructuredOutputError(f"{label} did not contain valid JSON.")


def iter_json_payloads(
    raw: str,
    *,
    label: str = "Provider response",
) -> Iterator[Any]:
    """Yield JSON candidates from a provider response in preferred order."""
    text = raw.strip()
    if not text:
        raise StructuredOutputError(f"{label} was empty.")
    yield from _json_payload_candidates(text)


def parse_model_payload(
    raw: str,
    model_type: type[_ModelT],
    *,
    label: str = "Provider response",
) -> _ModelT:
    """Parse and validate a provider JSON payload against a Pydantic model."""
    found_json = False
    last_error: Exception | None = None
    for payload in iter_json_payloads(raw, label=label):
        found_json = True
        try:
            return model_type.model_validate(payload)
        except (TypeError, ValidationError) as exc:
            last_error = exc
            continue

    if found_json:
        raise StructuredOutputError(
            f"{label} did not match the structured JSON schema."
        ) from last_error
    raise StructuredOutputError(f"{label} did not contain valid JSON.")


def _json_payload_candidates(text: str) -> Iterator[Any]:
    try:
        yield json.loads(text)
    except json.JSONDecodeError:
        pass

    for match in _FENCED_JSON_PATTERN.finditer(text):
        fenced = match.group("body").strip()
        if not fenced:
            continue
        try:
            yield json.loads(fenced)
        except json.JSONDecodeError:
            continue

    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            payload, _ = _JSON_DECODER.raw_decode(text[index:])
            yield payload
        except json.JSONDecodeError:
            continue
