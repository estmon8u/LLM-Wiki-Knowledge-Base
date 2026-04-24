from __future__ import annotations

import json
import re
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
    text = raw.strip()
    if not text:
        raise StructuredOutputError(f"{label} was empty.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for match in _FENCED_JSON_PATTERN.finditer(text):
        fenced = match.group("body").strip()
        if not fenced:
            continue
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            continue

    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            payload, _ = _JSON_DECODER.raw_decode(text[index:])
            return payload
        except json.JSONDecodeError:
            continue

    raise StructuredOutputError(f"{label} did not contain valid JSON.")


def parse_model_payload(
    raw: str,
    model_type: type[_ModelT],
    *,
    label: str = "Provider response",
) -> _ModelT:
    """Parse and validate a provider JSON payload against a Pydantic model."""
    payload = parse_json_payload(raw, label=label)
    try:
        return model_type.model_validate(payload)
    except (TypeError, ValidationError) as exc:
        raise StructuredOutputError(
            f"{label} did not match the structured JSON schema."
        ) from exc
