"""Tests for test provider structured.

This module belongs to `tests.test_provider_structured` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from src.providers.structured import (
    StructuredOutputError,
    parse_json_payload,
    parse_model_payload,
)


class _StructuredPayload(BaseModel):
    """Represents structured payload behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    ok: bool


def test_parse_json_payload_skips_empty_and_invalid_fences() -> None:
    """Verifies that parse json payload skips empty and invalid fences."""
    raw = """
```json

```
```json
{broken
```
Here is the JSON:
{"ok": true}
"""

    assert parse_json_payload(raw) == {"ok": True}


def test_parse_model_payload_raises_for_schema_mismatch() -> None:
    """Verifies that parse model payload raises for schema mismatch."""
    with pytest.raises(StructuredOutputError, match="structured JSON schema"):
        parse_model_payload('{"missing": true}', _StructuredPayload)
