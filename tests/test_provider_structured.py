from __future__ import annotations

import pytest
from pydantic import BaseModel

from src.providers.structured import (
    StructuredOutputError,
    parse_json_payload,
    parse_model_payload,
)


class _StructuredPayload(BaseModel):
    ok: bool


def test_parse_json_payload_skips_empty_and_invalid_fences() -> None:
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
    with pytest.raises(StructuredOutputError, match="structured JSON schema"):
        parse_model_payload('{"missing": true}', _StructuredPayload)
