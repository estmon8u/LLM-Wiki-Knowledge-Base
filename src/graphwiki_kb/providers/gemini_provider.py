"""Provider integration helpers for gemini provider.

This module belongs to `graphwiki_kb.providers.gemini_provider` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from graphwiki_kb.providers.base import (
    ProviderCapabilities,
    ProviderRequest,
    ProviderResponse,
    TextProvider,
)
from graphwiki_kb.providers.retry import provider_retry

logger = logging.getLogger(__name__)
_SUPPORTED_SCHEMA_TYPES = {
    "string",
    "number",
    "integer",
    "boolean",
    "object",
    "array",
    "null",
}
_SUPPORTED_SCHEMA_KEYS = {
    "type",
    "title",
    "description",
    "properties",
    "required",
    "additionalProperties",
    "enum",
    "format",
    "minimum",
    "maximum",
    "items",
    "prefixItems",
    "minItems",
    "maxItems",
    "propertyOrdering",
}
_DEFINITION_KEYS = ("$defs", "definitions")


class GeminiProvider(TextProvider):
    """Google Gemini provider via the google-genai SDK."""

    name = "gemini"
    capabilities = ProviderCapabilities(
        strict_json_schema=False,
        native_structured_output=True,
        reasoning_effort_values=("none", "minimal", "low", "medium", "high"),
    )

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
        reasoning_effort: str = "high",
    ) -> None:
        self.model = model
        self._reasoning_effort = reasoning_effort
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise ValueError(
                f"Environment variable {api_key_env} is not set. "
                "Set it to your Gemini API key to use the Gemini provider."
            )
        self._client = genai.Client(api_key=api_key)

    _EFFORT_TO_LEVEL = {
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
    }
    _EFFORT_TO_25_BUDGET = {
        "none": 0,
        "minimal": 1024,
        "low": 1024,
        "medium": 8192,
        "high": 24576,
    }

    @provider_retry()
    def generate(self, request: ProviderRequest) -> ProviderResponse:
        """Generate.

        Args:
            request: Request value used by the operation.

        Returns:
            ProviderResponse produced by the operation.
        """
        effort = request.reasoning_effort or self._reasoning_effort
        config_kwargs: dict = {
            "max_output_tokens": request.max_tokens,
            "thinking_config": self._thinking_config(effort),
        }
        if request.system_prompt:
            config_kwargs["system_instruction"] = request.system_prompt
        if request.response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            schema, report = _gemini_response_schema_with_report(
                request.response_schema
            )
            if report.weakened:
                logger.warning(
                    "Gemini response schema was weakened for provider compatibility: "
                    "%s",
                    ", ".join(report.removed_keywords),
                )
            config_kwargs["response_schema"] = schema
        response = self._client.models.generate_content(
            model=self.model,
            contents=request.prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        text = response.text or ""
        candidates = getattr(response, "candidates", None) or []
        candidate = candidates[0] if candidates else None
        usage = getattr(response, "usage_metadata", None)
        return ProviderResponse(
            text=text.strip(),
            model_name=self.model,
            provider=self.name,
            finish_reason=(
                str(getattr(candidate, "finish_reason", ""))
                if candidate is not None
                else None
            ),
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            raw=response,
        )

    def _thinking_config(self, effort: str) -> types.ThinkingConfig:
        if _uses_thinking_budget(self.model):
            budget = self._EFFORT_TO_25_BUDGET.get(effort, 24576)
            return types.ThinkingConfig(thinking_budget=budget)
        level = self._EFFORT_TO_LEVEL.get(effort, "high")
        return types.ThinkingConfig(thinking_level=level)


@dataclass(frozen=True, slots=True)
class GeminiSchemaTransformationReport:
    """Describes provider schema compatibility transformations."""

    removed_keywords: tuple[str, ...] = ()
    weakened: bool = False


def _gemini_response_schema(schema: object) -> object:
    """Return the JSON schema payload sent to Gemini."""
    converted, _report = _gemini_response_schema_with_report(schema)
    return converted


def _gemini_response_schema_with_report(
    schema: object,
) -> tuple[object, GeminiSchemaTransformationReport]:
    """Return schema plus any compatibility downgrades applied."""
    removed: list[str] = []
    root = deepcopy(schema)
    converted = _normalize_gemini_schema(root, root=root, removed=removed, path="$")
    removed_keywords = tuple(dict.fromkeys(removed))
    return converted, GeminiSchemaTransformationReport(
        removed_keywords=removed_keywords,
        weakened=bool(removed_keywords),
    )


def _normalize_gemini_schema(
    schema: object,
    *,
    root: object,
    removed: list[str],
    path: str,
) -> object:
    if isinstance(schema, bool):
        return schema
    if not isinstance(schema, dict):
        return deepcopy(schema)

    resolved = _resolve_schema_ref(schema, root=root, removed=removed, path=path)
    if resolved is not schema:
        return _normalize_gemini_schema(resolved, root=root, removed=removed, path=path)

    nullable = _collapse_nullable_composition(schema)
    if nullable is not None:
        schema = nullable

    normalized: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _DEFINITION_KEYS or key == "$schema":
            removed.append(f"{path}.{key}")
            continue
        if key not in _SUPPORTED_SCHEMA_KEYS:
            removed.append(f"{path}.{key}")
            continue
        if key == "type":
            normalized_type = _normalize_schema_type(value)
            if normalized_type is None:
                removed.append(f"{path}.type")
                continue
            normalized[key] = normalized_type
        elif key == "properties" and isinstance(value, dict):
            normalized[key] = {
                str(name): _normalize_gemini_schema(
                    prop,
                    root=root,
                    removed=removed,
                    path=f"{path}.properties.{name}",
                )
                for name, prop in value.items()
                if isinstance(prop, (dict, bool))
            }
        elif key == "required" and isinstance(value, list):
            normalized[key] = [str(item) for item in value if isinstance(item, str)]
        elif key == "items" and isinstance(value, (dict, bool)):
            normalized[key] = _normalize_gemini_schema(
                value,
                root=root,
                removed=removed,
                path=f"{path}.items",
            )
        elif key == "prefixItems" and isinstance(value, list):
            normalized[key] = [
                _normalize_gemini_schema(
                    item,
                    root=root,
                    removed=removed,
                    path=f"{path}.prefixItems.{index}",
                )
                for index, item in enumerate(value)
                if isinstance(item, (dict, bool))
            ]
        elif key == "additionalProperties" and isinstance(value, (dict, bool)):
            normalized[key] = (
                _normalize_gemini_schema(
                    value,
                    root=root,
                    removed=removed,
                    path=f"{path}.additionalProperties",
                )
                if isinstance(value, dict)
                else value
            )
        else:
            normalized[key] = deepcopy(value)
    return normalized


def _resolve_schema_ref(
    schema: dict[str, Any],
    *,
    root: object,
    removed: list[str],
    path: str,
) -> object:
    ref = schema.get("$ref")
    if not isinstance(ref, str):
        return schema
    resolved = _local_schema_ref(root, ref)
    if resolved is None:
        removed.append(f"{path}.$ref")
        return {key: value for key, value in schema.items() if key != "$ref"}
    merged = deepcopy(resolved)
    if isinstance(merged, dict):
        for key, value in schema.items():
            if key != "$ref":
                merged[key] = value
    return merged


def _local_schema_ref(root: object, ref: str) -> object | None:
    if not ref.startswith("#/"):
        return None
    current = root
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _collapse_nullable_composition(schema: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("anyOf", "oneOf"):
        value = schema.get(key)
        if not isinstance(value, list) or len(value) != 2:
            continue
        typed = [item for item in value if isinstance(item, dict)]
        if len(typed) != 2:
            continue
        null_item = next((item for item in typed if item.get("type") == "null"), None)
        other = next((item for item in typed if item.get("type") != "null"), None)
        if null_item is None or other is None:
            continue
        collapsed = {k: v for k, v in schema.items() if k != key}
        collapsed.update(other)
        other_type = other.get("type")
        if isinstance(other_type, list):
            types = [*other_type, "null"]
        elif isinstance(other_type, str):
            types = [other_type, "null"]
        else:
            types = ["null"]
        collapsed["type"] = list(dict.fromkeys(types))
        return collapsed
    return None


def _normalize_schema_type(value: object) -> str | list[str] | None:
    if isinstance(value, str):
        return value if value in _SUPPORTED_SCHEMA_TYPES else None
    if isinstance(value, list):
        types = [
            item
            for item in value
            if isinstance(item, str) and item in _SUPPORTED_SCHEMA_TYPES
        ]
        if not types:
            return None
        return list(dict.fromkeys(types))
    return None


def _uses_thinking_budget(model: str) -> bool:
    return model.casefold().startswith("gemini-2.5")
