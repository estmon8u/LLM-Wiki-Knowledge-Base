from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
)
import yaml

from src.services.project_service import ProjectPaths, atomic_write_text


CURRENT_CONFIG_VERSION = 4


DEFAULT_CONFIG: dict[str, Any] = {
    "version": CURRENT_CONFIG_VERSION,
    "project": {
        "name": "Capstone Knowledge Base",
        "description": "Markdown-first research knowledge base maintained through a CLI workflow.",
    },
    "storage": {
        "raw_dir": "raw/sources",
        "raw_normalized_dir": "raw/normalized",
        "wiki_sources_dir": "wiki/sources",
        "wiki_concepts_dir": "wiki/concepts",
        "vault_dir": "vault/obsidian",
    },
    "compile": {
        "excerpt_character_limit": 900,
    },
    "lint": {
        "required_frontmatter_fields": [
            "title",
            "summary",
            "source_id",
            "raw_path",
            "source_hash",
            "compiled_at",
        ],
    },
    "provider": {},
    "providers": {
        "openai": {
            "model": "gpt-5.4-mini",
            "api_key_env": "OPENAI_API_KEY",
            "reasoning_effort": "high",
        },
        "anthropic": {
            "model": "claude-sonnet-4-6",
            "api_key_env": "ANTHROPIC_API_KEY",
            "thinking_budget": 10_000,
        },
        "gemini": {
            "model": "gemini-3.1-flash-lite-preview",
            "api_key_env": "GEMINI_API_KEY",
            "reasoning_effort": "high",
        },
    },
    "conversion": {
        "mistral_ocr": {
            "model": "mistral-ocr-latest",
            "api_key_env": "MISTRAL_API_KEY",
            "table_format": "markdown",
        },
        "html": {
            "renderer": "wkhtmltopdf",
            "wkhtmltopdf_path": None,
        },
        "fallbacks": {
            "pdf": "docling",
            "docx": "markitdown",
            "pptx": "markitdown",
            "html": "markitdown",
        },
    },
}


DEFAULT_SCHEMA = """# kb.schema.md

Operational rules for building and maintaining this knowledge base.

## Page Types

- source — one per ingested document, in wiki/sources/
- concept — synthesizes multiple source pages, in wiki/concepts/
- analysis — saved answer to a user question, in wiki/analysis/

## Source Pages

- Create one source page for every ingested document.
- Preserve source traceability: include source_id, raw_path, and content hash.
- Keep the summary concise (2-4 sentences) and grounded in the ingested file.
- Extract the document's core thesis, methods, findings, and open questions.
- Do not include author names, affiliations, or publication metadata in the summary.

## Concept Pages

- Concept pages synthesize across multiple source pages.
- Only create a concept page when two or more sources support the topic.
- Use explicit backlinks to source pages with wiki-link syntax.

## Analysis Pages

- Saved answers to user questions, stored under wiki/analysis/.
- Include the original question, the answer, and citation backlinks.
- Analysis pages are indexed and searchable like any other wiki page.

## Index Rules

- wiki/index.md catalogs all source, concept, and analysis pages.
- wiki/_index.json provides the same catalog in machine-readable form.
- The index is regenerated after every compile and after saving an analysis page.

## Log Rules

- wiki/log.md records every wiki-modifying action chronologically.
- Each entry uses a heading format: ## [ISO-date] action | details.
- Log entries must be parseable with grep or simple text tools.

## Query Behavior

- Search the compiled wiki first using the local index.
- Answer from wiki evidence only; cite each claim with [Source Title].
- If the evidence is insufficient, say so explicitly.
- Saved answers compound into the wiki as analysis pages.

## Lint Goals

- Treat broken links and missing citations as errors.
- Treat empty summaries, orphan pages, and missing page-type fields as warnings.
- Treat weak cross-linking as a suggestion.
"""


def schema_excerpt(schema_text: str, headings: list[str]) -> str:
    """Extract specific sections from the schema by heading name.

    Returns the concatenated text of all matching ``## Heading`` sections.
    Sections are extracted in the order they appear in *headings*.
    """
    parts: list[str] = []
    for heading in headings:
        pattern = rf"(?m)^## {re.escape(heading)}\s*\n"
        match = re.search(pattern, schema_text)
        if match is None:
            continue
        start = match.start()
        next_heading = re.search(r"(?m)^## ", schema_text[match.end() :])
        end = match.end() + next_heading.start() if next_heading else len(schema_text)
        parts.append(schema_text[start:end].rstrip())
    return "\n\n".join(parts)


class ConfigService:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def load(self) -> dict[str, Any]:
        if not self.paths.config_file.exists():
            return deepcopy(DEFAULT_CONFIG)
        with self.paths.config_file.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("kb.config.yaml must contain a YAML mapping.")
        migrated, changed = _apply_config_migrations(loaded)
        if changed:
            atomic_write_text(
                self.paths.config_file,
                yaml.safe_dump(migrated, sort_keys=False),
            )
        merged = deepcopy(DEFAULT_CONFIG)
        merged = _deep_merge(merged, migrated)
        return _validate_config(merged)

    def load_schema(self) -> str:
        if not self.paths.schema_file.exists():
            return DEFAULT_SCHEMA
        return self.paths.schema_file.read_text(encoding="utf-8")

    def save(self, config: dict[str, Any]) -> None:
        """Write *config* back to kb.config.yaml (atomic)."""
        atomic_write_text(
            self.paths.config_file,
            yaml.safe_dump(config, sort_keys=False),
        )

    def ensure_files(self) -> list[str]:
        created: list[str] = []
        if not self.paths.config_file.exists():
            atomic_write_text(
                self.paths.config_file,
                yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False),
            )
            created.append(self.paths.config_file.name)
        if not self.paths.schema_file.exists():
            atomic_write_text(self.paths.schema_file, DEFAULT_SCHEMA)
            created.append(self.paths.schema_file.name)
        return created


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _config_version(config: dict[str, Any]) -> int:
    version = config.get("version", 1)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("kb.config.yaml version must be an integer.")
    if version < 1:
        raise ValueError("kb.config.yaml version must be >= 1.")
    return version


def _apply_config_migrations(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    migrated = deepcopy(config)
    changed = False
    version = _config_version(migrated)
    if version > CURRENT_CONFIG_VERSION:
        raise ValueError(
            "Unsupported kb.config.yaml version: "
            f"{version}. This CLI supports up to version {CURRENT_CONFIG_VERSION}."
        )

    while version < CURRENT_CONFIG_VERSION:
        if version == 1:
            migrated = _migrate_v1_to_v2(migrated)
            changed = True
            version = _config_version(migrated)
            continue
        if version == 2:
            migrated = _migrate_v2_to_v3(migrated)
            changed = True
            version = _config_version(migrated)
            continue
        if version == 3:
            migrated = _migrate_v3_to_v4(migrated)
            changed = True
            version = _config_version(migrated)
            continue
        raise ValueError(f"Unsupported kb.config.yaml version: {version}")

    return migrated, changed


def _migrate_v1_to_v2(config: dict[str, Any]) -> dict[str, Any]:
    migrated = deepcopy(config)
    storage = migrated.setdefault("storage", {})
    if isinstance(storage, dict):
        storage.setdefault(
            "raw_normalized_dir",
            DEFAULT_CONFIG["storage"]["raw_normalized_dir"],
        )

    compile_config = migrated.setdefault("compile", {})
    if isinstance(compile_config, dict):
        compile_config.pop("summary_paragraph_limit", None)

    migrated.setdefault("provider", {})
    migrated["version"] = 2
    return migrated


def _migrate_v2_to_v3(config: dict[str, Any]) -> dict[str, Any]:
    migrated = deepcopy(config)
    existing_providers = migrated.get("providers", {})
    providers = deepcopy(DEFAULT_CONFIG["providers"])
    if isinstance(existing_providers, dict):
        for name, entry in existing_providers.items():
            if isinstance(entry, dict) and isinstance(providers.get(name), dict):
                providers[name] = _deep_merge(providers[name], entry)
    migrated["providers"] = providers

    provider_section = migrated.setdefault("provider", {})
    if isinstance(provider_section, dict):
        provider_section.pop("tier", None)
        name = str(provider_section.get("name", "")).strip().lower()
        if name in DEFAULT_CONFIG["providers"]:
            target = providers.setdefault(
                name,
                deepcopy(DEFAULT_CONFIG["providers"][name]),
            )
            if not isinstance(target, dict):
                target = deepcopy(DEFAULT_CONFIG["providers"][name])
                providers[name] = target
            for key in ("model", "api_key_env", "reasoning_effort", "thinking_budget"):
                if key in provider_section:
                    target[key] = provider_section.pop(key)

    migrated["version"] = 3
    return migrated


def _migrate_v3_to_v4(config: dict[str, Any]) -> dict[str, Any]:
    migrated = deepcopy(config)
    existing_conversion = migrated.get("conversion", {})
    conversion = deepcopy(DEFAULT_CONFIG["conversion"])
    if isinstance(existing_conversion, dict):
        conversion = _deep_merge(conversion, existing_conversion)
    migrated["conversion"] = conversion
    migrated["version"] = 4
    return migrated


class _StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _ProviderSelection(_StrictConfigModel):
    name: StrictStr | None = None


class _OpenAIProviderConfig(_StrictConfigModel):
    model: StrictStr
    api_key_env: StrictStr
    reasoning_effort: StrictStr

    @field_validator("model", "api_key_env", "reasoning_effort")
    @classmethod
    def _must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class _AnthropicProviderConfig(_StrictConfigModel):
    model: StrictStr
    api_key_env: StrictStr
    thinking_budget: StrictInt = Field(ge=0)

    @field_validator("model", "api_key_env")
    @classmethod
    def _must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class _GeminiProviderConfig(_OpenAIProviderConfig):
    pass


class _ProvidersConfig(_StrictConfigModel):
    openai: _OpenAIProviderConfig
    anthropic: _AnthropicProviderConfig
    gemini: _GeminiProviderConfig


class _MistralOcrConfig(_StrictConfigModel):
    model: StrictStr
    api_key_env: StrictStr
    table_format: Literal["markdown", "html"]

    @field_validator("model", "api_key_env")
    @classmethod
    def _must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class _HtmlConversionConfig(_StrictConfigModel):
    renderer: Literal["wkhtmltopdf"]
    wkhtmltopdf_path: StrictStr | None = None

    @field_validator("wkhtmltopdf_path")
    @classmethod
    def _path_must_be_non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must be null or a non-empty string")
        return value


class _FallbacksConfig(_StrictConfigModel):
    pdf: Literal["docling"]
    docx: Literal["markitdown"]
    pptx: Literal["markitdown"]
    html: Literal["markitdown"]


class _ConversionConfig(_StrictConfigModel):
    mistral_ocr: _MistralOcrConfig
    html: _HtmlConversionConfig
    fallbacks: _FallbacksConfig


class _KbConfigModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: StrictInt = Field(ge=1)
    project: dict[str, Any]
    storage: dict[str, Any]
    compile: dict[str, Any]
    lint: dict[str, Any]
    provider: _ProviderSelection = Field(default_factory=_ProviderSelection)
    providers: _ProvidersConfig
    conversion: _ConversionConfig


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    try:
        model = _KbConfigModel.model_validate(config)
    except ValidationError as exc:
        raise ValueError(_format_config_validation_error(exc)) from exc

    validated = model.model_dump(mode="python")
    if validated.get("provider", {}).get("name") is None:
        validated["provider"] = {}
    return validated


def _format_config_validation_error(exc: ValidationError) -> str:
    error = exc.errors()[0]
    loc_parts = tuple(str(part) for part in error.get("loc", ()))
    location = ".".join(loc_parts) or "root"
    error_type = str(error.get("type", ""))

    if loc_parts[:1] == ("provider",) and error_type == "extra_forbidden":
        return (
            "kb.config.yaml 'provider' only supports 'name'. Move provider settings "
            "under the top-level 'providers' section."
        )

    if loc_parts == ("conversion",) and error_type == "model_type":
        return "kb.config.yaml 'conversion' must contain a YAML mapping."
    if loc_parts[:1] == ("conversion",) and error_type == "extra_forbidden":
        if len(loc_parts) == 2:
            return (
                "kb.config.yaml 'conversion' contains unknown sections: "
                f"{loc_parts[1]}."
            )

    conversion_sections = {"mistral_ocr", "html", "fallbacks"}
    if len(loc_parts) >= 2 and loc_parts[0] == "conversion":
        section = loc_parts[1]
        if section in conversion_sections and error_type == "model_type":
            return f"kb.config.yaml 'conversion.{section}' must contain a YAML mapping."
        if section in conversion_sections and error_type == "extra_forbidden":
            return (
                f"kb.config.yaml 'conversion.{section}' contains unknown keys: "
                f"{loc_parts[-1]}."
            )

    message = error.get("msg", "is invalid")
    return f"kb.config.yaml '{location}' {message}."


def _validate_provider_configs(config: dict[str, Any]) -> None:
    _validate_config(config)


def _validate_conversion_config(config: dict[str, Any]) -> None:
    _validate_config(config)
