"""Configuration defaults, migrations, validation, and persistence."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
)

from graphwiki_kb.services.file_lock import file_lock
from graphwiki_kb.services.graphrag_defaults import (
    DEFAULT_GRAPHRAG_API_KEY_ENV,
    DEFAULT_GRAPHRAG_CHUNK_OVERLAP,
    DEFAULT_GRAPHRAG_CHUNK_SIZE,
    DEFAULT_GRAPHRAG_EMBEDDING_MODEL,
    DEFAULT_GRAPHRAG_ENTITY_TYPES,
    DEFAULT_GRAPHRAG_EXTRACTION_MAX_GLEANINGS,
    DEFAULT_GRAPHRAG_MAX_SOURCE_BYTES,
    DEFAULT_GRAPHRAG_MODEL,
    DEFAULT_GRAPHRAG_PROVIDER,
    LEGACY_GRAPHRAG_API_KEY_ENV,
)
from graphwiki_kb.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    utc_now_iso,
)

CURRENT_CONFIG_VERSION = 7
PROVIDER_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
GEMINI_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high"}
ANTHROPIC_THINKING_EFFORTS = {"low", "medium", "high", "xhigh", "max"}


@dataclass(frozen=True)
class GraphRAGRuntimeConfig:
    """Resolved provider and embedding settings for GraphRAG runtime calls."""

    provider: str
    model: str
    embedding_provider: str
    embedding_model: str
    api_key_env: str
    embedding_api_key_env: str
    chunk_size: int
    chunk_overlap: int
    entity_types: tuple[str, ...]
    max_gleanings: int
    max_source_bytes: int


DEFAULT_CONFIG: dict[str, Any] = {
    "version": CURRENT_CONFIG_VERSION,
    "project": {
        "name": "Capstone Knowledge Base",
        "description": "CLI-first GraphRAG research-memory system with inspectable wiki artifacts.",
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
    "concepts": {
        "enabled": False,
        "provider_backed": False,
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
    "graph": {
        "provider": DEFAULT_GRAPHRAG_PROVIDER,
        "model": DEFAULT_GRAPHRAG_MODEL,
        "embedding_provider": DEFAULT_GRAPHRAG_PROVIDER,
        "embedding_model": DEFAULT_GRAPHRAG_EMBEDDING_MODEL,
        "api_key_env": None,
        "embedding_api_key_env": None,
        "chunking": {
            "size": DEFAULT_GRAPHRAG_CHUNK_SIZE,
            "overlap": DEFAULT_GRAPHRAG_CHUNK_OVERLAP,
        },
        "extraction": {
            "entity_types": list(DEFAULT_GRAPHRAG_ENTITY_TYPES),
            "max_gleanings": DEFAULT_GRAPHRAG_EXTRACTION_MAX_GLEANINGS,
        },
        "input": {
            "max_source_bytes": DEFAULT_GRAPHRAG_MAX_SOURCE_BYTES,
        },
        "routing": {
            "aliases": {},
        },
    },
    "providers": {
        "openai": {
            "model": "gpt-5.4-nano",
            "api_key_env": "OPENAI_API_KEY",
            "reasoning_effort": "high",
            "api": "responses",
            "store_responses": False,
        },
        "anthropic": {
            "model": "claude-sonnet-4-6",
            "api_key_env": "ANTHROPIC_API_KEY",
            "thinking_effort": "medium",
        },
        "gemini": {
            "model": "gemini-2.5-flash",
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
            "allow_local_file_access": False,
        },
        "fallbacks": {
            "pdf": ["docling", "markitdown"],
            "docx": ["markitdown"],
            "pptx": ["markitdown"],
            "html": ["markitdown"],
        },
    },
    "extensions": {},
    "agent": {
        "enabled": True,
        "model": "gpt-5.4-nano",
        "max_turns": 8,
        "require_approval_for_writes": True,
        "save_runs": True,
        "trace": True,
        "session_backend": "sqlite",
    },
    "research": {
        "web_enabled": True,
        "web_model": "gpt-5.4-nano",
        "search_context_size": "medium",
        "max_recommendations": 5,
        "max_web_sources": 12,
        "require_approval_for_ingest": True,
        "default_domains_allowlist": [],
        "default_domains_blocklist": ["reddit.com", "quora.com"],
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
- Concept page generation is opt-in. GraphRAG is the default cross-document retrieval layer.

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

- Use GraphRAG for top-level `kb ask` when graph output is complete.
- Use the local wiki index for `kb find` and deprecated legacy comparators.
- Answer from wiki evidence only; cite each legacy claim with [Source Title] and record graph source trace/support metadata for GraphRAG answers.
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
    """Loads, validates, migrates, and writes project config files."""

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def load(self) -> dict[str, Any]:
        """Load config from disk or return defaults for a new project."""
        if not self.paths.config_file.exists():
            return deepcopy(DEFAULT_CONFIG)
        with file_lock(self.paths.config_file):
            with self.paths.config_file.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            if not isinstance(loaded, dict):
                raise ValueError("kb.config.yaml must contain a YAML mapping.")
            migrated, changed = _apply_config_migrations(loaded)
            merged = deepcopy(DEFAULT_CONFIG)
            merged = _deep_merge(merged, migrated)
            validated = _validate_config(merged)
            if changed:
                atomic_write_text(
                    self.paths.config_file,
                    yaml.safe_dump(migrated, sort_keys=False),
                )
            return validated

    def load_schema(self) -> str:
        """Loads schema.

        Returns:
            str produced by the operation.
        """
        if not self.paths.schema_file.exists():
            return DEFAULT_SCHEMA
        return self.paths.schema_file.read_text(encoding="utf-8")

    def save(self, config: dict[str, Any]) -> None:
        """Write *config* back to kb.config.yaml (atomic)."""
        with file_lock(self.paths.config_file):
            atomic_write_text(
                self.paths.config_file,
                yaml.safe_dump(config, sort_keys=False),
            )

    def ensure_files(self, *, repair_invalid: bool = False) -> list[str]:
        """Ensure files.

        Returns:
            list[str] produced by the operation.
        """
        created: list[str] = []
        if not self.paths.config_file.exists():
            with file_lock(self.paths.config_file):
                atomic_write_text(
                    self.paths.config_file,
                    yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False),
                )
            created.append(self.paths.config_file.name)
        elif repair_invalid:
            try:
                self.load()
            except (ValueError, yaml.YAMLError):
                with file_lock(self.paths.config_file):
                    backup_name = _backup_config_file(self.paths.config_file)
                    atomic_write_text(
                        self.paths.config_file,
                        yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False),
                    )
                created.append(
                    f"{self.paths.config_file.name} (regenerated; backup: {backup_name})"
                )
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


def _backup_config_file(config_file: Path) -> str:
    timestamp = re.sub(r"[^0-9A-Za-z]+", "-", utc_now_iso()).strip("-")
    backup_path = config_file.with_name(f"{config_file.name}.bak.{timestamp}")
    backup_path.write_text(config_file.read_text(encoding="utf-8"), encoding="utf-8")
    return backup_path.name


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
        if version == 4:
            migrated = _migrate_v4_to_v5(migrated)
            changed = True
            version = _config_version(migrated)
            continue
        if version == 5:
            migrated = _migrate_v5_to_v6(migrated)
            changed = True
            version = _config_version(migrated)
            continue
        if version == 6:
            migrated = _migrate_v6_to_v7(migrated)
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
            for key in (
                "model",
                "api_key_env",
                "reasoning_effort",
                "thinking_budget",
                "thinking_effort",
            ):
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


def _migrate_v4_to_v5(config: dict[str, Any]) -> dict[str, Any]:
    migrated = deepcopy(config)
    existing_graph = migrated.get("graph", {})
    graph = deepcopy(DEFAULT_CONFIG["graph"])
    if isinstance(existing_graph, dict):
        graph = _deep_merge(graph, existing_graph)
        if "embedding_provider" not in existing_graph:
            graph["embedding_provider"] = graph.get("provider")
    migrated["graph"] = graph
    migrated["version"] = 5
    return migrated


def _migrate_v5_to_v6(config: dict[str, Any]) -> dict[str, Any]:
    migrated = deepcopy(config)
    existing_graph = migrated.get("graph", {})
    graph = deepcopy(DEFAULT_CONFIG["graph"])
    if isinstance(existing_graph, dict):
        graph = _deep_merge(graph, existing_graph)
        if "embedding_provider" not in existing_graph:
            graph["embedding_provider"] = graph.get("provider")
        if graph.get("api_key_env") == LEGACY_GRAPHRAG_API_KEY_ENV:
            graph["api_key_env"] = None
        if graph.get("embedding_api_key_env") == LEGACY_GRAPHRAG_API_KEY_ENV:
            graph["embedding_api_key_env"] = None
    migrated["graph"] = graph
    migrated["version"] = 6
    return migrated


def _migrate_v6_to_v7(config: dict[str, Any]) -> dict[str, Any]:
    migrated = deepcopy(config)

    existing_graph = migrated.get("graph", {})
    graph = deepcopy(DEFAULT_CONFIG["graph"])
    if isinstance(existing_graph, dict):
        graph = _deep_merge(graph, existing_graph)
    graph.setdefault("routing", deepcopy(DEFAULT_CONFIG["graph"]["routing"]))
    migrated["graph"] = graph

    existing_providers = migrated.get("providers", {})
    providers = deepcopy(DEFAULT_CONFIG["providers"])
    if isinstance(existing_providers, dict):
        for name, entry in existing_providers.items():
            if isinstance(providers.get(name), dict):
                if isinstance(entry, dict):
                    providers[name] = _deep_merge(providers[name], entry)
                else:
                    providers[name] = entry
    migrated["providers"] = providers

    existing_conversion = migrated.get("conversion", {})
    if "conversion" in migrated and not isinstance(existing_conversion, dict):
        migrated["conversion"] = existing_conversion
    else:
        conversion = deepcopy(DEFAULT_CONFIG["conversion"])
        if isinstance(existing_conversion, dict):
            conversion = _deep_merge(conversion, existing_conversion)
        migrated["conversion"] = conversion

    migrated["version"] = 7
    return migrated


class _StrictConfigModel(BaseModel):
    """Base model that rejects unknown keys for nested config sections."""

    model_config = ConfigDict(extra="forbid")


class _ProviderSelection(_StrictConfigModel):
    """Top-level active provider selection."""

    name: StrictStr | None = None


class _OpenAIProviderConfig(_StrictConfigModel):
    """Validated OpenAI provider settings."""

    model: StrictStr
    api_key_env: StrictStr
    reasoning_effort: StrictStr
    api: Literal["responses", "chat_completions"] = "responses"
    store_responses: StrictBool = False

    @field_validator("model", "api_key_env", "reasoning_effort")
    @classmethod
    def _must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("reasoning_effort")
    @classmethod
    def _reasoning_effort_must_be_supported(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in PROVIDER_REASONING_EFFORTS:
            supported = ", ".join(sorted(PROVIDER_REASONING_EFFORTS))
            raise ValueError(f"must be one of: {supported}")
        return normalized


class _AnthropicProviderConfig(_StrictConfigModel):
    """Validated Anthropic provider settings."""

    model: StrictStr
    api_key_env: StrictStr
    thinking_budget: StrictInt | None = Field(default=None, ge=0)
    thinking_effort: StrictStr = "medium"

    @field_validator("model", "api_key_env", "thinking_effort")
    @classmethod
    def _must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("thinking_effort")
    @classmethod
    def _thinking_effort_must_be_supported(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ANTHROPIC_THINKING_EFFORTS:
            supported = ", ".join(sorted(ANTHROPIC_THINKING_EFFORTS))
            raise ValueError(f"must be one of: {supported}")
        return normalized


class _GeminiProviderConfig(_StrictConfigModel):
    """Validated Gemini provider settings."""

    model: StrictStr
    api_key_env: StrictStr
    reasoning_effort: StrictStr

    @field_validator("model", "api_key_env", "reasoning_effort")
    @classmethod
    def _must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("reasoning_effort")
    @classmethod
    def _reasoning_effort_must_be_supported(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in GEMINI_REASONING_EFFORTS:
            supported = ", ".join(sorted(GEMINI_REASONING_EFFORTS))
            raise ValueError(f"must be one of: {supported}")
        return normalized


class _ProvidersConfig(_StrictConfigModel):
    """Validated settings for all configured text providers."""

    openai: _OpenAIProviderConfig
    anthropic: _AnthropicProviderConfig
    gemini: _GeminiProviderConfig


class _GraphRoutingConfig(_StrictConfigModel):
    aliases: dict[StrictStr, list[StrictStr]] = Field(default_factory=dict)

    @field_validator("aliases")
    @classmethod
    def _aliases_must_have_non_empty_terms(
        cls, value: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        normalized: dict[str, list[str]] = {}
        for name, terms in value.items():
            alias_name = str(name).strip()
            if not alias_name:
                raise ValueError("alias names must be non-empty strings")
            if not terms:
                raise ValueError("alias terms must not be empty")
            normalized_terms: list[str] = []
            for term in terms:
                stripped = str(term).strip()
                if not stripped:
                    raise ValueError("alias terms must be non-empty strings")
                normalized_terms.append(stripped)
            normalized[alias_name] = normalized_terms
        return normalized


class _GraphChunkingConfig(_StrictConfigModel):
    """Validated GraphRAG chunking settings managed from kb.config.yaml."""

    size: StrictInt = Field(default=DEFAULT_GRAPHRAG_CHUNK_SIZE, ge=100)
    overlap: StrictInt = Field(default=DEFAULT_GRAPHRAG_CHUNK_OVERLAP, ge=0)

    @field_validator("overlap")
    @classmethod
    def _overlap_must_be_smaller_than_size(cls, value: int, info: Any) -> int:
        size = info.data.get("size")
        if isinstance(size, int) and value >= size:
            raise ValueError("must be smaller than chunking.size")
        return value


class _GraphExtractionConfig(_StrictConfigModel):
    """Validated GraphRAG entity extraction settings."""

    entity_types: list[StrictStr] = Field(
        default_factory=lambda: list(DEFAULT_GRAPHRAG_ENTITY_TYPES)
    )
    max_gleanings: StrictInt = Field(
        default=DEFAULT_GRAPHRAG_EXTRACTION_MAX_GLEANINGS,
        ge=0,
    )

    @field_validator("entity_types")
    @classmethod
    def _entity_types_must_be_unique_non_empty(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            entity_type = str(item).strip().lower()
            if not entity_type:
                raise ValueError("entity types must be non-empty strings")
            normalized.append(entity_type)
        if not normalized:
            raise ValueError("must include at least one entity type")
        if len(set(normalized)) != len(normalized):
            raise ValueError("must not repeat entity types")
        return normalized


class _GraphInputConfig(_StrictConfigModel):
    """Validated GraphRAG input sync limits."""

    max_source_bytes: StrictInt = Field(
        default=DEFAULT_GRAPHRAG_MAX_SOURCE_BYTES,
        ge=1,
    )


class _GraphConfig(_StrictConfigModel):
    """Validated GraphRAG provider, embedding, and routing settings."""

    provider: StrictStr
    model: StrictStr
    embedding_provider: StrictStr | None = None
    embedding_model: StrictStr
    api_key_env: StrictStr | None = None
    embedding_api_key_env: StrictStr | None = None
    chunking: _GraphChunkingConfig = Field(default_factory=_GraphChunkingConfig)
    extraction: _GraphExtractionConfig = Field(default_factory=_GraphExtractionConfig)
    input: _GraphInputConfig = Field(default_factory=_GraphInputConfig)
    routing: _GraphRoutingConfig = Field(default_factory=_GraphRoutingConfig)

    @field_validator("provider", "model", "embedding_model")
    @classmethod
    def _must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("embedding_provider", "api_key_env", "embedding_api_key_env")
    @classmethod
    def _optional_value_must_be_non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must be null or a non-empty string")
        return value


class _ConceptsConfig(_StrictConfigModel):
    """Configuration for legacy LLM-wiki concept page generation."""

    enabled: StrictBool = False
    provider_backed: StrictBool = False


class _MistralOcrConfig(_StrictConfigModel):
    """Validated Mistral OCR conversion settings."""

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
    """Validated HTML renderer settings."""

    renderer: Literal["wkhtmltopdf"]
    wkhtmltopdf_path: StrictStr | None = None
    allow_local_file_access: StrictBool = False

    @field_validator("wkhtmltopdf_path")
    @classmethod
    def _path_must_be_non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must be null or a non-empty string")
        return value


class _FallbacksConfig(_StrictConfigModel):
    """Validated fallback converter choices."""

    pdf: list[Literal["docling", "markitdown"]]
    docx: list[Literal["markitdown"]]
    pptx: list[Literal["markitdown"]]
    html: list[Literal["markitdown"]]

    @field_validator("pdf", "docx", "pptx", "html", mode="before")
    @classmethod
    def _coerce_fallback_list(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value

    @field_validator("pdf", "docx", "pptx", "html")
    @classmethod
    def _fallbacks_must_be_non_empty_and_unique(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must include at least one fallback converter")
        if len(set(value)) != len(value):
            raise ValueError("must not repeat fallback converters")
        return value


class _ConversionConfig(_StrictConfigModel):
    """Validated document conversion settings."""

    mistral_ocr: _MistralOcrConfig
    html: _HtmlConversionConfig
    fallbacks: _FallbacksConfig


class _AgentConfig(_StrictConfigModel):
    """Natural-language kb agent settings."""

    enabled: StrictBool = True
    model: StrictStr
    max_turns: StrictInt = Field(ge=1, le=50)
    require_approval_for_writes: StrictBool = True
    save_runs: StrictBool = True
    trace: StrictBool = True
    session_backend: Literal["sqlite", "none"] = "sqlite"


class _ResearchConfig(_StrictConfigModel):
    """Web-backed research settings for kb agent."""

    web_enabled: StrictBool = True
    web_model: StrictStr
    search_context_size: Literal["low", "medium", "high"] = "medium"
    max_recommendations: StrictInt = Field(ge=1, le=25)
    max_web_sources: StrictInt = Field(ge=1, le=50)
    require_approval_for_ingest: StrictBool = True
    default_domains_allowlist: list[StrictStr] = Field(default_factory=list)
    default_domains_blocklist: list[StrictStr] = Field(default_factory=list)


class _KbConfigModel(BaseModel):
    """Top-level config schema with strict nested sections."""

    model_config = ConfigDict(extra="forbid")

    version: StrictInt = Field(ge=1)
    project: dict[str, Any]
    storage: dict[str, Any]
    compile: dict[str, Any]
    concepts: _ConceptsConfig = Field(default_factory=_ConceptsConfig)
    lint: dict[str, Any]
    provider: _ProviderSelection = Field(default_factory=_ProviderSelection)
    graph: _GraphConfig
    providers: _ProvidersConfig
    conversion: _ConversionConfig
    agent: _AgentConfig
    research: _ResearchConfig
    extensions: dict[str, Any] = Field(default_factory=dict)


def resolve_graph_config(config: dict[str, Any]) -> GraphRAGRuntimeConfig:
    """Resolve graph config.

    Args:
        config: Loaded knowledge-base configuration mapping.

    Returns:
        GraphRAGRuntimeConfig produced by the operation.
    """
    graph = config.get("graph", DEFAULT_CONFIG["graph"])
    if not isinstance(graph, dict):
        raise ValueError("kb.config.yaml 'graph' must contain a YAML mapping.")
    try:
        validated = _GraphConfig.model_validate(graph).model_dump(mode="python")
    except ValidationError as exc:
        error = exc.errors()[0]
        if str(error.get("type", "")) == "extra_forbidden":
            loc_parts = tuple(str(part) for part in error.get("loc", ()))
            key = loc_parts[-1] if loc_parts else "unknown"
            raise ValueError(
                f"kb.config.yaml 'graph' contains unknown keys: {key}."
            ) from exc
        raise ValueError(_format_config_validation_error(exc)) from exc
    provider = validated["provider"].strip()
    explicit_api_key_env = _optional_str(validated.get("api_key_env"))
    api_key_env = _resolve_provider_api_key_env(
        config,
        provider=provider,
        explicit_api_key_env=explicit_api_key_env,
        field_name="api_key_env",
    )
    embedding_provider = _optional_str(validated.get("embedding_provider")) or provider
    explicit_embedding_api_key_env = _optional_str(
        validated.get("embedding_api_key_env")
    )
    if explicit_embedding_api_key_env:
        embedding_api_key_env = explicit_embedding_api_key_env
    elif embedding_provider == provider and explicit_api_key_env:
        embedding_api_key_env = api_key_env
    else:
        embedding_api_key_env = _resolve_provider_api_key_env(
            config,
            provider=embedding_provider,
            explicit_api_key_env=None,
            field_name="embedding_api_key_env",
        )
    return GraphRAGRuntimeConfig(
        provider=provider,
        model=validated["model"].strip(),
        embedding_provider=embedding_provider,
        embedding_model=validated["embedding_model"].strip(),
        api_key_env=api_key_env,
        embedding_api_key_env=embedding_api_key_env,
        chunk_size=int(validated["chunking"]["size"]),
        chunk_overlap=int(validated["chunking"]["overlap"]),
        entity_types=tuple(validated["extraction"]["entity_types"]),
        max_gleanings=int(validated["extraction"]["max_gleanings"]),
        max_source_bytes=int(validated["input"]["max_source_bytes"]),
    )


def graph_routing_aliases(config: dict[str, Any]) -> dict[str, list[str]]:
    """Return configured corpus/domain aliases used by the graph query router."""
    graph = config.get("graph", DEFAULT_CONFIG["graph"])
    if not isinstance(graph, dict):
        return {}
    routing = graph.get("routing") or {}
    if not isinstance(routing, dict):
        return {}
    aliases = routing.get("aliases") or {}
    if not isinstance(aliases, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for name, terms in aliases.items():
        if not isinstance(name, str) or not isinstance(terms, list):
            continue
        clean_terms = [str(term).strip() for term in terms if str(term).strip()]
        if clean_terms:
            normalized[name.strip()] = clean_terms
    return normalized


def concept_generation_enabled(config: dict[str, Any]) -> bool:
    concepts = config.get("concepts", DEFAULT_CONFIG["concepts"])
    if not isinstance(concepts, dict):
        return False
    return bool(concepts.get("enabled", False))


def concept_provider_backed_enabled(config: dict[str, Any]) -> bool:
    concepts = config.get("concepts", DEFAULT_CONFIG["concepts"])
    if not isinstance(concepts, dict):
        return False
    return bool(concepts.get("provider_backed", False))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _resolve_provider_api_key_env(
    config: dict[str, Any],
    *,
    provider: str,
    explicit_api_key_env: str | None,
    field_name: str,
) -> str:
    if explicit_api_key_env:
        return explicit_api_key_env
    provider_api_key_env = _provider_catalog_api_key_env(config, provider)
    if provider_api_key_env:
        return provider_api_key_env
    if provider == DEFAULT_GRAPHRAG_PROVIDER:
        return DEFAULT_GRAPHRAG_API_KEY_ENV
    raise ValueError(
        "kb.config.yaml 'graph' must set "
        f"'{field_name}' when provider '{provider}' is not configured under "
        "'providers'."
    )


def _provider_catalog_api_key_env(
    config: dict[str, Any],
    provider: str,
) -> str | None:
    providers = config.get("providers", DEFAULT_CONFIG["providers"])
    if not isinstance(providers, dict):
        return None
    provider_config = providers.get(provider.strip().lower())
    if not isinstance(provider_config, dict):
        return None
    return _optional_str(provider_config.get("api_key_env"))


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

    if loc_parts == ("graph",) and error_type == "model_type":
        return "kb.config.yaml 'graph' must contain a YAML mapping."
    if loc_parts[:1] == ("graph",) and error_type == "extra_forbidden":
        return f"kb.config.yaml 'graph' contains unknown keys: {loc_parts[-1]}."

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

    if len(loc_parts) <= 1 and error_type == "extra_forbidden":
        key = loc_parts[-1] if loc_parts else "unknown"
        return (
            "kb.config.yaml contains unknown top-level keys: "
            f"{key}. Put custom values under 'extensions'."
        )

    message = error.get("msg", "is invalid")
    return f"kb.config.yaml '{location}' {message}."


def _validate_provider_configs(config: dict[str, Any]) -> None:
    _validate_config(config)


def _validate_conversion_config(config: dict[str, Any]) -> None:
    _validate_config(config)
