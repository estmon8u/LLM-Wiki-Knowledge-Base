"""Tests for the LightRAG-style WikiGraphRAG config (v9 migration + resolvers)."""

from __future__ import annotations

from copy import deepcopy

import pytest

from graphwiki_kb.services.config_service import (
    CURRENT_CONFIG_VERSION,
    DEFAULT_CONFIG,
    EmbeddingsRuntimeConfig,
    LightRagRuntimeConfig,
    _apply_config_migrations,
    _validate_config,
    resolve_embeddings_config,
    resolve_wikigraph_config,
)


def test_current_version_is_nine() -> None:
    """The LightRAG backend bumps the config schema to version 9."""
    assert CURRENT_CONFIG_VERSION == 9


def test_default_config_has_lightrag_and_embeddings() -> None:
    """Defaults validate and expose the new sections with classic mode."""
    validated = _validate_config(deepcopy(DEFAULT_CONFIG))
    wikigraph = validated["wikigraph"]
    assert wikigraph["mode"] == "classic"
    lightrag = wikigraph["lightrag"]
    assert lightrag["chunk_token_size"] == 1200
    assert lightrag["chunk_overlap_tokens"] == 100
    assert lightrag["entity_extract_max_gleaning"] == 1
    assert "MODEL" in lightrag["entity_types"]
    assert "USES" in lightrag["relation_types"]
    assert lightrag["retrieval"]["default_method"] == "hybrid"
    assert lightrag["embeddings"]["local_fallback"] == "bm25"
    assert validated["embeddings"]["provider"] == "openai"
    assert validated["embeddings"]["model"] == "text-embedding-3-large"
    assert validated["embeddings"]["dimension"] == 3072


def test_v8_config_migrates_to_v9() -> None:
    """A v8 config without lightrag/embeddings gains them on migration."""
    old = deepcopy(DEFAULT_CONFIG)
    old["version"] = 8
    old["wikigraph"].pop("mode", None)
    old["wikigraph"].pop("lightrag", None)
    old.pop("embeddings", None)

    migrated, changed = _apply_config_migrations(old)

    assert changed is True
    assert migrated["version"] == CURRENT_CONFIG_VERSION
    assert migrated["wikigraph"]["mode"] == "classic"
    assert "lightrag" in migrated["wikigraph"]
    assert migrated["embeddings"]["model"] == "text-embedding-3-large"


def test_v8_migration_preserves_user_embedding_overrides() -> None:
    """User-supplied embedding overrides survive the v8 -> v9 migration."""
    old = deepcopy(DEFAULT_CONFIG)
    old["version"] = 8
    old["embeddings"] = {"model": "text-embedding-3-small", "dimension": 1536}

    migrated, _ = _apply_config_migrations(old)

    assert migrated["embeddings"]["model"] == "text-embedding-3-small"
    assert migrated["embeddings"]["dimension"] == 1536
    # Untouched defaults still arrive from DEFAULT_CONFIG.
    assert migrated["embeddings"]["provider"] == "openai"


def test_unknown_lightrag_key_rejected() -> None:
    """Strict validation rejects unknown wikigraph keys (lightrag typos)."""
    bad = deepcopy(DEFAULT_CONFIG)
    bad["wikigraph"]["lightrag"]["not_a_real_key"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        resolve_wikigraph_config(bad)


def test_invalid_mode_rejected() -> None:
    """The mode must be one of classic|lightrag."""
    bad = deepcopy(DEFAULT_CONFIG)
    bad["wikigraph"]["mode"] = "nonsense"
    with pytest.raises(ValueError):
        resolve_wikigraph_config(bad)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    """chunk_overlap_tokens >= chunk_token_size is rejected."""
    bad = deepcopy(DEFAULT_CONFIG)
    bad["wikigraph"]["lightrag"]["chunk_overlap_tokens"] = 5000
    bad["wikigraph"]["lightrag"]["chunk_token_size"] = 1200
    with pytest.raises(ValueError):
        resolve_wikigraph_config(bad)


def test_entity_types_normalized_and_deduped() -> None:
    """Entity/relation types are uppercased and de-duplicated."""
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["wikigraph"]["lightrag"]["entity_types"] = ["model", "Model", "method"]
    runtime = resolve_wikigraph_config(cfg)
    assert runtime.lightrag.entity_types == ("MODEL", "METHOD")


def test_empty_entity_types_rejected() -> None:
    """An empty entity-type list is rejected."""
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["wikigraph"]["lightrag"]["entity_types"] = ["   "]
    with pytest.raises(ValueError):
        resolve_wikigraph_config(cfg)


def test_resolve_wikigraph_config_populates_lightrag() -> None:
    """The resolved runtime config exposes a typed LightRagRuntimeConfig."""
    runtime = resolve_wikigraph_config(deepcopy(DEFAULT_CONFIG))
    assert runtime.mode == "classic"
    assert isinstance(runtime.lightrag, LightRagRuntimeConfig)
    assert runtime.lightrag.chunk_token_size == 1200
    assert runtime.lightrag.retrieval.top_k_entities == 12
    assert runtime.lightrag.retrieval.max_total_tokens == 24000
    assert runtime.lightrag.embeddings_required_for_strict is True
    assert runtime.lightrag.local_fallback == "bm25"


def test_resolve_embeddings_config_defaults_api_key_env() -> None:
    """Embedding config resolves provider + default api_key_env."""
    resolved = resolve_embeddings_config(deepcopy(DEFAULT_CONFIG))
    assert isinstance(resolved, EmbeddingsRuntimeConfig)
    assert resolved.provider == "openai"
    assert resolved.dimension == 3072
    assert resolved.api_key_env == "OPENAI_API_KEY"


def test_resolve_embeddings_config_explicit_key_env() -> None:
    """An explicit api_key_env is honored; gemini default derived otherwise."""
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["embeddings"]["provider"] = "gemini"
    cfg["embeddings"]["api_key_env"] = None
    assert resolve_embeddings_config(cfg).api_key_env == "GEMINI_API_KEY"

    cfg["embeddings"]["api_key_env"] = "CUSTOM_EMBED_KEY"
    assert resolve_embeddings_config(cfg).api_key_env == "CUSTOM_EMBED_KEY"


def test_resolve_embeddings_config_rejects_unknown_key() -> None:
    """Unknown embedding keys raise a friendly error."""
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["embeddings"]["bogus"] = True
    with pytest.raises(ValueError, match="unknown keys"):
        resolve_embeddings_config(cfg)


def test_resolve_embeddings_config_rejects_non_mapping() -> None:
    """A non-mapping embeddings section raises a friendly error."""
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["embeddings"] = ["not", "a", "mapping"]
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        resolve_embeddings_config(cfg)
