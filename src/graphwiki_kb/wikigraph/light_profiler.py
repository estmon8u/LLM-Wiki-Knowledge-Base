"""Profiling helpers for LightRAG entity/relation cards.

The heavy profiling work lives in :mod:`graphwiki_kb.wikigraph.light_deduper`;
this module exposes a narrow API for callers that only need profile refresh.
"""

from __future__ import annotations

from graphwiki_kb.wikigraph.light_deduper import dedupe_and_profile
from graphwiki_kb.wikigraph.light_extractor import LightExtractionResult
from graphwiki_kb.wikigraph.light_models import EntityProfile, RelationProfile

__all__ = ["dedupe_and_profile", "profile_from_extractions"]


def profile_from_extractions(
    extracted: list[tuple[str, LightExtractionResult]],
    *,
    existing_entities: list[EntityProfile] | None = None,
    existing_relations: list[RelationProfile] | None = None,
) -> tuple[list[EntityProfile], list[RelationProfile]]:
    """Build entity/relation profiles from chunk extraction results."""
    return dedupe_and_profile(
        extracted,
        existing_entities=existing_entities,
        existing_relations=existing_relations,
    )
