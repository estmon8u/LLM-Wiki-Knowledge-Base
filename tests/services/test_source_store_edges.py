"""Edge cases for source recommendation store."""

from __future__ import annotations

import pytest

from graphwiki_kb.services.source_recommendation_store import SourceRecommendationStore


def test_resolve_unknown_id_raises(test_project) -> None:
    store = SourceRecommendationStore(test_project.paths)
    with pytest.raises(ValueError, match="No research run found"):
        store.resolve_recommendations([1])
