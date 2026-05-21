"""Low-level WikiGraphRAG query primitives (no provider calls)."""

from __future__ import annotations

from dataclasses import dataclass

from graphwiki_kb.wikigraph.context_builder import (
    ContextBuilderConfig,
    WikiGraphContextBuilder,
)
from graphwiki_kb.wikigraph.models import (
    QueryMethod,
    WikiGraphFindResult,
    WikiGraphIndex,
)

_WIKIGRAPH_GLOBAL_KEYWORDS = (
    "main theme",
    "main themes",
    "overall",
    "across",
    "patterns",
    "landscape",
    "whole corpus",
)
_WIKIGRAPH_DRIFT_KEYWORDS = (
    "compare",
    "differ",
    "difference",
    "tradeoff",
    "trade-off",
    "relate",
    "related to",
    "relationship",
    "relationship between",
    " versus ",
    " vs ",
    "contrast",
)


@dataclass
class WikiGraphQueryEngine:
    """Provider-free retrieval engine for the WikiGraphRAG backend."""

    index: WikiGraphIndex
    config: ContextBuilderConfig | None = None

    def __post_init__(self) -> None:
        self._builder = WikiGraphContextBuilder(self.index, config=self.config)

    @property
    def builder(self) -> WikiGraphContextBuilder:
        """Return the underlying :class:`WikiGraphContextBuilder`."""
        return self._builder

    def find(
        self, question: str, *, method: QueryMethod = "auto"
    ) -> WikiGraphFindResult:
        """Retrieve contexts for ``question`` using ``method``.

        ``method="auto"`` mirrors the GraphRAG router intent classes:
        comparison questions use ``drift-lite``, corpus/theme questions use
        ``global``, entity matches use ``local``, and unmatched questions fall
        back to ``basic``.
        """
        chosen_method: QueryMethod = method
        diagnostics: list[str] = []
        if method == "auto":
            auto_seed_entities = self._builder._match_entities(question)
            normalized_question = f" {question.casefold()} "
            if _matched_keywords(_WIKIGRAPH_DRIFT_KEYWORDS, normalized_question):
                chosen_method = "drift-lite"
            elif _matched_keywords(_WIKIGRAPH_GLOBAL_KEYWORDS, normalized_question):
                chosen_method = "global"
            elif auto_seed_entities:
                chosen_method = "local"
            else:
                chosen_method = "basic"
            diagnostics.append(f"auto-selected {chosen_method}")
        if chosen_method == "basic":
            contexts = self._builder.basic_search(question)
            return WikiGraphFindResult(
                query=question,
                method="basic",
                contexts=contexts,
                entities=[],
                communities=[],
                trace=[{"step": "basic_search", "contexts": len(contexts)}],
                diagnostics=diagnostics,
            )
        if chosen_method == "local":
            contexts, seed_entities = self._builder.local_search(question)
            return WikiGraphFindResult(
                query=question,
                method="local",
                contexts=contexts,
                entities=seed_entities,
                communities=[],
                trace=[
                    {
                        "step": "local_search",
                        "seed_entities": seed_entities,
                        "contexts": len(contexts),
                    }
                ],
                diagnostics=diagnostics,
            )
        if chosen_method == "global":
            contexts, community_ids = self._builder.global_search(question)
            return WikiGraphFindResult(
                query=question,
                method="global",
                contexts=contexts,
                entities=[],
                communities=community_ids,
                trace=[
                    {
                        "step": "global_search",
                        "communities": community_ids,
                        "contexts": len(contexts),
                    }
                ],
                diagnostics=diagnostics,
            )
        if chosen_method == "drift-lite":
            contexts, seed_entities, sub_questions = self._builder.drift_lite(question)
            return WikiGraphFindResult(
                query=question,
                method="drift-lite",
                contexts=contexts,
                entities=seed_entities,
                communities=[],
                trace=[
                    {
                        "step": "drift_lite",
                        "seed_entities": seed_entities,
                        "sub_questions": sub_questions,
                        "contexts": len(contexts),
                    }
                ],
                diagnostics=diagnostics,
            )
        raise ValueError(f"Unknown wikigraph method: {method}")


def _matched_keywords(keywords: tuple[str, ...], normalized_question: str) -> bool:
    """Return true when any routing keyword appears in the normalized query."""
    return any(keyword in normalized_question for keyword in keywords)
