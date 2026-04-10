from __future__ import annotations

from dataclasses import dataclass

from kb.models.wiki_models import SearchResult
from kb.services.search_service import SearchService


@dataclass
class QueryAnswer:
    answer: str
    citations: list[SearchResult]


class QueryService:
    def __init__(self, search_service: SearchService) -> None:
        self.search_service = search_service

    def answer_question(self, question: str, *, limit: int = 3) -> QueryAnswer:
        matches = self.search_service.search(question, limit=limit)
        if not matches:
            return QueryAnswer(
                answer="No compiled wiki pages matched that question yet. Ingest more sources or re-run compile.",
                citations=[],
            )

        evidence_lines = [f"{match.title}: {match.snippet}" for match in matches]
        answer = " ".join(evidence_lines)
        return QueryAnswer(answer=answer, citations=matches)
