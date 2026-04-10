from __future__ import annotations

from pathlib import Path
import re

from src.models.wiki_models import SearchResult
from src.services.project_service import ProjectPaths


class SearchService:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        terms = [term for term in re.split(r"\W+", query.lower()) if term]
        if not terms:
            return []
        results: list[SearchResult] = []
        for file_path in sorted(self.paths.wiki_dir.rglob("*.md")):
            text = file_path.read_text(encoding="utf-8")
            normalized = text.lower()
            score = sum(normalized.count(term) for term in terms)
            if score <= 0:
                continue
            snippet = _extract_snippet(text, terms)
            results.append(
                SearchResult(
                    title=file_path.stem.replace("-", " ").title(),
                    path=file_path.relative_to(self.paths.root).as_posix(),
                    score=score,
                    snippet=snippet,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]


def _extract_snippet(text: str, terms: list[str]) -> str:
    lowered = text.lower()
    first_position = min(
        (lowered.find(term) for term in terms if lowered.find(term) != -1), default=0
    )
    start = max(0, first_position - 80)
    end = min(len(text), first_position + 220)
    snippet = " ".join(text[start:end].split())
    return snippet or text[:220].strip()
