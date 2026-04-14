from __future__ import annotations

from dataclasses import dataclass

import yaml

from src.models.wiki_models import SearchResult
from src.services.project_service import ProjectPaths, slugify, utc_now_iso
from src.services.search_service import SearchService


@dataclass
class QueryAnswer:
    answer: str
    citations: list[SearchResult]
    saved_path: str | None = None


class QueryService:
    def __init__(self, paths: ProjectPaths, search_service: SearchService) -> None:
        self.paths = paths
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

    def save_answer(self, question: str, answer: QueryAnswer) -> str:
        slug = slugify(question)
        if not slug or slug == "untitled":
            slug = "analysis-" + slugify(answer.answer[:40])
        timestamp = utc_now_iso()
        frontmatter = {
            "title": question,
            "type": "analysis",
            "question": question,
            "saved_at": timestamp,
            "citations": [c.path for c in answer.citations],
        }
        yaml_block = yaml.safe_dump(frontmatter, sort_keys=False).strip()
        citation_lines = ""
        if answer.citations:
            citation_lines = "\n".join(
                f"- [[{c.title}]] (`{c.path}`)" for c in answer.citations
            )
        page_text = (
            f"---\n{yaml_block}\n---\n\n"
            f"# {question}\n\n"
            "## Answer\n\n"
            f"{answer.answer}\n\n"
            "## Citations\n\n"
            f"{citation_lines or 'No citations.'}\n"
        )
        dest = self.paths.wiki_concepts_dir / f"{slug}.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(page_text, encoding="utf-8")
        return dest.relative_to(self.paths.root).as_posix()
