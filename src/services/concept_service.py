from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

import yaml

from src.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    slugify,
    utc_now_iso,
)


_WORD_PATTERN = re.compile(r"[a-z]+(?:-[a-z]+)*")
_MANAGED_START = "<!-- kb:concept-backlinks:start -->"
_MANAGED_END = "<!-- kb:concept-backlinks:end -->"
_MANAGED_BLOCK_PATTERN = re.compile(
    r"\n## Related Concept Pages\n\n<!-- kb:concept-backlinks:start -->.*?<!-- kb:concept-backlinks:end -->\n*",
    re.DOTALL,
)
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "not",
        "but",
        "can",
        "its",
        "also",
        "may",
        "more",
        "into",
        "each",
        "than",
        "which",
        "when",
        "how",
        "where",
        "what",
        "use",
        "used",
        "using",
        "such",
        "will",
        "been",
        "does",
        "should",
        "would",
        "could",
        "about",
        "other",
        "some",
        "them",
        "they",
        "their",
        "then",
        "only",
        "over",
        "most",
        "just",
        "paper",
        "papers",
        "approach",
        "approaches",
        "method",
        "methods",
        "model",
        "models",
        "system",
        "systems",
        "results",
        "tasks",
        "task",
        "based",
    }
)
_PHRASE_CANDIDATES = (
    "retrieval augmented generation",
    "question answering",
    "language models",
    "dense retrieval",
    "few shot learning",
    "self reflection",
    "pre training",
    "open domain question answering",
)
_MIN_SHARED_TERMS = 3
_MIN_JACCARD = 0.16


@dataclass
class ConceptGenerationResult:
    concept_paths: list[str]
    updated_source_paths: list[str]
    removed_paths: list[str]


@dataclass
class _SourcePage:
    file_path: Path
    relative_path: str
    slug: str
    title: str
    summary: str
    terms: set[str]


@dataclass
class _ConceptDraft:
    title: str
    slug: str
    summary: str
    topic_terms: list[str]
    source_pages: list[_SourcePage]


class ConceptService:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def generate(self) -> ConceptGenerationResult:
        source_pages = self._load_source_pages()
        drafts = self._build_concept_drafts(source_pages)
        managed_pages = self._list_managed_pages()

        concept_paths: list[str] = []
        for draft in drafts:
            concept_paths.append(self._write_concept_page(draft, managed_pages))

        kept = {self.paths.root / path for path in concept_paths}
        removed_paths: list[str] = []
        for existing_path in managed_pages:
            if existing_path not in kept and existing_path.exists():
                removed_paths.append(
                    existing_path.relative_to(self.paths.root).as_posix()
                )
                existing_path.unlink()

        updated_source_paths = self._maintain_source_backlinks(source_pages, drafts)
        return ConceptGenerationResult(
            concept_paths=sorted(concept_paths),
            updated_source_paths=sorted(updated_source_paths),
            removed_paths=sorted(removed_paths),
        )

    def _load_source_pages(self) -> list[_SourcePage]:
        pages: list[_SourcePage] = []
        if not self.paths.wiki_sources_dir.exists():
            return pages

        for page_path in sorted(self.paths.wiki_sources_dir.glob("*.md")):
            text = page_path.read_text(encoding="utf-8")
            frontmatter, content = _split_frontmatter(text)
            content = _MANAGED_BLOCK_PATTERN.sub("", content)
            title = str(
                frontmatter.get("title", page_path.stem.replace("-", " ").title())
            )
            summary = str(frontmatter.get("summary", "")).strip()
            terms = _extract_terms(f"{title}\n{summary}\n{content}")
            pages.append(
                _SourcePage(
                    file_path=page_path,
                    relative_path=page_path.relative_to(self.paths.root).as_posix(),
                    slug=page_path.stem,
                    title=title,
                    summary=summary,
                    terms=terms,
                )
            )
        return pages

    def _build_concept_drafts(
        self, source_pages: list[_SourcePage]
    ) -> list[_ConceptDraft]:
        if len(source_pages) < 2:
            return []

        groups = _connected_components(source_pages)
        drafts: list[_ConceptDraft] = []
        for group in groups:
            if len(group) < 2:
                continue
            topic_terms = _derive_topic_terms(group)
            if len(topic_terms) < 2:
                continue
            title = _format_concept_title(topic_terms)
            summary = _format_concept_summary(group, topic_terms)
            slug = slugify("-".join(topic_terms[:3]))
            drafts.append(
                _ConceptDraft(
                    title=title,
                    slug=slug,
                    summary=summary,
                    topic_terms=topic_terms,
                    source_pages=sorted(group, key=lambda page: page.title.casefold()),
                )
            )
        return drafts

    def _list_managed_pages(self) -> set[Path]:
        managed: set[Path] = set()
        if not self.paths.wiki_concepts_dir.exists():
            return managed
        for page_path in self.paths.wiki_concepts_dir.glob("*.md"):
            try:
                frontmatter, _ = _split_frontmatter(
                    page_path.read_text(encoding="utf-8")
                )
            except Exception:
                continue
            if (
                frontmatter.get("type") == "concept"
                and frontmatter.get("generator") == "concept-service-v1"
            ):
                managed.add(page_path)
        return managed

    def _write_concept_page(
        self,
        draft: _ConceptDraft,
        managed_pages: set[Path],
    ) -> str:
        destination = self._resolve_destination(draft.slug, managed_pages)
        frontmatter = {
            "title": draft.title,
            "summary": draft.summary,
            "type": "concept",
            "generated_at": utc_now_iso(),
            "generator": "concept-service-v1",
            "source_pages": [page.relative_path for page in draft.source_pages],
            "topic_terms": draft.topic_terms,
        }
        yaml_block = yaml.safe_dump(frontmatter, sort_keys=False).strip()
        theme_lines = "\n".join(f"- {_format_term(term)}" for term in draft.topic_terms)
        source_lines = "\n".join(
            f"- [[{page.slug}|{page.title}]] (`{page.relative_path}`)"
            + (f" — {page.summary}" if page.summary else "")
            for page in draft.source_pages
        )
        page_text = (
            f"---\n{yaml_block}\n---\n\n"
            f"# {draft.title}\n\n"
            "## Overview\n\n"
            f"{draft.summary}\n\n"
            "## Key Themes\n\n"
            f"{theme_lines}\n\n"
            "## Source Pages\n\n"
            f"{source_lines}\n"
        )
        atomic_write_text(destination, page_text)
        return destination.relative_to(self.paths.root).as_posix()

    def _resolve_destination(self, slug: str, managed_pages: set[Path]) -> Path:
        destination = self.paths.wiki_concepts_dir / f"{slug}.md"
        if not destination.exists() or destination in managed_pages:
            return destination
        suffix = 2
        while True:
            candidate = self.paths.wiki_concepts_dir / f"{slug}-{suffix}.md"
            if not candidate.exists() or candidate in managed_pages:
                return candidate
            suffix += 1

    def _maintain_source_backlinks(
        self,
        source_pages: list[_SourcePage],
        drafts: list[_ConceptDraft],
    ) -> list[str]:
        backlinks: dict[str, list[tuple[str, str]]] = {
            page.relative_path: [] for page in source_pages
        }
        for draft in drafts:
            for page in draft.source_pages:
                backlinks.setdefault(page.relative_path, []).append(
                    (draft.slug, draft.title)
                )

        updated: list[str] = []
        for page in source_pages:
            current = page.file_path.read_text(encoding="utf-8")
            updated_text = _replace_backlinks_block(
                current,
                sorted(
                    backlinks.get(page.relative_path, []),
                    key=lambda item: item[1].casefold(),
                ),
            )
            if updated_text != current:
                atomic_write_text(page.file_path, updated_text)
                updated.append(page.relative_path)
        return updated


def _connected_components(
    source_pages: list[_SourcePage],
) -> list[list[_SourcePage]]:
    adjacency: dict[int, set[int]] = {
        index: set() for index in range(len(source_pages))
    }
    for index, left in enumerate(source_pages):
        for other_index in range(index + 1, len(source_pages)):
            right = source_pages[other_index]
            overlap = left.terms & right.terms
            if not overlap:
                continue
            union = left.terms | right.terms
            jaccard = len(overlap) / len(union) if union else 0.0
            if len(overlap) >= _MIN_SHARED_TERMS or jaccard >= _MIN_JACCARD:
                adjacency[index].add(other_index)
                adjacency[other_index].add(index)

    visited: set[int] = set()
    components: list[list[_SourcePage]] = []
    for index in range(len(source_pages)):
        if index in visited:
            continue
        stack = [index]
        component: list[_SourcePage] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(source_pages[current])
            stack.extend(
                neighbor for neighbor in adjacency[current] if neighbor not in visited
            )
        components.append(component)
    return components


def _derive_topic_terms(group: list[_SourcePage]) -> list[str]:
    title_text = " ".join(
        _normalize_phrase_text(f"{page.title} {page.summary}") for page in group
    )
    phrase_terms = [
        phrase.replace(" ", "-")
        for _, phrase in sorted(
            [
                (title_text.count(phrase), phrase)
                for phrase in _PHRASE_CANDIDATES
                if title_text.count(phrase) > 0
            ],
            key=lambda item: (-item[0], item[1]),
        )[:3]
    ]

    counts = Counter(term for page in group for term in page.terms)
    phrase_stems = {
        _stem_token(word)
        for phrase in phrase_terms
        for word in phrase.split("-")
        if len(word) >= 3
    }
    freq_terms = [
        term
        for term, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= 2 and term not in phrase_stems
    ][: 3 - len(phrase_terms)]

    combined = phrase_terms + freq_terms
    return combined[:3]


def _format_concept_title(topic_terms: list[str]) -> str:
    pretty = [_format_term(term) for term in topic_terms[:3]]
    if len(pretty) == 1:
        return pretty[0]
    if len(pretty) == 2:
        return f"{pretty[0]} and {pretty[1]}"
    return f"{pretty[0]}, {pretty[1]}, and {pretty[2]}"


def _format_concept_summary(group: list[_SourcePage], topic_terms: list[str]) -> str:
    theme_text = _format_concept_title(topic_terms).lower()
    return (
        f"Concept page generated from {len(group)} source pages covering {theme_text}."
    )


def _format_term(term: str) -> str:
    return term.replace("-", " ").title()


def _normalize_phrase_text(text: str) -> str:
    return " ".join(_WORD_PATTERN.findall(text.lower()))


def _extract_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in _WORD_PATTERN.findall(text.lower()):
        if token in _STOPWORDS or len(token) < 3:
            continue
        terms.add(_stem_token(token))
    return terms


def _stem_token(token: str) -> str:
    for suffix in ("ing", "ed", "es", "al", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def _split_frontmatter(contents: str) -> tuple[dict[str, object], str]:
    if not contents.startswith("---\n"):
        return {}, contents
    marker = contents.find("\n---\n", 4)
    if marker == -1:
        return {}, contents
    try:
        payload = yaml.safe_load(contents[4:marker]) or {}
    except yaml.YAMLError:
        return {}, contents
    frontmatter = payload if isinstance(payload, dict) else {}
    return frontmatter, contents[marker + 5 :]


def _replace_backlinks_block(contents: str, links: list[tuple[str, str]]) -> str:
    if links:
        block = (
            "\n## Related Concept Pages\n\n"
            f"{_MANAGED_START}\n"
            + "\n".join(f"- [[{slug}|{title}]]" for slug, title in links)
            + f"\n{_MANAGED_END}\n"
        )
        if _MANAGED_BLOCK_PATTERN.search(contents):
            return _MANAGED_BLOCK_PATTERN.sub(block, contents)
        normalized = contents.rstrip() + "\n"
        return normalized + block

    if _MANAGED_BLOCK_PATTERN.search(contents):
        updated = _MANAGED_BLOCK_PATTERN.sub("\n", contents).rstrip() + "\n"
        return updated
    return contents
