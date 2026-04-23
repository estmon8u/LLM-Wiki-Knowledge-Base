from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import re

from nltk.collocations import (
    BigramAssocMeasures,
    BigramCollocationFinder,
    TrigramAssocMeasures,
    TrigramCollocationFinder,
)
from nltk.stem import SnowballStemmer
from pydantic import BaseModel, Field, ValidationError
import yaml

from src.providers.base import ProviderRequest, TextProvider
from src.services.markdown_document import parse_document
from src.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    slugify,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

_SNOWBALL = SnowballStemmer("english")
_BIGRAM_MEASURES = BigramAssocMeasures()
_TRIGRAM_MEASURES = TrigramAssocMeasures()


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
        "question",
        "questions",
        "answer",
        "answers",
        "language",
        "languages",
        "knowledge",
        "generation",
        "augmented",
        "domain",
        "learn",
        "learning",
        "open",
        "paper",
        "papers",
        "approach",
        "approaches",
        "method",
        "methods",
        "model",
        "models",
        "available",
        "abstract",
        "introduction",
        "section",
        "sections",
        "figure",
        "figures",
        "table",
        "tables",
        "result",
        "system",
        "systems",
        "results",
        "tasks",
        "task",
        "based",
        "performance",
        "perform",
        "different",
        "various",
        "several",
        "show",
        "shown",
        "compared",
        "existing",
        "previous",
        "novel",
        "first",
        "new",
        "high",
        "large",
        "train",
        "training",
        "set",
        "given",
        "one",
        "two",
        "three",
        "capstone",
        "canonical",
        "canonic",
    }
)
_GENERIC_PHRASES = frozenset(
    {
        "question answering",
        "language model",
        "language models",
        "knowledge intensive",
        "open domain question answering",
        "retrieval augmented",
    }
)
_MIN_SHARED_TERMS = 2
_MIN_JACCARD = 0.18
_MIN_SOURCE_PAGES = 3
_CONCEPT_CACHE_VERSION = 1
_PLACEHOLDER_SUMMARIES = {
    "no summary available yet.",
    "summary unavailable.",
    "no content available for summarization.",
}

_CONCEPT_SYSTEM_PROMPT = (
    "You cluster a small Markdown knowledge base into durable concept pages. "
    "Return only JSON matching the provided schema. Create 0 to 5 clusters. "
    "Each cluster must be supported by at least three listed source pages. "
    "Use only source page paths from the prompt. Avoid broad generic topics, "
    "method words, and clusters that are only weakly related. Use concise "
    "human-readable titles and 1 to 3 kebab-case topic_terms."
)

_CONCEPT_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "topic_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "source_pages": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["title", "summary", "topic_terms", "source_pages"],
            },
        }
    },
    "required": ["concepts"],
}


class _ProviderConcept(BaseModel):
    title: str = Field(min_length=1)
    summary: str = Field(default="")
    topic_terms: list[str] = Field(default_factory=list)
    source_pages: list[str] = Field(default_factory=list)


class _ProviderConceptReport(BaseModel):
    concepts: list[_ProviderConcept] = Field(default_factory=list)


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
    def __init__(
        self,
        paths: ProjectPaths,
        *,
        provider: TextProvider | None = None,
    ) -> None:
        self.paths = paths
        self.provider = provider

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
            if summary.casefold() in _PLACEHOLDER_SUMMARIES:
                summary = ""
            terms = _extract_terms(f"{title}\n{summary}")
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
        if len(source_pages) < _MIN_SOURCE_PAGES:
            return []

        if self.provider is not None:
            provider_drafts = self._provider_concept_drafts(source_pages)
            if provider_drafts is not None:
                return provider_drafts

        return self._deterministic_concept_drafts(source_pages)

    def _provider_concept_drafts(
        self,
        source_pages: list[_SourcePage],
    ) -> list[_ConceptDraft] | None:
        source_digest = _source_pages_digest(source_pages)
        cached = _load_concept_cache(self._concept_cache_path(), source_digest)
        if cached is not None:
            return _drafts_from_provider_report(cached, source_pages)

        prompt = _provider_concept_prompt(source_pages)
        try:
            response = self.provider.generate(
                ProviderRequest(
                    prompt=prompt,
                    system_prompt=_CONCEPT_SYSTEM_PROMPT,
                    max_tokens=2048,
                    response_schema=_CONCEPT_RESPONSE_SCHEMA,
                    response_schema_name="kb_concept_clusters",
                )
            )
            report = _parse_provider_concept_report(response.text)
            _write_concept_cache(
                self._concept_cache_path(),
                source_digest,
                report,
            )
            return _drafts_from_provider_report(report, source_pages)
        except Exception as exc:
            logger.warning(
                "Provider concept clustering failed; using deterministic fallback: %s",
                exc,
            )
            return None

    def _concept_cache_path(self) -> Path:
        return self.paths.graph_exports_dir / "concept_clusters.json"

    def _deterministic_concept_drafts(
        self,
        source_pages: list[_SourcePage],
    ) -> list[_ConceptDraft]:
        groups = _connected_components(source_pages)
        drafts: list[_ConceptDraft] = []
        for group in groups:
            if len(group) < _MIN_SOURCE_PAGES:
                continue
            topic_terms = _derive_topic_terms(group)
            if not topic_terms:
                continue
            if len(topic_terms) < 2 and "-" not in topic_terms[0]:
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


def _provider_concept_prompt(source_pages: list[_SourcePage]) -> str:
    entries: list[str] = ["## Source Pages"]
    for page in source_pages:
        entries.extend(
            [
                f"### {page.relative_path}",
                f"Title: {page.title}",
                f"Summary: {page.summary or '(no summary)'}",
                "",
            ]
        )
    return "\n".join(entries).strip()


def _parse_provider_concept_report(raw: str) -> _ProviderConceptReport:
    payload = json.loads(raw.strip())
    return _ProviderConceptReport.model_validate(payload)


def _drafts_from_provider_report(
    report: _ProviderConceptReport,
    source_pages: list[_SourcePage],
) -> list[_ConceptDraft]:
    pages_by_key: dict[str, _SourcePage] = {}
    for page in source_pages:
        pages_by_key[page.relative_path] = page
        pages_by_key[page.slug] = page
        pages_by_key[f"{page.slug}.md"] = page

    drafts: list[_ConceptDraft] = []
    seen_slugs: set[str] = set()
    for concept in report.concepts:
        group: list[_SourcePage] = []
        seen_pages: set[str] = set()
        for raw_page in concept.source_pages:
            page_key = raw_page.strip()
            page = pages_by_key.get(page_key)
            if page is None or page.relative_path in seen_pages:
                continue
            seen_pages.add(page.relative_path)
            group.append(page)

        if len(group) < _MIN_SOURCE_PAGES:
            continue

        topic_terms = _normalize_topic_terms(concept.topic_terms)
        if not topic_terms:
            topic_terms = _derive_topic_terms(group)
        if not topic_terms:
            continue

        title = concept.title.strip() or _format_concept_title(topic_terms)
        summary = concept.summary.strip() or _format_concept_summary(group, topic_terms)
        slug = slugify("-".join(topic_terms[:3]) or title)
        if not slug or slug in seen_slugs:
            continue
        seen_slugs.add(slug)
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


def _normalize_topic_terms(raw_terms: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_term in raw_terms:
        term = slugify(str(raw_term))
        if not term or term in seen:
            continue
        seen.add(term)
        normalized.append(term)
        if len(normalized) == 3:
            break
    return normalized


def _source_pages_digest(source_pages: list[_SourcePage]) -> str:
    payload = [
        {
            "path": page.relative_path,
            "title": page.title,
            "summary": page.summary,
            "terms": sorted(page.terms),
        }
        for page in sorted(source_pages, key=lambda item: item.relative_path)
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_concept_cache(
    cache_path: Path,
    source_digest: str,
) -> _ProviderConceptReport | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("version") != _CONCEPT_CACHE_VERSION:
        return None
    if payload.get("source_digest") != source_digest:
        return None
    try:
        return _ProviderConceptReport.model_validate(payload.get("report"))
    except ValidationError:
        return None


def _write_concept_cache(
    cache_path: Path,
    source_digest: str,
    report: _ProviderConceptReport,
) -> None:
    payload = {
        "version": _CONCEPT_CACHE_VERSION,
        "source_digest": source_digest,
        "report": report.model_dump(mode="python"),
    }
    atomic_write_text(cache_path, json.dumps(payload, indent=2, sort_keys=True))


def _derive_topic_terms(group: list[_SourcePage]) -> list[str]:
    page_tokens = [
        _candidate_phrase_tokens(f"{page.title} {page.summary}") for page in group
    ]
    phrase_terms = _collocation_topic_terms(page_tokens)

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
        if count >= _MIN_SOURCE_PAGES
        and count < len(group)
        and term not in phrase_stems
    ][: 3 - len(phrase_terms)]

    if not phrase_terms:
        return []
    combined = phrase_terms + freq_terms
    return combined[:3]


def _candidate_phrase_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in _WORD_PATTERN.findall(text.lower().replace("-", " ")):
        if token in _STOPWORDS or len(token) < 3:
            continue
        if _stem_token(token) in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _collocation_topic_terms(page_tokens: list[list[str]]) -> list[str]:
    support: Counter[str] = Counter()
    all_tokens: list[str] = []

    for tokens in page_tokens:
        all_tokens.extend(tokens)
        seen_phrases: set[str] = set()
        for size in (2, 3):
            for gram in zip(*(tokens[index:] for index in range(size))):
                if len(set(gram)) != len(gram):
                    continue
                phrase = " ".join(gram)
                if _is_generic_phrase(phrase):
                    continue
                seen_phrases.add(phrase)
        support.update(seen_phrases)

    ranked: list[tuple[int, float, str]] = []
    if len(all_tokens) >= 2:
        bigrams = BigramCollocationFinder.from_words(all_tokens)
        bigrams.apply_freq_filter(2)
        for gram, score in bigrams.score_ngrams(_BIGRAM_MEASURES.likelihood_ratio):
            phrase = " ".join(gram)
            if support[phrase] >= 2 and not _is_generic_phrase(phrase):
                ranked.append((support[phrase], score, phrase))

    if len(all_tokens) >= 3:
        trigrams = TrigramCollocationFinder.from_words(all_tokens)
        trigrams.apply_freq_filter(2)
        for gram, score in trigrams.score_ngrams(_TRIGRAM_MEASURES.likelihood_ratio):
            phrase = " ".join(gram)
            if support[phrase] >= 2 and not _is_generic_phrase(phrase):
                ranked.append((support[phrase], score, phrase))

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    terms: list[str] = []
    seen_terms: set[str] = set()
    for _, _, phrase in ranked:
        term = phrase.replace(" ", "-")
        if term in seen_terms:
            continue
        seen_terms.add(term)
        terms.append(term)
        if len(terms) == 3:
            break
    return terms


def _is_generic_phrase(phrase: str) -> bool:
    normalized = " ".join(phrase.split())
    if normalized in _GENERIC_PHRASES:
        return True
    return all(token in _STOPWORDS for token in normalized.split())


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
        stemmed = _stem_token(token)
        if stemmed in _STOPWORDS:
            continue
        terms.add(stemmed)
    return terms


def _stem_token(token: str) -> str:
    stemmed = _SNOWBALL.stem(token)
    if len(stemmed) < 3:
        return token
    return stemmed


def _split_frontmatter(contents: str) -> tuple[dict[str, object], str]:
    document = parse_document(contents)
    if not document.has_frontmatter or not document.valid_frontmatter:
        return {}, contents
    return document.frontmatter, document.body


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
