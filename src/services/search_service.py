from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re

import yaml

from src.models.wiki_models import SearchResult
from src.services.project_service import ProjectPaths
from src.storage.search_index_store import (
    IndexedChunk,
    IndexedFileState,
    SearchIndexStore,
    SearchIndexUnavailable,
)

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_CHUNK_CHAR_LIMIT = 1200
_MAINTENANCE_PAGE_NAMES: frozenset[str] = frozenset({"wiki/index.md", "wiki/log.md"})
_INDEXABLE_FM_KEYS: frozenset[str] = frozenset(
    {"title", "summary", "tags", "aliases", "source_title", "description", "keywords"}
)


@dataclass(frozen=True)
class _SectionChunk:
    section: str
    body: str


class SearchService:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self.index_store = SearchIndexStore(
            self.paths.graph_exports_dir / "search_index.sqlite3"
        )
        self._fts_available = True

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        terms = _query_terms(query)
        if not terms:
            return []

        if self._fts_available:
            self.refresh()
            if self._fts_available:
                return self._search_index(terms, limit=limit)

        return self._scan_markdown_files(terms, limit=limit)

    def refresh(self, *, force: bool = False) -> bool:
        if not self._fts_available:
            return False

        inventory = self._wiki_inventory()
        try:
            indexed_files = self.index_store.load_indexed_files()
            version_ok = self.index_store.check_version()
            if not force and version_ok and indexed_files == inventory:
                return False

            chunks: list[IndexedChunk] = []
            for file_path in sorted(self.paths.wiki_dir.rglob("*.md")):
                relative_path = file_path.relative_to(self.paths.root).as_posix()
                chunks.extend(self._indexable_chunks(file_path, relative_path))

            file_states = [
                IndexedFileState(
                    page_path=page_path,
                    mtime_ns=mtime_ns,
                    size_bytes=size_bytes,
                )
                for page_path, (mtime_ns, size_bytes) in sorted(inventory.items())
            ]
            self.index_store.rebuild(file_states, chunks)
            return True
        except SearchIndexUnavailable as exc:
            logger.warning("SQLite FTS5 search index unavailable: %s", exc)
            self._fts_available = False
            return False

    def refresh_file(self, file_path: Path) -> None:
        """Insert or update the index for a single file. No-op when FTS is unavailable."""
        if not self._fts_available:
            return
        try:
            if not self.index_store.check_version():
                self.refresh(force=True)
                return
            relative_path = file_path.relative_to(self.paths.root).as_posix()
            stat = file_path.stat()
            file_state = IndexedFileState(
                page_path=relative_path,
                mtime_ns=stat.st_mtime_ns,
                size_bytes=stat.st_size,
            )
            chunks = self._indexable_chunks(file_path, relative_path)
            self.index_store.upsert_file(file_state, chunks)
        except SearchIndexUnavailable as exc:
            logger.warning("Search index file refresh failed: %s", exc)
            self._fts_available = False
        except OSError as exc:
            logger.warning("Search index file refresh skipped: %s", exc)

    def _search_index(self, terms: list[str], *, limit: int) -> list[SearchResult]:
        try:
            hits = self.index_store.search(
                _build_match_query(terms),
                limit=max(limit * 30, 50),
            )
        except SearchIndexUnavailable as exc:
            logger.warning("SQLite FTS5 search query unavailable: %s", exc)
            self._fts_available = False
            return self._scan_markdown_files(terms, limit=limit)

        results: list[SearchResult] = []
        seen_paths: set[str] = set()
        for hit in hits:
            if hit.page_path in seen_paths:
                continue
            seen_paths.add(hit.page_path)
            snippet = " ".join(hit.snippet.split())
            if not snippet:
                snippet = hit.section or hit.title
            results.append(
                SearchResult(
                    title=hit.title,
                    path=hit.page_path,
                    score=hit.score,
                    snippet=snippet,
                    section=hit.section,
                    chunk_index=hit.chunk_index,
                )
            )
            if len(results) >= limit:
                break

        return results

    def _scan_markdown_files(
        self, terms: list[str], *, limit: int
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        for file_path in sorted(self.paths.wiki_dir.rglob("*.md")):
            relative_path = file_path.relative_to(self.paths.root).as_posix()
            if _is_maintenance_page(relative_path):
                continue
            if _is_generated_concept_page(file_path, self.paths):
                continue
            text = file_path.read_text(encoding="utf-8")
            normalized = text.lower()
            score = sum(normalized.count(term) for term in terms)
            if score <= 0:
                continue
            body = _strip_frontmatter(text)
            snippet = _extract_snippet(body, terms)
            results.append(
                SearchResult(
                    title=file_path.stem.replace("-", " ").title(),
                    path=relative_path,
                    score=score,
                    snippet=snippet,
                    section="",
                    chunk_index=None,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def _wiki_inventory(self) -> dict[str, tuple[int, int]]:
        inventory: dict[str, tuple[int, int]] = {}
        if not self.paths.wiki_dir.exists():
            return inventory
        for file_path in self.paths.wiki_dir.rglob("*.md"):
            stat = file_path.stat()
            inventory[file_path.relative_to(self.paths.root).as_posix()] = (
                stat.st_mtime_ns,
                stat.st_size,
            )
        return inventory

    def _indexable_chunks(
        self, file_path: Path, relative_path: str
    ) -> list[IndexedChunk]:
        if _is_generated_concept_page(file_path, self.paths):
            return []

        if _is_maintenance_page(relative_path):
            return []

        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError:
            return []

        frontmatter = _extract_frontmatter(text)
        title = _page_title(file_path, text, frontmatter)
        page_type = _frontmatter_value(frontmatter, "type")
        metadata = _frontmatter_search_text(frontmatter)
        chunks = _chunk_markdown_body(text, title)
        if not chunks:
            chunks = [_SectionChunk(section=title, body=metadata or title)]

        return [
            IndexedChunk(
                page_path=relative_path,
                page_type=page_type,
                title=title,
                section=chunk.section,
                chunk_index=index,
                metadata=metadata,
                body=chunk.body,
            )
            for index, chunk in enumerate(chunks)
        ]


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    marker = text.find("\n---\n", 4)
    if marker == -1:
        return text
    return text[marker + 5 :]


def _extract_frontmatter_type(text: str) -> str:
    """Return the ``type`` value from YAML frontmatter, or empty string."""
    if not text.startswith("---\n"):
        return ""
    marker = text.find("\n---\n", 4)
    if marker == -1:
        return ""
    fm_block = text[4:marker]
    for line in fm_block.splitlines():
        if line.startswith("type:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return ""


def _is_generated_concept_page(file_path: Path, paths: ProjectPaths) -> bool:
    """Skip generated concept pages but include analysis (saved query) pages."""
    if not paths.wiki_concepts_dir.exists():
        return False
    if file_path == paths.wiki_concepts_dir:
        return False
    if paths.wiki_concepts_dir not in file_path.parents:
        return False
    # Inside concepts dir — check the page type
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError:
        return True
    page_type = _extract_frontmatter_type(text)
    # analysis pages are saved query answers and should be searchable
    if page_type == "analysis":
        return False
    # concept pages (generated) are skipped
    return True


def _extract_frontmatter(text: str) -> dict[str, object]:
    if not text.startswith("---\n"):
        return {}
    marker = text.find("\n---\n", 4)
    if marker == -1:
        return {}
    try:
        payload = yaml.safe_load(text[4:marker]) or {}
    except yaml.YAMLError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _frontmatter_value(frontmatter: dict[str, object], key: str) -> str:
    value = frontmatter.get(key)
    return value.strip() if isinstance(value, str) else ""


def _page_title(file_path: Path, text: str, frontmatter: dict[str, object]) -> str:
    frontmatter_title = _frontmatter_value(frontmatter, "title")
    if frontmatter_title:
        return frontmatter_title

    body = _strip_frontmatter(text)
    for line in body.splitlines():
        stripped = line.strip()
        if _is_heading_line(stripped):
            return stripped.lstrip("#").strip().rstrip("#").strip()

    return file_path.stem.replace("-", " ").title()


def _frontmatter_text(frontmatter: dict[str, object]) -> str:
    values: list[str] = []

    def append_value(value: object) -> None:
        if isinstance(value, str):
            if value.strip():
                values.append(value.strip())
            return
        if isinstance(value, (int, float, bool)):
            values.append(str(value))
            return
        if isinstance(value, list):
            for item in value:
                append_value(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                append_value(item)

    for item in frontmatter.values():
        append_value(item)
    return "\n".join(values)


def _frontmatter_search_text(frontmatter: dict[str, object]) -> str:
    """Return searchable text from selected semantic frontmatter fields only.

    Excludes raw paths, hashes, timestamps, and provider metadata so they
    cannot distort relevance ranking.
    """
    selected = {k: v for k, v in frontmatter.items() if k in _INDEXABLE_FM_KEYS}
    return _frontmatter_text(selected)


def _is_maintenance_page(relative_path: str) -> bool:
    """Return True for wiki maintenance pages that should not be indexed."""
    return relative_path in _MAINTENANCE_PAGE_NAMES


def _chunk_markdown_body(text: str, title: str) -> list[_SectionChunk]:
    body = _strip_frontmatter(text)
    if not body.strip():
        return []

    section_bodies: list[tuple[str, str]] = []
    current_section = title
    current_lines: list[str] = []

    def flush_section() -> None:
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if content:
            section_bodies.append((current_section, content))
        current_lines = []

    for line in body.splitlines():
        stripped = line.strip()
        if _is_heading_line(stripped):
            flush_section()
            current_section = stripped.lstrip("#").strip().rstrip("#").strip() or title
            continue
        current_lines.append(line)
    flush_section()

    chunks: list[_SectionChunk] = []
    for section, section_text in section_bodies or [(title, body)]:
        paragraphs = _paragraphs(section_text)
        if not paragraphs and section_text.strip():
            paragraphs = [" ".join(section_text.split())]

        current_parts: list[str] = []
        current_length = 0
        for paragraph in paragraphs:
            normalized = " ".join(paragraph.split()).strip()
            if not normalized:
                continue
            addition = len(normalized) + 2
            if current_parts and current_length + addition > _CHUNK_CHAR_LIMIT:
                chunks.append(
                    _SectionChunk(section=section, body="\n\n".join(current_parts))
                )
                current_parts = []
                current_length = 0
            current_parts.append(normalized)
            current_length += addition

        if current_parts:
            chunks.append(
                _SectionChunk(section=section, body="\n\n".join(current_parts))
            )

    return chunks


def _paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            paragraphs.append(" ".join(current).strip())
            current = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if _is_heading_line(stripped):
            flush()
            continue
        current.append(stripped)

    flush()
    return paragraphs


def _is_heading_line(line: str) -> bool:
    return bool(re.match(r"^#{1,6}\s+\S", line))


def _query_terms(query: str) -> list[str]:
    return _TOKEN_PATTERN.findall(query.lower())


def _build_match_query(terms: list[str]) -> str:
    return " OR ".join(f'"{term}"' for term in terms)


def _extract_snippet(text: str, terms: list[str]) -> str:
    lowered = text.lower()
    first_position = min(
        (lowered.find(term) for term in terms if lowered.find(term) != -1), default=0
    )
    start = max(0, first_position - 80)
    end = min(len(text), first_position + 220)
    snippet = " ".join(text[start:end].split())
    return snippet or text[:220].strip()
