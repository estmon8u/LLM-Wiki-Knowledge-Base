"""Deterministic wiki-derived priors for optional GraphRAG prompt steering."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.services.config_service import (
    concept_generation_enabled,
    resolve_graph_config,
)
from graphwiki_kb.services.manifest_service import ManifestService
from graphwiki_kb.services.markdown_document import headings, parse_document, plain_text
from graphwiki_kb.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    utc_now_iso,
)

WIKI_PRIORS_SCHEMA_VERSION = 1

_PAREN_ALIAS_PATTERN = re.compile(
    r"\b(?P<term>[A-Za-z][A-Za-z0-9-]+(?:\s+[A-Za-z][A-Za-z0-9-]+){1,8})"
    r"\s+\((?P<alias>[A-Za-z][A-Za-z0-9-]{1,15})\)"
)
_GENERIC_HEADINGS = {
    "abstract",
    "analysis",
    "background",
    "conclusion",
    "discussion",
    "evidence",
    "introduction",
    "key excerpt",
    "key points",
    "methods",
    "open questions",
    "overview",
    "references",
    "results",
    "source details",
    "summary",
}


@dataclass(frozen=True)
class WikiPriorsResult:
    """Result of generating or previewing the deterministic priors artifact."""

    artifact_path: Path
    artifact: dict[str, Any] | None
    artifact_digest: str | None
    enabled: bool
    written: bool = False

    @property
    def glossary_count(self) -> int:
        if self.artifact is None:
            return 0
        glossary = self.artifact.get("glossary")
        return len(glossary) if isinstance(glossary, list) else 0

    @property
    def entity_type_count(self) -> int:
        if self.artifact is None:
            return 0
        entity_types = self.artifact.get("entity_types")
        return len(entity_types) if isinstance(entity_types, list) else 0


@dataclass
class _Candidate:
    term: str
    aliases: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _SourceText:
    source_id: str
    title: str
    text: str


class WikiPriorsService:
    """Builds ``graph/wiki_priors.json`` from source-grounded KB artifacts only."""

    def __init__(
        self,
        paths: ProjectPaths,
        manifest_service: ManifestService,
        *,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.paths = paths
        self.manifest_service = manifest_service
        self.config = config or {}
        self.artifact_path = paths.graph_dir / "wiki_priors.json"

    def sync(self, *, preview_only: bool = False) -> WikiPriorsResult:
        """Generate the priors artifact, writing it unless this is a preview."""
        graph_config = resolve_graph_config(self.config)
        priors_config = graph_config.wiki_priors
        if not priors_config.enabled:
            return WikiPriorsResult(
                artifact_path=self.artifact_path,
                artifact=None,
                artifact_digest=None,
                enabled=False,
            )

        artifact = self._build_artifact()
        artifact = self._preserve_generated_at_when_semantically_equal(artifact)
        artifact_digest = wiki_priors_behavior_digest(artifact)

        written = False
        if not preview_only:
            payload = json.dumps(artifact, indent=2, sort_keys=False) + "\n"
            existing = (
                self.artifact_path.read_text(encoding="utf-8")
                if self.artifact_path.exists()
                else None
            )
            if existing != payload:
                atomic_write_text(self.artifact_path, payload)
                written = True

        return WikiPriorsResult(
            artifact_path=self.artifact_path,
            artifact=artifact,
            artifact_digest=artifact_digest,
            enabled=True,
            written=written,
        )

    def _build_artifact(self) -> dict[str, Any]:
        graph_config = resolve_graph_config(self.config)
        priors_config = graph_config.wiki_priors
        sources = sorted(
            self.manifest_service.list_sources(),
            key=lambda source: (source.slug, source.source_id),
        )
        source_texts = self._source_texts(sources)
        entity_types = list(graph_config.entity_types[: priors_config.max_entity_types])
        glossary = self._glossary(
            source_texts,
            max_terms=priors_config.max_glossary_terms,
            min_support_count=priors_config.min_support_count,
        )
        return {
            "schema_version": WIKI_PRIORS_SCHEMA_VERSION,
            "source_digest": self._source_digest(sources),
            "wiki_compile_digest": self._wiki_compile_digest(),
            "generated_at": utc_now_iso(),
            "entity_types": entity_types,
            "glossary": glossary,
        }

    def _source_texts(self, sources: list[RawSourceRecord]) -> list[_SourceText]:
        source_texts: list[_SourceText] = []
        for source in sources:
            parts: list[str] = [source.title]
            source_page = self.paths.wiki_sources_dir / f"{source.slug}.md"
            if source_page.exists():
                page_text = source_page.read_text(encoding="utf-8")
                document = parse_document(page_text)
                parts.append(str(document.frontmatter.get("title") or ""))
                parts.append(str(document.frontmatter.get("summary") or ""))
                for heading in headings(page_text):
                    parts.append(heading.title)
                parts.append(plain_text(page_text))

            normalized_path = self._normalized_path(source)
            if normalized_path is not None and normalized_path.exists():
                parts.append(normalized_path.read_text(encoding="utf-8"))

            text = "\n".join(part for part in parts if part)
            source_texts.append(
                _SourceText(source_id=source.source_id, title=source.title, text=text)
            )

        if self._include_legacy_concepts():
            for page in sorted(self.paths.wiki_concepts_dir.glob("*.md")):
                text = page.read_text(encoding="utf-8")
                source_texts.append(
                    _SourceText(
                        source_id=f"concept:{page.stem}",
                        title=page.stem.replace("-", " ").title(),
                        text=text,
                    )
                )
        return source_texts

    def _normalized_path(self, source: RawSourceRecord) -> Path | None:
        if source.normalized_path is None:
            return None
        candidate = Path(source.normalized_path)
        if not candidate.is_absolute():
            candidate = self.paths.root / candidate
        try:
            resolved = candidate.resolve()
            normalized_root = self.paths.raw_normalized_dir.resolve()
        except OSError:
            return None
        if resolved == normalized_root or normalized_root in resolved.parents:
            return resolved
        return None

    def _glossary(
        self,
        source_texts: list[_SourceText],
        *,
        max_terms: int,
        min_support_count: int,
    ) -> list[dict[str, Any]]:
        candidates = self._candidate_terms(source_texts)
        pattern_cache: dict[str, re.Pattern[str]] = {}
        rows: list[dict[str, Any]] = []
        for candidate in candidates.values():
            support_count = 0
            source_ids: set[str] = set()
            aliases = sorted(candidate.aliases, key=str.lower)
            search_patterns = [
                _cached_term_pattern(pattern_cache, term)
                for term in (candidate.term, *aliases)
            ]
            for source in source_texts:
                matches = sum(
                    _term_pattern_count(source.text, pattern)
                    for pattern in search_patterns
                )
                if matches:
                    support_count += matches
                    source_ids.add(source.source_id)
            if support_count < min_support_count:
                continue
            rows.append(
                {
                    "term": candidate.term,
                    "aliases": aliases,
                    "support_count": support_count,
                    "source_ids": sorted(source_ids),
                }
            )

        rows.sort(
            key=lambda row: (
                -int(row["support_count"]),
                str(row["term"]).casefold(),
            )
        )
        return rows[:max_terms]

    def _candidate_terms(
        self, source_texts: list[_SourceText]
    ) -> dict[str, _Candidate]:
        candidates: dict[str, _Candidate] = {}
        for source in source_texts:
            self._add_candidate(candidates, source.title)
            for match in _PAREN_ALIAS_PATTERN.finditer(source.text):
                term = _clean_term(match.group("term"))
                alias = _clean_term(match.group("alias"))
                if not (_looks_like_term(term) and _looks_like_alias(alias)):
                    continue
                self._add_candidate(candidates, term, alias=alias)
            for line in headings(source.text):
                self._add_candidate(candidates, line.title)
        return candidates

    def _add_candidate(
        self,
        candidates: dict[str, _Candidate],
        term: str,
        *,
        alias: str | None = None,
    ) -> None:
        clean_term = _clean_term(term)
        if not _looks_like_term(clean_term):
            return
        key = _canonical_term_key(clean_term)
        candidate = candidates.setdefault(key, _Candidate(clean_term))
        if alias is not None:
            clean_alias = _clean_term(alias)
            if _looks_like_alias(clean_alias) and clean_alias.casefold() != key:
                candidate.aliases.add(clean_alias)

    def _source_digest(self, sources: list[RawSourceRecord]) -> str:
        payload: list[dict[str, Any]] = []
        for source in sources:
            normalized_path = self._normalized_path(source)
            normalized_digest = (
                _file_sha256(normalized_path)
                if normalized_path is not None and normalized_path.exists()
                else None
            )
            payload.append(
                {
                    "source": source.to_dict(),
                    "normalized_digest": normalized_digest,
                }
            )
        return _sha256_json(payload)

    def _wiki_compile_digest(self) -> str:
        digest = hashlib.sha256()
        for page in sorted(self.paths.wiki_sources_dir.glob("*.md")):
            _digest_file(digest, page, page.relative_to(self.paths.root).as_posix())
        if self._include_legacy_concepts():
            for page in sorted(self.paths.wiki_concepts_dir.glob("*.md")):
                _digest_file(digest, page, page.relative_to(self.paths.root).as_posix())
        return f"sha256:{digest.hexdigest()}"

    def _include_legacy_concepts(self) -> bool:
        graph_config = resolve_graph_config(self.config)
        return (
            graph_config.wiki_priors.include_legacy_concepts
            and concept_generation_enabled(self.config)
        )

    def _preserve_generated_at_when_semantically_equal(
        self, artifact: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.artifact_path.exists():
            return artifact
        try:
            existing = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return artifact
        if not isinstance(existing, dict):
            return artifact
        existing_semantic = _semantic_payload(existing)
        new_semantic = _semantic_payload(artifact)
        generated_at = existing.get("generated_at")
        if existing_semantic == new_semantic and isinstance(generated_at, str):
            artifact = dict(artifact)
            artifact["generated_at"] = generated_at
        return artifact


def wiki_priors_behavior_digest(artifact: dict[str, Any]) -> str:
    """Return a stable digest for the fields that can steer GraphRAG behavior."""
    return _sha256_json(_semantic_payload(artifact))


def _semantic_payload(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": artifact.get("schema_version"),
        "source_digest": artifact.get("source_digest"),
        "wiki_compile_digest": artifact.get("wiki_compile_digest"),
        "entity_types": artifact.get("entity_types") or [],
        "glossary": artifact.get("glossary") or [],
    }


def _clean_term(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().strip("`*_[](){}:;,.")).strip()


def _looks_like_term(value: str) -> bool:
    if not value or len(value) > 96:
        return False
    if value.casefold() in _GENERIC_HEADINGS:
        return False
    if not any(character.isalpha() for character in value):
        return False
    if value.startswith(("Source ID", "Raw file")):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", value)
    return 1 <= len(words) <= 10


def _looks_like_alias(value: str) -> bool:
    if not (2 <= len(value) <= 16):
        return False
    if not any(character.isupper() or character.isdigit() for character in value):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", value))


def _canonical_term_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _term_count(text: str, term: str) -> int:
    return _term_pattern_count(text, _term_pattern(term))


def _cached_term_pattern(
    cache: dict[str, re.Pattern[str]],
    term: str,
) -> re.Pattern[str]:
    key = term.casefold()
    pattern = cache.get(key)
    if pattern is None:
        pattern = _term_pattern(term)
        cache[key] = pattern
    return pattern


def _term_pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    return re.compile(
        rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )


def _term_pattern_count(text: str, pattern: re.Pattern[str]) -> int:
    return len(pattern.findall(text))


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _digest_file(digest: Any, path: Path, label: str) -> None:
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(b"\0")
