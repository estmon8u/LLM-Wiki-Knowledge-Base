"""Stage web-sourced recommendations as local files for ingestion.

This module belongs to `graphwiki_kb.services.web_source_acquisition_service`
and keeps related behavior close to the command, service, model, provider,
storage, script, or test surface that uses it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from graphwiki_kb.agents.models import SourceRecommendation
from graphwiki_kb.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    slugify,
)

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "graphwiki-kb-agent/0.1 (+https://github.com/estmon8u/LLM-Wiki-Knowledge-Base)"
)
_DEFAULT_TIMEOUT_SECONDS = 20
_MAX_BYTES = 10 * 1024 * 1024


class WebSourceAcquisitionError(RuntimeError):
    """Raised when a web source cannot be fetched or staged for ingestion."""


@dataclass(frozen=True)
class StagedSource:
    """Result of staging one source recommendation as a local file."""

    recommendation_id: int
    title: str
    url: str
    staged_path: Path
    suffix: str


HttpFetcher = Any  # callable: (url) -> tuple[bytes, str (content_type)]


class WebSourceAcquisitionService:
    """Fetches a recommendation URL and writes a durable local file."""

    def __init__(
        self,
        paths: ProjectPaths,
        *,
        http_fetcher: HttpFetcher | None = None,
        max_bytes: int = _MAX_BYTES,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.paths = paths
        self._http_fetcher = http_fetcher or _default_http_fetcher
        self.max_bytes = max_bytes
        self.timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------
    def stage_recommendation(
        self,
        recommendation: SourceRecommendation,
        *,
        run_id: str,
    ) -> StagedSource:
        """Fetch the recommendation URL and stage it as a local artifact."""
        if not recommendation.ingestable:
            raise WebSourceAcquisitionError(
                f"Recommendation {recommendation.id} is marked as not ingestable."
            )
        url = recommendation.url.strip()
        if not _is_http_url(url):
            raise WebSourceAcquisitionError(
                f"Recommendation {recommendation.id} has an unsupported URL: {url!r}"
            )
        body, content_type = self._fetch(url)
        suffix = _suffix_for(url, content_type, body)
        slug = slugify(recommendation.title or url)[:48] or "recommendation"
        staging_dir = self.paths.raw_dir / "web_staging" / _safe_run_slug(run_id)
        staging_dir.mkdir(parents=True, exist_ok=True)
        filename = f"rec-{recommendation.id:02d}-{slug}{suffix}"
        target = staging_dir / filename
        if suffix in {".md", ".html", ".htm", ".txt"}:
            target = staging_dir / f"rec-{recommendation.id:02d}-{slug}.md"
            atomic_write_text(target, _to_markdown(body, suffix, url, recommendation))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
        return StagedSource(
            recommendation_id=recommendation.id,
            title=recommendation.title,
            url=url,
            staged_path=target,
            suffix=suffix,
        )

    # ------------------------------------------------------------------
    def _fetch(self, url: str) -> tuple[bytes, str]:
        try:
            body, content_type = self._http_fetcher(
                url,
                timeout=self.timeout_seconds,
                max_bytes=self.max_bytes,
            )
        except Exception as exc:
            raise WebSourceAcquisitionError(f"Failed to fetch {url}: {exc}") from exc
        if not body:
            raise WebSourceAcquisitionError(f"Fetched empty body from {url}.")
        if len(body) > self.max_bytes:
            raise WebSourceAcquisitionError(
                f"Fetched body from {url} exceeds {self.max_bytes} bytes."
            )
        return body, content_type or ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _safe_run_slug(run_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", run_id)
    return cleaned[:80] or "run"


def _suffix_for(url: str, content_type: str, body: bytes) -> str:
    lowered = content_type.lower()
    if "pdf" in lowered or body[:5] == b"%PDF-":
        return ".pdf"
    if "html" in lowered or body.lstrip()[:1] == b"<":
        return ".html"
    if "markdown" in lowered:
        return ".md"
    if "text/plain" in lowered:
        return ".txt"
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in {".pdf", ".html", ".htm", ".md", ".txt"}:
        return suffix
    return ".html"


def _to_markdown(
    body: bytes,
    suffix: str,
    url: str,
    recommendation: SourceRecommendation,
) -> str:
    text = body.decode("utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        text = _strip_html(text)
    lines = [
        f"# {recommendation.title or url}",
        "",
        f"Source URL: {url}",
        f"Recommendation id: {recommendation.id}",
        f"Why added: {recommendation.why_add or '(no rationale provided)'}",
        "",
        "---",
        "",
        text.strip(),
        "",
    ]
    return "\n".join(lines)


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HTML_SCRIPT_RE = re.compile(
    r"<script.*?</script>|<style.*?</style>", re.DOTALL | re.IGNORECASE
)


def _strip_html(text: str) -> str:
    cleaned = _HTML_SCRIPT_RE.sub(" ", text)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _default_http_fetcher(
    url: str,
    *,
    timeout: int,
    max_bytes: int,
) -> tuple[bytes, str]:
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        body = response.read(max_bytes + 1)
    return body, content_type
