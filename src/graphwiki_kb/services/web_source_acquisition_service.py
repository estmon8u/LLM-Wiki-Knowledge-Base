"""Fetch web recommendations and stage durable local ingest artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from graphwiki_kb.agents.models import SourceRecommendation
from graphwiki_kb.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    slugify,
)

_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class StagedSource:
    """A recommendation staged for ingest."""

    recommendation_id: int
    title: str
    url: str
    staged_path: Path
    destination: str


class WebSourceAcquisitionError(RuntimeError):
    """Raised when a recommendation cannot be fetched or staged."""


class WebSourceAcquisitionService:
    """Download recommendation URLs into raw/web_staging for ingest."""

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def stage(
        self,
        recommendation: SourceRecommendation,
        *,
        run_id: str,
    ) -> StagedSource:
        """Fetch and stage one recommendation."""
        if not recommendation.ingestable:
            raise WebSourceAcquisitionError(
                f"Recommendation {recommendation.id} is not marked ingestable."
            )
        staging_root = self.paths.raw_dir / "web_staging" / run_id.replace("/", "-")
        staging_root.mkdir(parents=True, exist_ok=True)
        filename = f"rec-{recommendation.id}-{slugify(recommendation.title)}"
        url = recommendation.url
        if url.lower().endswith(".pdf"):
            dest = staging_root / f"{filename}.pdf"
            self._download_binary(url, dest)
        else:
            dest = staging_root / f"{filename}.md"
            body = self._download_text(url)
            markdown = self._html_to_markdown(body, recommendation)
            atomic_write_text(dest, markdown)
        relative = dest.relative_to(self.paths.root).as_posix()
        return StagedSource(
            recommendation_id=recommendation.id,
            title=recommendation.title,
            url=url,
            staged_path=dest,
            destination=relative,
        )

    def _download_text(self, url: str, *, timeout: float = 30.0) -> str:
        request = Request(url, headers={"User-Agent": "GraphWiki-KB-Agent/1.0"})
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except URLError as exc:
            raise WebSourceAcquisitionError(f"Failed to fetch {url}: {exc}") from exc
        charset = "utf-8"
        return raw.decode(charset, errors="replace")

    def _download_binary(self, url: str, dest: Path, *, timeout: float = 60.0) -> None:
        request = Request(url, headers={"User-Agent": "GraphWiki-KB-Agent/1.0"})
        try:
            with urlopen(request, timeout=timeout) as response:
                data = response.read()
        except URLError as exc:
            raise WebSourceAcquisitionError(f"Failed to fetch {url}: {exc}") from exc
        dest.write_bytes(data)

    @staticmethod
    def _html_to_markdown(body: str, recommendation: SourceRecommendation) -> str:
        text = _HTML_TAG_PATTERN.sub(" ", body)
        text = re.sub(r"\s+", " ", text).strip()
        return (
            f"# {recommendation.title}\n\n"
            f"Source URL: {recommendation.url}\n\n"
            f"{text[:50000]}\n"
        )
