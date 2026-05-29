"""Data models for wiki models.

This module belongs to `graphwiki_kb.models.wiki_models` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LintIssue:
    """Represents lint issue behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    severity: str
    code: str
    path: str
    message: str


@dataclass
class LintReport:
    """Stores lint report data.

    Attributes:
        See annotated class attributes for stored values.
    """

    issues: list[LintIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        """Error count.

        Returns:
            int produced by the operation.
        """
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        """Warning count.

        Returns:
            int produced by the operation.
        """
        return sum(1 for issue in self.issues if issue.severity == "warning")

    @property
    def suggestion_count(self) -> int:
        """Suggestion count.

        Returns:
            int produced by the operation.
        """
        return sum(1 for issue in self.issues if issue.severity == "suggestion")


@dataclass
class SearchResult:
    """Stores search result data.

    Attributes:
        See annotated class attributes for stored values.
    """

    title: str
    path: str
    score: float
    snippet: str
    section: str = ""
    chunk_index: int | None = None
    retriever: str = ""

    @property
    def citation_ref(self) -> str:
        """Citation ref.

        Returns:
            str produced by the operation.
        """
        if self.chunk_index is None or self.chunk_index < 0:
            return self.path
        return f"{self.path}#chunk-{self.chunk_index}"


@dataclass
class StatusSnapshot:
    """Represents status snapshot behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    initialized: bool
    source_count: int
    compiled_source_count: int
    concept_page_count: int
    analysis_page_count: int
    last_compile_at: str | None
    provider_summary: str
    index_status: str
    export_status: str
    graph_status: dict[str, Any] = field(default_factory=dict)
    wikigraph_status: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewIssue:
    """Represents review issue behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    severity: str
    code: str
    pages: list[str]
    message: str


@dataclass
class ReviewReport:
    """Stores review report data.

    Attributes:
        See annotated class attributes for stored values.
    """

    issues: list[ReviewIssue] = field(default_factory=list)
    mode: str = ""

    @property
    def issue_count(self) -> int:
        """Issue count.

        Returns:
            int produced by the operation.
        """
        return len(self.issues)


@dataclass
class DiffEntry:
    """Represents diff entry behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    source_id: str
    slug: str
    title: str
    status: str  # "new", "changed", "missing", or "up_to_date"
    raw_path: str
    details: str = ""


@dataclass
class DiffReport:
    """Stores diff report data.

    Attributes:
        See annotated class attributes for stored values.
    """

    entries: list[DiffEntry] = field(default_factory=list)

    @property
    def new_count(self) -> int:
        """New count.

        Returns:
            int produced by the operation.
        """
        return sum(1 for e in self.entries if e.status == "new")

    @property
    def changed_count(self) -> int:
        """Changed count.

        Returns:
            int produced by the operation.
        """
        return sum(1 for e in self.entries if e.status == "changed")

    @property
    def missing_count(self) -> int:
        """Missing count.

        Returns:
            int produced by the operation.
        """
        return sum(1 for e in self.entries if e.status == "missing")

    @property
    def up_to_date_count(self) -> int:
        """Up to date count.

        Returns:
            int produced by the operation.
        """
        return sum(1 for e in self.entries if e.status == "up_to_date")
