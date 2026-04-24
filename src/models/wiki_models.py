from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LintIssue:
    severity: str
    code: str
    path: str
    message: str


@dataclass
class LintReport:
    issues: list[LintIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    @property
    def suggestion_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "suggestion")


@dataclass
class SearchResult:
    title: str
    path: str
    score: float
    snippet: str
    section: str = ""
    chunk_index: Optional[int] = None

    @property
    def citation_ref(self) -> str:
        if self.chunk_index is None or self.chunk_index < 0:
            return self.path
        return f"{self.path}#chunk-{self.chunk_index}"


@dataclass
class StatusSnapshot:
    initialized: bool
    source_count: int
    compiled_source_count: int
    concept_page_count: int
    analysis_page_count: int
    last_compile_at: Optional[str]
    provider_summary: str
    index_status: str
    export_status: str


@dataclass
class ReviewIssue:
    severity: str
    code: str
    pages: list[str]
    message: str


@dataclass
class ReviewReport:
    issues: list[ReviewIssue] = field(default_factory=list)
    mode: str = ""

    @property
    def issue_count(self) -> int:
        return len(self.issues)


@dataclass
class DiffEntry:
    source_id: str
    slug: str
    title: str
    status: str  # "new", "changed", or "up_to_date"
    raw_path: str
    details: str = ""


@dataclass
class DiffReport:
    entries: list[DiffEntry] = field(default_factory=list)

    @property
    def new_count(self) -> int:
        return sum(1 for e in self.entries if e.status == "new")

    @property
    def changed_count(self) -> int:
        return sum(1 for e in self.entries if e.status == "changed")

    @property
    def up_to_date_count(self) -> int:
        return sum(1 for e in self.entries if e.status == "up_to_date")
