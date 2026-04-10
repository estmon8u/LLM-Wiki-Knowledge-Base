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
    score: int
    snippet: str


@dataclass
class StatusSnapshot:
    initialized: bool
    source_count: int
    compiled_source_count: int
    concept_page_count: int
    last_compile_at: Optional[str]
