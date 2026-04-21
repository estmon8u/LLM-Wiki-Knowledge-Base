"""LangSmith tracing wrapper for TextProvider instances."""

from __future__ import annotations

import time
from typing import Optional

from langsmith import traceable

from src.providers.base import ProviderRequest, ProviderResponse, TextProvider


class LangSmithTracingProvider(TextProvider):
    """Thin decorator that traces provider.generate() calls via LangSmith.

    Delegates all generation to the wrapped provider and emits a trace span
    for each call.  The wrapper is intentionally minimal — its job is
    delegation and trace capture, not prompt mutation.
    """

    def __init__(
        self,
        wrapped: TextProvider,
        *,
        task: str,
        project_name: Optional[str] = None,
    ) -> None:
        self.wrapped = wrapped
        self.task = task
        self.name = getattr(wrapped, "name", "unknown")
        self.project_name = project_name

    def generate(self, request: ProviderRequest) -> ProviderResponse:
        return self._generate_traced(request)

    @traceable(name="provider.generate", run_type="llm")
    def _generate_traced(self, request: ProviderRequest) -> ProviderResponse:
        started = time.perf_counter()
        response = self.wrapped.generate(request)
        latency_ms = int((time.perf_counter() - started) * 1000)
        # Metadata is attached to the LangSmith span automatically via
        # the @traceable decorator context; the return value is what
        # matters for the caller.
        return response
