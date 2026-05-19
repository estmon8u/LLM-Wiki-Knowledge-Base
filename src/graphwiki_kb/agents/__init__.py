"""OpenAI Agents SDK orchestration layer for the kb agent command.

This package owns agent turns, tool selection, approvals, session history,
and tracing. Existing services remain responsible for GraphRAG readiness,
file ingest, manifest updates, source normalization, graph updates, wiki
artifact writing, and validation.
"""

from __future__ import annotations
