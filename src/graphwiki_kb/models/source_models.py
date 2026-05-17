"""Data models for source models.

This module belongs to `graphwiki_kb.models.source_models` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RawSourceRecord:
    """Stores raw source record data.

    Attributes:
        See annotated class attributes for stored values.
    """

    source_id: str
    slug: str
    title: str
    origin: str
    source_type: str
    raw_path: str
    content_hash: str
    ingested_at: str
    compiled_at: str | None = None
    compiled_from_hash: str | None = None
    normalized_path: str | None = None
    origin_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes this value to a dictionary.

        Returns:
            dict[str, Any] produced by the operation.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RawSourceRecord:
        """Builds an instance from a dictionary payload.

        Args:
            payload: Structured payload being parsed or serialized.

        Returns:
            "RawSourceRecord" produced by the operation.
        """
        return cls(
            source_id=payload["source_id"],
            slug=payload["slug"],
            title=payload["title"],
            origin=payload["origin"],
            source_type=payload["source_type"],
            raw_path=payload["raw_path"],
            normalized_path=payload.get("normalized_path"),
            content_hash=payload["content_hash"],
            ingested_at=payload["ingested_at"],
            compiled_at=payload.get("compiled_at"),
            compiled_from_hash=payload.get("compiled_from_hash"),
            origin_hash=payload.get("origin_hash"),
            metadata=payload.get("metadata", {}),
        )
