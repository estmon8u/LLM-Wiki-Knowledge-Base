from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class RawSourceRecord:
    source_id: str
    slug: str
    title: str
    origin: str
    source_type: str
    raw_path: str
    content_hash: str
    ingested_at: str
    compiled_at: Optional[str] = None
    compiled_from_hash: Optional[str] = None
    normalized_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RawSourceRecord":
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
            metadata=payload.get("metadata", {}),
        )
