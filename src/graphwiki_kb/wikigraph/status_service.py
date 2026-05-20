"""WikiGraphRAG index status helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from graphwiki_kb.services.project_service import ProjectPaths
from graphwiki_kb.wikigraph.index_builder import load_built_index, wikigraph_output_dir


@dataclass(frozen=True)
class WikiGraphStatus:
    """Status summary for WikiGraphRAG artifacts."""

    built: bool
    output_dir: str
    built_at: str | None
    node_count: int
    edge_count: int
    chunk_count: int
    community_count: int
    lexical_backend: str | None
    include_graphrag_export_pages: bool | None
    latest_run: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "built": self.built,
            "output_dir": self.output_dir,
            "built_at": self.built_at,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "chunk_count": self.chunk_count,
            "community_count": self.community_count,
            "lexical_backend": self.lexical_backend,
            "include_graphrag_export_pages": self.include_graphrag_export_pages,
            "latest_run": self.latest_run,
        }


def wikigraph_status(paths: ProjectPaths) -> WikiGraphStatus:
    """Return WikiGraphRAG build status for a project."""
    output_dir = wikigraph_output_dir(paths)
    build = load_built_index(paths)
    latest_run = _latest_run_path(output_dir)
    if build is None:
        return WikiGraphStatus(
            built=False,
            output_dir=(
                paths.root.joinpath(output_dir).as_posix()
                if not output_dir.is_absolute()
                else output_dir.as_posix()
            ),
            built_at=None,
            node_count=0,
            edge_count=0,
            chunk_count=0,
            community_count=0,
            lexical_backend=None,
            include_graphrag_export_pages=None,
            latest_run=latest_run,
        )
    return WikiGraphStatus(
        built=True,
        output_dir=output_dir.relative_to(paths.root).as_posix(),
        built_at=build.snapshot.built_at,
        node_count=build.snapshot.node_count,
        edge_count=build.snapshot.edge_count,
        chunk_count=build.snapshot.chunk_count,
        community_count=build.snapshot.community_count,
        lexical_backend=build.snapshot.lexical_backend,
        include_graphrag_export_pages=build.snapshot.include_graphrag_export_pages,
        latest_run=latest_run,
    )


def _latest_run_path(output_dir: Path) -> str | None:
    latest = output_dir / "runs" / "latest.json"
    if latest.exists():
        return latest.relative_to(output_dir.parent.parent).as_posix()
    return None
