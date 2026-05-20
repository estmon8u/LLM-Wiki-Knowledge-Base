"""Service wrapper for building and exporting WikiGraphRAG indexes."""

from __future__ import annotations

from pathlib import Path

from graphwiki_kb.services.config_service import (
    WikiGraphRuntimeConfig,
    resolve_wikigraph_config,
)
from graphwiki_kb.services.project_service import (
    ProjectPaths,
    atomic_write_text,
    utc_now_iso,
)
from graphwiki_kb.wikigraph.deps import require_networkx
from graphwiki_kb.wikigraph.index_builder import (
    WikiGraphBuildResult,
    build_wikigraph_index,
    load_built_index,
    wiki_generated_dir,
)
from graphwiki_kb.wikigraph.models import WikiGraphNode


class WikiGraphIndexService:
    """Build, load, and export WikiGraphRAG indexes."""

    def __init__(self, paths: ProjectPaths, config: dict) -> None:
        self.paths = paths
        self.config = config

    def runtime(self) -> WikiGraphRuntimeConfig:
        return resolve_wikigraph_config(self.config)

    def build(
        self,
        *,
        include_graphrag_export_pages: bool | None = None,
    ) -> WikiGraphBuildResult:
        require_networkx()
        runtime = self.runtime()
        return build_wikigraph_index(
            self.paths,
            include_graphrag_export_pages=(
                include_graphrag_export_pages
                if include_graphrag_export_pages is not None
                else runtime.include_graphrag_export_pages
            ),
            lexical_backend=runtime.lexical_backend,
            community_algorithm=runtime.community_algorithm,
        )

    def load(self) -> WikiGraphBuildResult | None:
        return load_built_index(self.paths)

    def export_artifacts(self) -> list[str]:
        """Write generated wiki/wikigraph artifact pages."""
        build = self.load()
        if build is None:
            raise FileNotFoundError(
                "WikiGraphRAG index is not built. Run `kb update` first."
            )
        created: list[str] = []
        base = wiki_generated_dir(self.paths)
        for subdir in (
            "entities",
            "claims",
            "relationships",
            "communities",
            "chunks",
            "evidence",
        ):
            (base / subdir).mkdir(parents=True, exist_ok=True)
        for node in build.nodes:
            if node.kind == "entity":
                created.append(self._write_entity_card(base, node))
            elif node.kind == "community":
                created.append(self._write_community_card(base, node))
            elif node.kind == "chunk":
                created.append(self._write_chunk_card(base, node))
        return [path for path in created if path]

    def _write_entity_card(self, base: Path, node: WikiGraphNode) -> str:
        rel = f"wiki/wikigraph/entities/{_slug_file(node.title)}.md"
        path = self.paths.root / rel
        derived = node.metadata.get("origins", [])
        body = (
            "---\n"
            "type: wikigraph_entity\n"
            "generated: true\n"
            "retrieval_backend: wikigraph\n"
            f'title: "{node.title}"\n'
            f"confidence: medium\n"
            f'generated_at: "{utc_now_iso()}"\n'
            "derived_from:\n"
            + "".join(f"  - {item}\n" for item in derived[:8])
            + "---\n\n"
            f"# {node.title}\n\n"
            f"{node.text}\n"
        )
        atomic_write_text(path, body)
        return rel

    def _write_community_card(self, base: Path, node: WikiGraphNode) -> str:
        rel = f"wiki/wikigraph/communities/{_slug_file(node.title)}.md"
        path = self.paths.root / rel
        body = (
            "---\n"
            "type: wikigraph_community\n"
            "generated: true\n"
            "retrieval_backend: wikigraph\n"
            f'title: "{node.title}"\n'
            f'generated_at: "{utc_now_iso()}"\n'
            "---\n\n"
            f"# {node.title}\n\n"
            f"{node.text}\n"
        )
        atomic_write_text(path, body)
        return rel

    def _write_chunk_card(self, base: Path, node: WikiGraphNode) -> str:
        chunk_id = str(node.metadata.get("chunk_id", node.id))
        rel = f"wiki/wikigraph/chunks/{_slug_file(chunk_id)}.md"
        path = self.paths.root / rel
        body = (
            "---\n"
            "type: wikigraph_chunk\n"
            "generated: true\n"
            "retrieval_backend: wikigraph\n"
            f'title: "{node.title}"\n'
            f'generated_at: "{utc_now_iso()}"\n'
            "---\n\n"
            f"# {node.title}\n\n"
            f"{node.text}\n"
        )
        atomic_write_text(path, body)
        return rel


def _slug_file(value: str) -> str:
    from graphwiki_kb.services.project_service import slugify

    return slugify(value) or "item"
