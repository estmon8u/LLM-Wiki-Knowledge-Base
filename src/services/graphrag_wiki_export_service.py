from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import yaml

from src.services.graphrag_status_service import (
    GRAPH_OUTPUT_TABLES,
    GraphRAGStatus,
    GraphRAGStatusService,
)
from src.services.project_service import ProjectPaths, atomic_write_text, slugify
from src.services.search_service import SearchService


GRAPH_TABLE_DIRS = {
    "documents": "documents",
    "text_units": "text-units",
    "entities": "entities",
    "relationships": "relationships",
    "communities": "communities",
    "community_reports": "communities",
}
MAX_EXPORTED_RELATIONSHIP_PAGES = 500
MAX_ENTITY_RELATIONSHIP_ROWS = 50


class GraphRAGWikiExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class GraphRAGWikiExportResult:
    exported_paths: list[str]
    table_counts: dict[str, int]
    missing_tables: list[str] = field(default_factory=list)

    @property
    def exported_count(self) -> int:
        return len(self.exported_paths)

    def to_dict(self) -> dict[str, object]:
        return {
            "exported_count": self.exported_count,
            "exported_paths": self.exported_paths,
            "table_counts": self.table_counts,
            "missing_tables": self.missing_tables,
        }


class GraphRAGWikiExportService:
    def __init__(
        self,
        paths: ProjectPaths,
        status_service: GraphRAGStatusService,
        search_service: SearchService,
        *,
        refresh_index: Callable[[], None] | None = None,
    ) -> None:
        self.paths = paths
        self.status_service = status_service
        self.search_service = search_service
        self._refresh_index = refresh_index
        self.graph_wiki_dir = paths.wiki_dir / "graph"

    def export_wiki(self) -> GraphRAGWikiExportResult:
        status = self.status_service.status()
        self._require_export_ready(status)
        tables, missing = self._load_tables()
        self._clear_graph_wiki_dir()

        exported_paths: list[str] = []
        index_run_id = status.last_index_run_id
        relationships = tables.get("relationships", [])
        context = _ExportContext(
            index_run_id=index_run_id,
            relationships=relationships,
            relationships_by_entity=_relationships_by_entity(relationships),
            community_reports=tables.get("community_reports", []),
        )

        exported_paths.extend(
            self._write_documents(tables.get("documents", []), context)
        )
        exported_paths.extend(
            self._write_text_units(tables.get("text_units", []), context)
        )
        exported_paths.extend(self._write_entities(tables.get("entities", []), context))
        exported_paths.extend(
            self._write_relationships(tables.get("relationships", []), context)
        )
        exported_paths.extend(
            self._write_communities(
                tables.get("communities", []),
                tables.get("community_reports", []),
                context,
            )
        )
        index_path = self._write_index(
            tables=tables,
            missing_tables=missing,
            index_run_id=index_run_id,
        )
        exported_paths.insert(0, index_path)

        if self._refresh_index is not None:
            self._refresh_index()
        self.search_service.refresh(force=True)

        return GraphRAGWikiExportResult(
            exported_paths=sorted(exported_paths),
            table_counts={name: len(records) for name, records in tables.items()},
            missing_tables=missing,
        )

    def _require_export_ready(self, status: GraphRAGStatus) -> None:
        if not status.workspace_initialized:
            raise GraphRAGWikiExportError(
                "GraphRAG workspace is not initialized. Run `kb init` first."
            )
        if not status.output_present:
            raise GraphRAGWikiExportError(
                "GraphRAG index output not found. Run `kb update` before exporting "
                "graph wiki pages."
            )

    def _load_tables(self) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
        tables: dict[str, list[dict[str, Any]]] = {}
        missing: list[str] = []
        for table_name, tokens in GRAPH_OUTPUT_TABLES.items():
            table_path = self._find_table_path(*tokens)
            if table_path is None:
                missing.append(table_name)
                tables[table_name] = []
                continue
            tables[table_name] = _read_parquet_records(table_path)
        return tables, missing

    def _find_table_path(self, *tokens: str) -> Path | None:
        output_dir = self.status_service.output_dir
        if not output_dir.exists():
            return None
        for token in tokens:
            exact = output_dir / f"{token}.parquet"
            if exact.exists():
                return exact
        lowered = tuple(token.lower() for token in tokens)
        for path in sorted(output_dir.rglob("*.parquet")):
            stem = path.stem.lower()
            if any(stem == token or token in stem for token in lowered):
                return path
        return None

    def _clear_graph_wiki_dir(self) -> None:
        if not self.graph_wiki_dir.exists():
            return
        for path in sorted(self.graph_wiki_dir.rglob("*.md"), reverse=True):
            path.unlink()
        for directory in sorted(
            [p for p in self.graph_wiki_dir.rglob("*") if p.is_dir()],
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                continue

    def _write_documents(
        self, records: list[dict[str, Any]], context: "_ExportContext"
    ) -> list[str]:
        exported: list[str] = []
        used: set[str] = set()
        for record in records:
            title = _first_text(
                record, "title", "human_readable_id", "id", default="Document"
            )
            slug = _unique_slug(slugify(title), used, prefix="document")
            frontmatter = {
                "type": "graph_document",
                "document_id": _first_text(record, "id"),
                "title": title,
                "index_run_id": context.index_run_id,
            }
            body = [
                f"# {title}",
                "",
                "## GraphRAG Document",
                "",
                _field_list(record, exclude={"text", "raw_content", "raw_data"}),
            ]
            if _first_text(record, "text", "raw_content"):
                body.extend(
                    [
                        "",
                        "## Text",
                        "",
                        _fenced_text(_first_text(record, "text", "raw_content")),
                    ]
                )
            exported.append(
                self._write_page("documents", slug, frontmatter, "\n".join(body))
            )
        return exported

    def _write_text_units(
        self, records: list[dict[str, Any]], context: "_ExportContext"
    ) -> list[str]:
        exported: list[str] = []
        used: set[str] = set()
        for index, record in enumerate(records, start=1):
            text_unit_id = _first_text(record, "id", "human_readable_id") or str(index)
            slug = _unique_slug(
                slugify(f"text-unit-{text_unit_id}"), used, prefix="text-unit"
            )
            frontmatter = {
                "type": "graph_text_unit",
                "text_unit_id": text_unit_id,
                "index_run_id": context.index_run_id,
            }
            text = _first_text(record, "text", "content", "chunk", default="")
            body = [
                f"# Text Unit {text_unit_id}",
                "",
                "## Text",
                "",
                _fenced_text(text) if text else "No text content exported by GraphRAG.",
                "",
                "## Metadata",
                "",
                _field_list(record, exclude={"text", "content", "chunk", "raw_data"}),
            ]
            exported.append(
                self._write_page("text-units", slug, frontmatter, "\n".join(body))
            )
        return exported

    def _write_entities(
        self, records: list[dict[str, Any]], context: "_ExportContext"
    ) -> list[str]:
        exported: list[str] = []
        used: set[str] = set()
        for record in records:
            title = _first_text(record, "title", "name", "id", default="Entity")
            slug = _unique_slug(slugify(title), used, prefix="entity")
            frontmatter = {
                "type": "graph_entity",
                "entity_id": _first_text(record, "id"),
                "entity_title": title,
                "frequency": _sequence_count(record.get("text_unit_ids")),
                "degree": _first_number(record, "degree", "rank", "combined_degree"),
                "index_run_id": context.index_run_id,
            }
            relationships = _relationships_for_entity(
                context.relationships_by_entity,
                title,
                limit=MAX_ENTITY_RELATIONSHIP_ROWS,
            )
            body = [
                f"# {title}",
                "",
                "## GraphRAG Description",
                "",
                _first_text(
                    record,
                    "description",
                    default="No description exported by GraphRAG.",
                ),
                "",
                "## Appears In",
                "",
                _bullet_values(
                    record.get("text_unit_ids"), empty="No text units listed."
                ),
                "",
                "## Connected Entities",
                "",
                _relationship_table(relationships),
                "",
                "## Metadata",
                "",
                _field_list(
                    record,
                    exclude={"description", "title", "name", "raw_data"},
                ),
            ]
            exported.append(
                self._write_page("entities", slug, frontmatter, "\n".join(body))
            )
        return exported

    def _write_relationships(
        self, records: list[dict[str, Any]], context: "_ExportContext"
    ) -> list[str]:
        exported: list[str] = []
        used: set[str] = set()
        for record in _top_relationships(records, MAX_EXPORTED_RELATIONSHIP_PAGES):
            source = _first_text(record, "source", "source_title", default="source")
            target = _first_text(record, "target", "target_title", default="target")
            slug = _unique_slug(
                slugify(f"{source}--{target}"),
                used,
                prefix="relationship",
            )
            frontmatter = {
                "type": "graph_relationship",
                "relationship_id": _first_text(record, "id", "human_readable_id"),
                "source": source,
                "target": target,
                "weight": _first_number(record, "weight", "combined_degree", "rank"),
                "index_run_id": context.index_run_id,
            }
            body = [
                f"# {source} -> {target}",
                "",
                "## Description",
                "",
                _first_text(
                    record,
                    "description",
                    default="No relationship description exported by GraphRAG.",
                ),
                "",
                "## Metadata",
                "",
                _field_list(record, exclude={"description", "raw_data"}),
            ]
            exported.append(
                self._write_page("relationships", slug, frontmatter, "\n".join(body))
            )
        return exported

    def _write_communities(
        self,
        communities: list[dict[str, Any]],
        reports: list[dict[str, Any]],
        context: "_ExportContext",
    ) -> list[str]:
        exported: list[str] = []
        used: set[str] = set()
        reports_by_community = {
            _first_text(report, "community", "id"): report for report in reports
        }
        source_records = reports if reports else communities
        for index, record in enumerate(source_records, start=1):
            community_id = _first_text(record, "community", "id") or str(index)
            report = reports_by_community.get(community_id, {})
            merged = {**record, **report}
            title = _first_text(
                merged,
                "title",
                default=f"Community {community_id}",
            )
            slug = _unique_slug(
                slugify(f"community-{community_id}-{title}"), used, prefix="community"
            )
            frontmatter = {
                "type": "graph_community",
                "community_id": community_id,
                "level": _first_number(merged, "level"),
                "entity_count": _sequence_count(merged.get("entity_ids")),
                "text_unit_count": _sequence_count(merged.get("text_unit_ids")),
                "index_run_id": context.index_run_id,
            }
            body = [
                f"# Community {community_id} - {title}",
                "",
                "## Summary",
                "",
                _first_text(
                    merged,
                    "summary",
                    "full_content",
                    default="No community summary exported by GraphRAG.",
                ),
                "",
                "## Key Findings",
                "",
                _findings_markdown(merged),
                "",
                "## Entities",
                "",
                _bullet_values(merged.get("entity_ids"), empty="No entities listed."),
                "",
                "## Source Text Units",
                "",
                _bullet_values(
                    merged.get("text_unit_ids"), empty="No text units listed."
                ),
                "",
                "## Metadata",
                "",
                _field_list(
                    merged,
                    exclude={
                        "summary",
                        "full_content",
                        "findings",
                        "entity_ids",
                        "text_unit_ids",
                        "raw_data",
                    },
                ),
            ]
            exported.append(
                self._write_page("communities", slug, frontmatter, "\n".join(body))
            )
        return exported

    def _write_index(
        self,
        *,
        tables: dict[str, list[dict[str, Any]]],
        missing_tables: list[str],
        index_run_id: str | None,
    ) -> str:
        frontmatter = {
            "type": "graph_index",
            "index_run_id": index_run_id,
        }
        lines = [
            "# GraphRAG Index",
            "",
            "## Tables",
            "",
            "| Table | Rows |",
            "| --- | ---: |",
        ]
        for table_name in GRAPH_OUTPUT_TABLES:
            lines.append(f"| {table_name} | {len(tables.get(table_name, []))} |")
        lines.extend(["", "## Pages", ""])
        for table_name, directory in GRAPH_TABLE_DIRS.items():
            if table_name == "community_reports":
                continue
            row_count = len(tables.get(table_name, []))
            if (
                table_name == "relationships"
                and row_count > MAX_EXPORTED_RELATIONSHIP_PAGES
            ):
                lines.append(
                    f"- `{directory}/`: {MAX_EXPORTED_RELATIONSHIP_PAGES} of "
                    f"{row_count} page(s); export capped to the strongest "
                    "relationships."
                )
                continue
            lines.append(f"- `{directory}/`: {row_count} page(s)")
        if missing_tables:
            lines.extend(["", "## Missing Tables", ""])
            lines.extend(f"- `{table}`" for table in missing_tables)
        return self._write_page("", "index", frontmatter, "\n".join(lines))

    def _write_page(
        self,
        directory: str,
        slug: str,
        frontmatter: dict[str, Any],
        body: str,
    ) -> str:
        target_dir = (
            self.graph_wiki_dir / directory if directory else self.graph_wiki_dir
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{slug}.md"
        yaml_block = yaml.safe_dump(frontmatter, sort_keys=False).strip()
        atomic_write_text(path, f"---\n{yaml_block}\n---\n\n{body.rstrip()}\n")
        return path.relative_to(self.paths.root).as_posix()


@dataclass(frozen=True)
class _ExportContext:
    index_run_id: str | None
    relationships: list[dict[str, Any]]
    relationships_by_entity: dict[str, list[dict[str, Any]]]
    community_reports: list[dict[str, Any]]


def _read_parquet_records(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(path)
    return [
        {str(key): _clean_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "tolist"):
        return _clean_value(value.tolist())
    if isinstance(value, dict):
        return {str(key): _clean_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean_value(item) for item in value]
    return value


def _first_text(record: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if not isinstance(value, (dict, list, tuple, set)):
            text = str(value).strip()
            if text:
                return text
    return default


def _first_number(record: dict[str, Any], *keys: str) -> int | float | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float)) and not (
            isinstance(value, float) and math.isnan(value)
        ):
            return value
        if isinstance(value, str) and value.strip():
            try:
                number = float(value)
            except ValueError:
                continue
            return int(number) if number.is_integer() else number
    return None


def _sequence_count(value: Any) -> int:
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return 0


def _field_list(record: dict[str, Any], *, exclude: set[str] | None = None) -> str:
    excluded = exclude or set()
    lines: list[str] = []
    for key, value in sorted(record.items()):
        if key in excluded or value in (None, "", []):
            continue
        lines.append(f"- `{key}`: {_format_value(value)}")
    return "\n".join(lines) if lines else "No additional metadata."


def _format_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _bullet_values(value: Any, *, empty: str) -> str:
    if not isinstance(value, (list, tuple, set)) or not value:
        return empty
    return "\n".join(f"- `{item}`" for item in value)


def _fenced_text(text: str) -> str:
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}text\n{text.rstrip()}\n{fence}"


def _relationships_by_entity(
    relationships: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_entity: dict[str, list[dict[str, Any]]] = {}
    for relationship in relationships:
        source = _first_text(relationship, "source", "source_title").casefold()
        target = _first_text(relationship, "target", "target_title").casefold()
        if source:
            by_entity.setdefault(source, []).append(relationship)
        if target and target != source:
            by_entity.setdefault(target, []).append(relationship)
    return by_entity


def _relationships_for_entity(
    relationships_by_entity: dict[str, list[dict[str, Any]]],
    entity_title: str,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    return _top_relationships(
        relationships_by_entity.get(entity_title.casefold(), []),
        limit,
    )


def _top_relationships(
    relationships: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    return sorted(
        relationships,
        key=_relationship_sort_key,
        reverse=True,
    )[:limit]


def _relationship_sort_key(relationship: dict[str, Any]) -> float:
    score = _first_number(relationship, "weight", "combined_degree", "rank", "degree")
    return float(score) if score is not None else 0.0


def _relationship_table(relationships: list[dict[str, Any]]) -> str:
    if not relationships:
        return "No relationships listed."
    lines = ["| Entity | Relationship |", "| --- | --- |"]
    for relationship in relationships:
        source = _first_text(relationship, "source", "source_title")
        target = _first_text(relationship, "target", "target_title")
        description = _first_text(relationship, "description", default="related")
        lines.append(f"| {source} -> {target} | {description} |")
    return "\n".join(lines)


def _findings_markdown(record: dict[str, Any]) -> str:
    findings = record.get("findings")
    if isinstance(findings, list) and findings:
        lines = []
        for finding in findings:
            if isinstance(finding, dict):
                summary = _first_text(finding, "summary", "explanation", "text")
                if summary:
                    lines.append(f"- {summary}")
            elif str(finding).strip():
                lines.append(f"- {finding}")
        if lines:
            return "\n".join(lines)
    return "No key findings exported by GraphRAG."


def _unique_slug(base_slug: str, used: set[str], *, prefix: str) -> str:
    slug = base_slug or prefix
    if slug not in used:
        used.add(slug)
        return slug
    suffix = 2
    while f"{slug}-{suffix}" in used:
        suffix += 1
    unique = f"{slug}-{suffix}"
    used.add(unique)
    return unique
