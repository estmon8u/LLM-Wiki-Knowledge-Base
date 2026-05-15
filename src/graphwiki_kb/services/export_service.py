"""Export service service behavior for the knowledge-base workflow.

This module belongs to `graphwiki_kb.services.export_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from graphwiki_kb.services.project_service import ProjectPaths, atomic_copy_file


@dataclass
class ExportResult:
    """Stores export result data.

    Attributes:
        See annotated class attributes for stored values.
    """

    exported_paths: list[str]
    removed_paths: list[str] = field(default_factory=list)


class ExportService:
    """Coordinates export operations.

    Attributes:
        See annotated class attributes for stored values.
    """

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def export_vault(self, *, clean: bool = False) -> ExportResult:
        """Export vault.

        Args:
            clean: Clean value used by the operation.

        Returns:
            ExportResult produced by the operation.
        """
        exported_paths: list[str] = []
        removed_paths: list[str] = []
        exported_set: set[str] = set()
        # Copy current wiki pages into the vault.
        for file_path in sorted(self.paths.wiki_dir.rglob("*.md")):
            relative = file_path.relative_to(self.paths.wiki_dir)
            destination = self.paths.vault_obsidian_dir / relative
            atomic_copy_file(file_path, destination)
            exported_set.add(relative.as_posix())
            exported_paths.append(destination.relative_to(self.paths.root).as_posix())
        # Remove stale vault files that no longer exist in wiki.
        if clean and self.paths.vault_obsidian_dir.exists():
            for vault_file in sorted(self.paths.vault_obsidian_dir.rglob("*.md")):
                rel = vault_file.relative_to(self.paths.vault_obsidian_dir).as_posix()
                if rel not in exported_set:
                    vault_file.unlink()
                    removed_paths.append(
                        vault_file.relative_to(self.paths.root).as_posix()
                    )
        return ExportResult(exported_paths=exported_paths, removed_paths=removed_paths)
