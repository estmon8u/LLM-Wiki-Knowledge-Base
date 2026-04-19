from __future__ import annotations

from dataclasses import dataclass, field
import shutil

from src.services.project_service import ProjectPaths


@dataclass
class ExportResult:
    exported_paths: list[str]
    removed_paths: list[str] = field(default_factory=list)


class ExportService:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def export_vault(self, *, clean: bool = False) -> ExportResult:
        exported_paths: list[str] = []
        removed_paths: list[str] = []
        # Copy current wiki pages into the vault.
        for file_path in sorted(self.paths.wiki_dir.rglob("*.md")):
            relative = file_path.relative_to(self.paths.wiki_dir)
            destination = self.paths.vault_obsidian_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file_path, destination)
            exported_paths.append(destination.relative_to(self.paths.root).as_posix())
        # Remove stale vault files that no longer exist in wiki.
        if clean and self.paths.vault_obsidian_dir.exists():
            exported_set = {
                p.relative_to(self.paths.vault_obsidian_dir).as_posix()
                for p in self.paths.vault_obsidian_dir.rglob("*.md")
                if (
                    self.paths.wiki_dir / p.relative_to(self.paths.vault_obsidian_dir)
                ).exists()
            }
            for vault_file in sorted(self.paths.vault_obsidian_dir.rglob("*.md")):
                rel = vault_file.relative_to(self.paths.vault_obsidian_dir).as_posix()
                if rel not in exported_set:
                    vault_file.unlink()
                    removed_paths.append(
                        vault_file.relative_to(self.paths.root).as_posix()
                    )
        return ExportResult(exported_paths=exported_paths, removed_paths=removed_paths)
