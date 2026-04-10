from __future__ import annotations

from dataclasses import dataclass
import shutil

from src.services.project_service import ProjectPaths


@dataclass
class ExportResult:
    exported_paths: list[str]


class ExportService:
    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def export_vault(self) -> ExportResult:
        exported_paths: list[str] = []
        for file_path in sorted(self.paths.wiki_dir.rglob("*.md")):
            relative = file_path.relative_to(self.paths.wiki_dir)
            destination = self.paths.vault_obsidian_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file_path, destination)
            exported_paths.append(destination.relative_to(self.paths.root).as_posix())
        return ExportResult(exported_paths=exported_paths)
