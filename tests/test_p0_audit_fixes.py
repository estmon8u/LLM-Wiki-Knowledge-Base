"""Tests for P0 audit fixes: doctor, export --clean, search analysis pages,
diff/status file-hash recomputation, and click.Group migration."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from src.cli import main
from src.services.doctor_service import DoctorService
from src.services.export_service import ExportService
from src.services.search_service import (
    _extract_frontmatter_type,
    _is_generated_concept_page,
)


# ── Search: analysis pages are now searchable ────────────────────────


def _analysis_page(title: str, body: str) -> str:
    """Handles analysis page.

    Args:
        title: Title value used by the operation.
        body: Body value used by the operation.

    Returns:
        str produced by the operation.
    """
    return (
        "---\n"
        f"title: {title}\n"
        "type: analysis\n"
        "---\n\n"
        f"# {title}\n\n"
        f"{body}\n"
    )


def _concept_page(title: str, body: str) -> str:
    """Handles concept page.

    Args:
        title: Title value used by the operation.
        body: Body value used by the operation.

    Returns:
        str produced by the operation.
    """
    return (
        "---\n"
        f"title: {title}\n"
        "type: concept\n"
        "summary: A concept page\n"
        "generated_at: 2026-04-19T00:00:00Z\n"
        "source_pages: []\n"
        "---\n\n"
        f"# {title}\n\n"
        f"{body}\n"
    )


def test_search_includes_analysis_pages_in_concepts_dir(test_project) -> None:
    """Verifies that search includes analysis pages in concepts dir.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file(
        "wiki/concepts/my-analysis.md",
        _analysis_page("My Analysis", "traceability is key for citation grounding"),
    )
    test_project.write_file("wiki/sources/source-a.md", "traceability source content")

    results = test_project.services["search"].search("traceability")

    paths = {r.path for r in results}
    assert "wiki/concepts/my-analysis.md" in paths
    assert "wiki/sources/source-a.md" in paths


def test_search_excludes_generated_concept_pages(test_project) -> None:
    """Verifies that search excludes generated concept pages.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file(
        "wiki/concepts/gen-concept.md",
        _concept_page("Generated Concept", "traceability concept content"),
    )
    test_project.write_file("wiki/sources/source-a.md", "traceability source content")

    results = test_project.services["search"].search("traceability")

    paths = {r.path for r in results}
    assert "wiki/concepts/gen-concept.md" not in paths
    assert "wiki/sources/source-a.md" in paths


def test_extract_frontmatter_type_returns_type() -> None:
    """Verifies that extract frontmatter type returns type."""
    text = "---\ntitle: Foo\ntype: analysis\n---\n\n# Foo\n"
    assert _extract_frontmatter_type(text) == "analysis"


def test_extract_frontmatter_type_returns_empty_for_no_frontmatter() -> None:
    """Verifies that extract frontmatter type returns empty for no frontmatter."""
    assert _extract_frontmatter_type("# Just a heading\n") == ""


def test_extract_frontmatter_type_returns_empty_for_no_type_field() -> None:
    """Verifies that extract frontmatter type returns empty for no type field."""
    text = "---\ntitle: Foo\n---\n\n# Foo\n"
    assert _extract_frontmatter_type(text) == ""


def test_is_generated_concept_page_false_for_analysis(test_project) -> None:
    """Verifies that is generated concept page false for analysis.

    Args:
        test_project: Test project value used by the operation.
    """
    path = test_project.write_file(
        "wiki/concepts/an-analysis.md",
        _analysis_page("An Analysis", "body"),
    )
    assert _is_generated_concept_page(path, test_project.paths) is False


def test_is_generated_concept_page_true_for_concept(test_project) -> None:
    """Verifies that is generated concept page true for concept.

    Args:
        test_project: Test project value used by the operation.
    """
    path = test_project.write_file(
        "wiki/concepts/a-concept.md",
        _concept_page("A Concept", "body"),
    )
    assert _is_generated_concept_page(path, test_project.paths) is True


def test_is_generated_concept_page_false_outside_concepts_dir(test_project) -> None:
    """Verifies that is generated concept page false outside concepts dir.

    Args:
        test_project: Test project value used by the operation.
    """
    path = test_project.write_file("wiki/sources/some-source.md", "content")
    assert _is_generated_concept_page(path, test_project.paths) is False


def test_is_generated_concept_page_false_when_concepts_dir_missing(
    test_project,
) -> None:
    """Verifies that is generated concept page false when concepts dir missing.

    Args:
        test_project: Test project value used by the operation.
    """
    import shutil

    if test_project.paths.wiki_concepts_dir.exists():
        shutil.rmtree(test_project.paths.wiki_concepts_dir)
    path = test_project.root / "wiki" / "concepts" / "ghost.md"
    assert _is_generated_concept_page(path, test_project.paths) is False


def test_extract_frontmatter_type_handles_unclosed_frontmatter() -> None:
    """Verifies that extract frontmatter type handles unclosed frontmatter."""
    text = "---\ntitle: Foo\ntype: analysis\n"
    assert _extract_frontmatter_type(text) == ""


def test_strip_frontmatter_handles_unclosed_frontmatter() -> None:
    """Verifies that strip frontmatter handles unclosed frontmatter."""
    from src.services.search_service import _strip_frontmatter

    text = "---\ntitle: Foo\n"
    assert _strip_frontmatter(text) == text


def test_is_generated_concept_page_true_when_file_unreadable(
    test_project,
) -> None:
    """Verifies that is generated concept page true when file unreadable.

    Args:
        test_project: Test project value used by the operation.
    """
    # Create the concepts dir and a file, then remove the file to simulate OSError
    concepts_dir = test_project.paths.wiki_concepts_dir
    concepts_dir.mkdir(parents=True, exist_ok=True)
    ghost_path = concepts_dir / "ghost.md"
    ghost_path.write_text("temp", encoding="utf-8")
    ghost_path.unlink()
    # The file doesn't exist now, so reading it will fail with OSError
    assert _is_generated_concept_page(ghost_path, test_project.paths) is True


def test_is_generated_concept_page_false_when_path_is_dir_itself(
    test_project,
) -> None:
    """Verifies that is generated concept page false when path is dir itself.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.wiki_concepts_dir.mkdir(parents=True, exist_ok=True)
    assert (
        _is_generated_concept_page(
            test_project.paths.wiki_concepts_dir, test_project.paths
        )
        is False
    )


# ── Export --clean removes stale vault files ─────────────────────────


def test_export_vault_clean_removes_stale_files(test_project) -> None:
    """Verifies that export vault clean removes stale files.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file("wiki/sources/keep.md", "Keep this page")
    test_project.write_file("wiki/index.md", "Index")

    # First export — log.md is seeded by init, plus the two pages above
    export_service = test_project.services["export"]
    result1 = export_service.export_vault()
    assert len(result1.exported_paths) == 3
    assert (test_project.paths.vault_obsidian_dir / "sources" / "keep.md").exists()
    assert (test_project.paths.vault_obsidian_dir / "index.md").exists()
    assert (test_project.paths.vault_obsidian_dir / "log.md").exists()

    # Add a stale file directly in vault (simulating deleted wiki page)
    stale = test_project.paths.vault_obsidian_dir / "sources" / "deleted.md"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("Stale content", encoding="utf-8")
    assert stale.exists()

    # Export with --clean should remove the stale file
    result2 = export_service.export_vault(clean=True)
    assert not stale.exists()
    assert len(result2.removed_paths) == 1
    assert "deleted.md" in result2.removed_paths[0]


def test_export_vault_without_clean_keeps_stale_files(test_project) -> None:
    """Verifies that export vault without clean keeps stale files.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file("wiki/sources/keep.md", "Keep page")

    export_service = test_project.services["export"]
    export_service.export_vault()

    stale = test_project.paths.vault_obsidian_dir / "sources" / "deleted.md"
    stale.write_text("Stale content", encoding="utf-8")

    result = export_service.export_vault(clean=False)
    assert stale.exists()
    assert result.removed_paths == []


# ── Diff/Status: file-based hash recomputation ──────────────────────


def test_diff_detects_normalized_file_edit_on_disk(test_project) -> None:
    """Verifies that diff detects normalized file edit on disk.

    Args:
        test_project: Test project value used by the operation.
    """
    source_path = test_project.write_file("notes/doc.md", "# Doc\n\nOriginal body.\n")
    test_project.services["ingest"].ingest_path(source_path)
    test_project.services["compile"].compile()

    # Verify up-to-date
    report = test_project.services["diff"].diff()
    assert report.up_to_date_count == 1

    # Manually edit the normalized file on disk
    sources = test_project.services["manifest"].list_sources()
    norm_path = test_project.root / (sources[0].normalized_path or sources[0].raw_path)
    norm_path.write_text("# Doc\n\nModified body via disk edit.\n", encoding="utf-8")

    report2 = test_project.services["diff"].diff()
    assert report2.changed_count == 1
    assert "normalized file changed on disk" in report2.entries[0].details


def test_status_compiled_count_drops_when_file_edited_on_disk(test_project) -> None:
    """Verifies that status compiled count drops when file edited on disk.

    Args:
        test_project: Test project value used by the operation.
    """
    source_path = test_project.write_file("notes/doc.md", "# Doc\n\nOriginal body.\n")
    test_project.services["ingest"].ingest_path(source_path)
    test_project.services["compile"].compile()

    snap1 = test_project.services["status"].snapshot(initialized=True)
    assert snap1.compiled_source_count == 1

    # Edit normalized file on disk
    sources = test_project.services["manifest"].list_sources()
    norm_path = test_project.root / (sources[0].normalized_path or sources[0].raw_path)
    norm_path.write_text("# Doc\n\nSneaky edit.\n", encoding="utf-8")

    snap2 = test_project.services["status"].snapshot(initialized=True)
    assert snap2.compiled_source_count == 0


# ── Doctor command ───────────────────────────────────────────────────


def test_doctor_passes_for_initialized_project(test_project) -> None:
    """Verifies that doctor passes for initialized project.

    Args:
        test_project: Test project value used by the operation.
    """
    doctor = test_project.services["doctor"]
    report = doctor.diagnose()
    # project_structure, config_file, schema_file, manifest should pass
    passed_names = {c.name for c in report.checks if c.ok}
    assert "project_structure" in passed_names
    assert "config_file" in passed_names
    assert "schema_file" in passed_names
    assert "manifest" in passed_names


def test_doctor_detects_missing_provider_config(test_project) -> None:
    """Verifies that doctor detects missing provider config.

    Args:
        test_project: Test project value used by the operation.
    """
    doctor = DoctorService(test_project.paths, {})
    report = doctor.diagnose()
    provider_check = next(c for c in report.checks if c.name == "provider_config")
    assert not provider_check.ok
    assert "No provider configured" in provider_check.detail


def test_doctor_detects_missing_api_key(test_project) -> None:
    """Verifies that doctor detects missing api key.

    Args:
        test_project: Test project value used by the operation.
    """
    config = {"provider": {"name": "openai"}}
    doctor = DoctorService(test_project.paths, config)
    with patch.dict("os.environ", {}, clear=True):
        report = doctor.diagnose()
    api_check = next(c for c in report.checks if c.name == "api_key")
    assert not api_check.ok
    assert "OPENAI_API_KEY" in api_check.detail


def test_doctor_detects_api_key_set(test_project) -> None:
    """Verifies that doctor detects api key set.

    Args:
        test_project: Test project value used by the operation.
    """
    config = {"provider": {"name": "openai"}}
    doctor = DoctorService(test_project.paths, config)
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
        report = doctor.diagnose()
    api_check = next(c for c in report.checks if c.name == "api_key")
    assert api_check.ok


def test_doctor_detects_unknown_provider(test_project) -> None:
    """Verifies that doctor detects unknown provider.

    Args:
        test_project: Test project value used by the operation.
    """
    config = {"provider": {"name": "fakeprovider"}}
    doctor = DoctorService(test_project.paths, config)
    report = doctor.diagnose()
    provider_check = next(c for c in report.checks if c.name == "provider_config")
    assert not provider_check.ok
    assert "Unknown provider" in provider_check.detail


def test_doctor_checks_converters(test_project) -> None:
    """Verifies that doctor checks converters.

    Args:
        test_project: Test project value used by the operation.
    """
    doctor = test_project.services["doctor"]
    report = doctor.diagnose()
    conv_check = next(c for c in report.checks if c.name == "converters")
    # At minimum we check it doesn't crash; converter availability depends on env
    assert isinstance(conv_check.ok, bool)


def test_doctor_fails_for_uninitialized_project(uninitialized_project) -> None:
    """Verifies that doctor fails for uninitialized project.

    Args:
        uninitialized_project: Uninitialized project value used by the operation.
    """
    doctor = DoctorService(uninitialized_project.paths, {})
    report = doctor.diagnose()
    structure_check = next(c for c in report.checks if c.name == "project_structure")
    assert not structure_check.ok
    assert "Missing directories" in structure_check.detail
    config_check = next(c for c in report.checks if c.name == "config_file")
    assert not config_check.ok
    schema_check = next(c for c in report.checks if c.name == "schema_file")
    assert not schema_check.ok
    manifest_check = next(c for c in report.checks if c.name == "manifest")
    assert not manifest_check.ok


def test_doctor_no_provider_means_no_api_key_check(test_project) -> None:
    """Verifies that doctor no provider means no api key check.

    Args:
        test_project: Test project value used by the operation.
    """
    doctor = DoctorService(test_project.paths, {})
    report = doctor.diagnose()
    api_check = next(c for c in report.checks if c.name == "api_key")
    assert not api_check.ok
    assert "Cannot check API key" in api_check.detail
    assert api_check.severity == "warning"


def test_doctor_unknown_provider_has_no_api_key_env(test_project) -> None:
    """Verifies that doctor unknown provider has no api key env.

    Args:
        test_project: Test project value used by the operation.
    """
    config = {"provider": {"name": "mysterybox"}}
    doctor = DoctorService(test_project.paths, config)
    report = doctor.diagnose()
    api_check = next(c for c in report.checks if c.name == "api_key")
    assert not api_check.ok
    assert "No API key env variable" in api_check.detail


def test_doctor_configured_provider_with_model(test_project) -> None:
    """Verifies that doctor configured provider with model.

    Args:
        test_project: Test project value used by the operation.
    """
    config = {"provider": {"name": "anthropic", "model": "claude-opus-4-6"}}
    doctor = DoctorService(test_project.paths, config)
    report = doctor.diagnose()
    prov_check = next(c for c in report.checks if c.name == "provider_config")
    assert prov_check.ok
    assert "anthropic" in prov_check.detail
    assert "claude-opus-4-6" in prov_check.detail


def test_doctor_uses_catalog_default_model_when_override_missing(test_project) -> None:
    """Verifies that doctor uses catalog default model when override missing.

    Args:
        test_project: Test project value used by the operation.
    """
    config = {
        "provider": {"name": "gemini"},
        "providers": {
            "openai": {
                "model": "gpt-5.4-nano",
                "api_key_env": "OPENAI_API_KEY",
                "reasoning_effort": "high",
            },
            "anthropic": {
                "model": "claude-sonnet-4-6",
                "api_key_env": "ANTHROPIC_API_KEY",
                "thinking_budget": 10_000,
            },
            "gemini": {
                "model": "gemini-2.5-flash",
                "api_key_env": "GEMINI_ALT_KEY",
                "reasoning_effort": "medium",
            },
        },
    }
    doctor = DoctorService(test_project.paths, config)
    report = doctor.diagnose()
    prov_check = next(c for c in report.checks if c.name == "provider_config")
    api_check = next(c for c in report.checks if c.name == "api_key")
    assert prov_check.ok
    assert "gemini-2.5-flash" in prov_check.detail
    assert "GEMINI_ALT_KEY" in api_check.detail


def test_doctor_report_ok_property() -> None:
    """Verifies that doctor report ok property."""
    from src.services.doctor_service import DoctorCheck, DoctorReport

    report = DoctorReport(
        checks=[
            DoctorCheck(name="a", ok=True, detail="good", severity="ok"),
            DoctorCheck(name="b", ok=True, detail="fine", severity="ok"),
        ]
    )
    assert report.ok is True
    assert report.passed_count == 2
    assert report.failed_count == 0

    report2 = DoctorReport(
        checks=[
            DoctorCheck(name="a", ok=True, detail="good", severity="ok"),
            DoctorCheck(name="b", ok=False, detail="bad", severity="error"),
        ]
    )
    assert report2.ok is False
    assert report2.passed_count == 1
    assert report2.failed_count == 1

    # Warnings don't fail the report
    report3 = DoctorReport(
        checks=[
            DoctorCheck(name="a", ok=True, detail="good", severity="ok"),
            DoctorCheck(name="b", ok=False, detail="meh", severity="warning"),
        ]
    )
    assert report3.ok is True
    assert report3.warning_count == 1


def test_doctor_detects_missing_converters(test_project) -> None:
    """Verifies that doctor detects missing converters.

    Args:
        test_project: Test project value used by the operation.
    """
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        """Handles fake import.

        Args:
            name: Name value used for lookup or display.
            args: Parsed or forwarded command arguments.
            kwargs: Kwargs value used by the operation.
        """
        if name in ("markitdown", "docling"):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    doctor = test_project.services["doctor"]
    with patch("builtins.__import__", side_effect=_fake_import):
        report = doctor.diagnose()
    conv_check = next(c for c in report.checks if c.name == "converters")
    assert not conv_check.ok
    assert "Missing" in conv_check.detail
    assert "MarkItDown" in conv_check.detail
    assert "Docling" in conv_check.detail


# ── Diff/Status: missing normalized source detection ─────────────────


def test_diff_reports_missing_when_normalized_file_is_deleted(test_project) -> None:
    """Verifies that diff reports missing normalized source files.

    Args:
        test_project: Test project value used by the operation.
    """
    source_path = test_project.write_file("notes/doc.md", "# Doc\n\nBody.\n")
    test_project.services["ingest"].ingest_path(source_path)
    test_project.services["compile"].compile()

    sources = test_project.services["manifest"].list_sources()
    norm_path = test_project.root / (sources[0].normalized_path or sources[0].raw_path)
    norm_path.unlink()

    report = test_project.services["diff"].diff()
    assert report.missing_count == 1
    assert report.up_to_date_count == 0
    assert report.entries[0].status == "missing"
    assert "missing" in report.entries[0].details


def test_status_does_not_count_deleted_normalized_file_as_compiled(
    test_project,
) -> None:
    """Verifies that status does not count deleted normalized sources as compiled.

    Args:
        test_project: Test project value used by the operation.
    """
    source_path = test_project.write_file("notes/doc.md", "# Doc\n\nBody.\n")
    test_project.services["ingest"].ingest_path(source_path)
    test_project.services["compile"].compile()

    sources = test_project.services["manifest"].list_sources()
    norm_path = test_project.root / (sources[0].normalized_path or sources[0].raw_path)
    norm_path.unlink()

    snap = test_project.services["status"].snapshot(initialized=True)
    assert snap.compiled_source_count == 0


# ── Doctor CLI ───────────────────────────────────────────────────────


def test_doctor_cli_runs(tmp_path) -> None:
    """Verifies that doctor cli runs.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        assert runner.invoke(main, ["init"]).exit_code == 0
        result = runner.invoke(main, ["doctor"])
        assert "Health Checks" in result.output
        assert "project_structure" in result.output
        assert "passed" in result.output


def test_doctor_cli_in_help_output() -> None:
    """Verifies that doctor cli in help output."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert "doctor" in result.output


# ── Export --clean CLI ───────────────────────────────────────────────


def test_export_clean_cli_flag(tmp_path) -> None:
    """Verifies that export clean cli flag.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        assert runner.invoke(main, ["init"]).exit_code == 0
        Path("wiki/sources").mkdir(parents=True, exist_ok=True)
        Path("wiki/sources/page.md").write_text("Content", encoding="utf-8")
        assert runner.invoke(main, ["export"]).exit_code == 0

        # Add a stale file to the vault
        Path("vault/obsidian/sources/stale.md").write_text("Old", encoding="utf-8")

        result = runner.invoke(main, ["export", "--clean"])
        assert result.exit_code == 0
        assert "Removed" in result.output
        assert not Path("vault/obsidian/sources/stale.md").exists()


# ── click.Group migration ───────────────────────────────────────────


def test_kb_group_is_click_group_not_multi_command() -> None:
    """Verifies that kb group is click group not multi command."""
    import click
    from src.cli import KBGroup

    assert issubclass(KBGroup, click.Group)
