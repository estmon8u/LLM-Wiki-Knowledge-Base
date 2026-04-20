from __future__ import annotations

import argparse
import locale
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_SUFFIXES = {
    ".csv",
    ".docx",
    ".epub",
    ".htm",
    ".html",
    ".ipynb",
    ".markdown",
    ".md",
    ".pdf",
    ".pptx",
    ".txt",
    ".xls",
    ".xlsx",
}


@dataclass
class CommandRun:
    args: list[str]
    exit_code: int
    output: str


def discover_supported_sources(raw_root: Path) -> list[Path]:
    return sorted(
        path
        for path in raw_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def find_unsupported_probe(raw_root: Path) -> Path | None:
    unsupported = sorted(
        path
        for path in raw_root.rglob("*")
        if path.is_file() and path.suffix.lower() not in SUPPORTED_SUFFIXES
    )
    return unsupported[0] if unsupported else None


def run_cli_command(
    repo_root: Path,
    log_path: Path,
    args: list[str],
) -> CommandRun:
    command_display = "python -m src.cli " + " ".join(args)
    header = f"\n=== {command_display} ===\n"
    print(header, end="")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(header)

    result = subprocess.run(
        [sys.executable, "-m", "src.cli", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding=locale.getpreferredencoding(False),
        errors="replace",
    )
    combined_output = result.stdout
    if result.stderr:
        combined_output += result.stderr

    if combined_output:
        print(combined_output, end="")

    footer = f"exit_code={result.returncode}\n"
    print(footer, end="")
    with log_path.open("a", encoding="utf-8") as handle:
        if combined_output:
            handle.write(combined_output)
        handle.write(footer)

    return CommandRun(args=args, exit_code=result.returncode, output=combined_output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a disposable end-to-end CLI smoke test against a raw corpus."
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--search-query", default="knowledge base")
    parser.add_argument(
        "--question",
        default="How does the wiki help maintain knowledge over time?",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    raw_root = args.raw_root.resolve()
    project_root = args.project_root.resolve()
    log_path = (
        args.log_file.resolve()
        if args.log_file
        else project_root / "command-smoke-log.txt"
    )

    if not raw_root.exists():
        parser.error(f"Raw root does not exist: {raw_root}")

    if project_root.exists():
        shutil.rmtree(project_root)
    project_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    supported_sources = discover_supported_sources(raw_root)
    unsupported_probe = find_unsupported_probe(raw_root)

    failed_supported_ingests: list[Path] = []
    unexpected_failures: list[str] = []
    skipped_provider_commands: list[str] = []
    lint_failed = False

    for command_args, allow_failure in (
        (["--help"], False),
        (["--project-root", str(project_root), "init"], False),
        (["--project-root", str(project_root), "status"], False),
    ):
        result = run_cli_command(repo_root, log_path, command_args)
        if result.exit_code != 0 and not allow_failure:
            unexpected_failures.append(" ".join(command_args))

    for source_path in supported_sources:
        result = run_cli_command(
            repo_root,
            log_path,
            ["--project-root", str(project_root), "add", str(source_path)],
        )
        if result.exit_code != 0:
            failed_supported_ingests.append(source_path)

    if unsupported_probe is not None:
        result = run_cli_command(
            repo_root,
            log_path,
            ["--project-root", str(project_root), "add", str(unsupported_probe)],
        )
        if result.exit_code == 0:
            unexpected_failures.append(
                f"unsupported ingest unexpectedly succeeded: {unsupported_probe}"
            )

    for command_args in (
        ["--project-root", str(project_root), "status", "--changed"],
        ["--project-root", str(project_root), "update"],
        ["--project-root", str(project_root), "status", "--changed"],
        ["--project-root", str(project_root), "status"],
        ["--project-root", str(project_root), "find", *args.search_query.split()],
        ["--project-root", str(project_root), "ask", *args.question.split()],
    ):
        result = run_cli_command(repo_root, log_path, command_args)
        if result.exit_code != 0:
            command_text = " ".join(command_args)
            if (
                "requires a configured provider" in result.output
                or "Provider is not configured" in result.output
            ):
                skipped_provider_commands.append(command_text)
            else:
                unexpected_failures.append(command_text)

    lint_result = run_cli_command(
        repo_root,
        log_path,
        ["--project-root", str(project_root), "lint"],
    )
    lint_failed = lint_result.exit_code != 0

    export_result = run_cli_command(
        repo_root,
        log_path,
        ["--project-root", str(project_root), "export"],
    )
    if export_result.exit_code != 0:
        unexpected_failures.append(f"--project-root {project_root} export")

    summary_lines = [
        "",
        "Summary:",
        f"- supported_sources: {len(supported_sources)}",
        f"- failed_supported_ingests: {len(failed_supported_ingests)}",
        f"- lint_failed: {str(lint_failed).lower()}",
        f"- skipped_provider_commands: {len(skipped_provider_commands)}",
        f"- unexpected_failures: {len(unexpected_failures)}",
        f"- log_file: {log_path}",
    ]
    if failed_supported_ingests:
        summary_lines.append("- failed_ingest_paths:")
        summary_lines.extend(f"  - {path}" for path in failed_supported_ingests)
    if unexpected_failures:
        summary_lines.append("- unexpected_failure_commands:")
        summary_lines.extend(f"  - {item}" for item in unexpected_failures)
    if skipped_provider_commands:
        summary_lines.append("- skipped_provider_command_list:")
        summary_lines.extend(f"  - {item}" for item in skipped_provider_commands)

    summary_text = "\n".join(summary_lines) + "\n"
    print(summary_text, end="")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(summary_text)

    if failed_supported_ingests or lint_failed or unexpected_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
