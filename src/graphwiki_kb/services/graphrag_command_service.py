"""GraphRAG Python entrypoint adapter for indexing and querying."""

from __future__ import annotations

import inspect
import io
import os
import subprocess
from collections.abc import Callable, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from graphwiki_kb.services.file_lock import workspace_lock
from graphwiki_kb.services.graphrag_runtime import (
    GraphRAGCompatibilityError,
    validate_graphrag_runtime,
)
from graphwiki_kb.services.project_service import ProjectPaths

Runner = Callable[..., subprocess.CompletedProcess[str]]
StatusCallback = Callable[[str], None] | None
MAX_PROGRESS_LABEL_LENGTH = 120
DEFAULT_QUERY_RESPONSE_TYPE = "Multiple Paragraphs"
DEFAULT_QUERY_COMMUNITY_LEVEL = 2
_KNOWN_GRAPHRAG_ENTRYPOINT_DEFAULTS: dict[str, Any] = {
    "config_filepath": None,
    "memprofile": False,
    "output_dir": None,
    "streaming": False,
}


@dataclass(frozen=True)
class GraphRAGCommandResult:
    """Result metadata kept for GraphRAG run history and diagnostics."""

    command: tuple[str, ...]
    cwd: Path
    returncode: int
    stdout: str
    stderr: str


class GraphRAGCommandError(RuntimeError):
    """Error raised for GraphRAG library API failures."""

    def __init__(self, message: str, *, result: GraphRAGCommandResult | None = None):
        super().__init__(message)
        self.result = result


class GraphRAGCommandService:
    """Runs GraphRAG init, index, and query without spawning `python -m graphrag`."""

    @staticmethod
    def _default_runner(
        command: Sequence[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        """Compatibility helper retained for older subprocess-focused tests."""
        kwargs.setdefault("encoding", "utf-8")
        return subprocess.run(command, **kwargs)

    def __init__(
        self,
        paths: ProjectPaths,
        *,
        runner: Runner | None = None,
        api_backend: GraphRAGApiBackend | None = None,
        fallback_backend: SubprocessGraphRAGApiBackend | None = None,
        use_subprocess_backend: bool = False,
    ) -> None:
        self.paths = paths
        self.runner = runner or self._default_runner
        self.api_backend: (
            GraphRAGApiBackend
            | _RunnerGraphRAGApiBackend
            | SubprocessGraphRAGApiBackend
        )
        self._fallback_backend: SubprocessGraphRAGApiBackend | None = None
        if use_subprocess_backend:
            self.api_backend = fallback_backend or SubprocessGraphRAGApiBackend(
                paths, self.runner
            )
        elif api_backend is not None:
            self.api_backend = api_backend
            self._fallback_backend = fallback_backend
        elif runner is not None:
            self.api_backend = _RunnerGraphRAGApiBackend(paths, runner)
        else:
            self.api_backend = GraphRAGApiBackend(paths)
            self._fallback_backend = fallback_backend or SubprocessGraphRAGApiBackend(
                paths, self.runner
            )
        self.workspace_dir = paths.graph_dir / "graphrag"

    def init_workspace(
        self,
        *,
        model: str,
        embedding: str,
        force: bool,
    ) -> GraphRAGCommandResult:
        """Initialize the GraphRAG workspace through GraphRAG's Python helper."""
        with workspace_lock(self.workspace_dir):
            result = self._call_backend(
                "init_workspace",
                workspace_dir=self.workspace_dir,
                model=model,
                embedding=embedding,
                force=force,
            )
        self._raise_if_failed(result)
        return result

    def index(
        self,
        *,
        method: str,
        dry_run: bool,
        cache: bool,
        skip_validation: bool,
        verbose: bool = False,
        status_callback: StatusCallback = None,
    ) -> GraphRAGCommandResult:
        """Run GraphRAG indexing through installed Python entrypoints."""
        with workspace_lock(self.workspace_dir):
            result = self._call_backend(
                "index",
                workspace_dir=self.workspace_dir,
                method=method,
                dry_run=dry_run,
                cache=cache,
                skip_validation=skip_validation,
                verbose=verbose,
                status_callback=status_callback,
            )
        self._raise_if_failed(result)
        return result

    def query(
        self,
        question: str,
        *,
        method: str,
        data_dir: Path | None = None,
        community_level: int | None = None,
        dynamic_community_selection: bool | None = None,
        response_type: str | None = None,
        streaming: bool | None = None,
        verbose: bool = False,
    ) -> GraphRAGCommandResult:
        """Run a GraphRAG search query through installed Python entrypoints."""
        with workspace_lock(self.workspace_dir):
            result = self._call_backend(
                "query",
                workspace_dir=self.workspace_dir,
                question=question,
                method=method,
                data_dir=data_dir,
                community_level=community_level,
                dynamic_community_selection=dynamic_community_selection,
                response_type=response_type,
                streaming=streaming,
                verbose=verbose,
            )
        self._raise_if_failed(result)
        return result

    def _call_backend(self, operation: str, **kwargs: Any) -> GraphRAGCommandResult:
        method = getattr(self.api_backend, operation)
        try:
            result = method(**kwargs)
        except GraphRAGCommandError as exc:
            if self._fallback_backend is None or not _should_fallback_from_entrypoint(
                str(exc)
            ):
                raise
            fallback_method = getattr(self._fallback_backend, operation)
            return fallback_method(**kwargs)
        if self._fallback_backend is not None and _should_fallback_from_entrypoint(
            result.stderr
        ):
            fallback_method = getattr(self._fallback_backend, operation)
            return fallback_method(**kwargs)
        return result

    def _raise_if_failed(self, result: GraphRAGCommandResult) -> None:
        if result.returncode == 0:
            return
        detail = _last_non_empty_line(result.stderr) or _last_non_empty_line(
            result.stdout
        )
        message = "GraphRAG command failed"
        if detail:
            message = f"{message}: {detail}"
        raise GraphRAGCommandError(message, result=result)


class GraphRAGApiBackend:
    """Small adapter over GraphRAG's installed Python command entrypoints."""

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths
        self._runtime_validated = False

    def _ensure_runtime(self) -> None:
        if self._runtime_validated:
            return
        try:
            validate_graphrag_runtime()
        except GraphRAGCompatibilityError as exc:
            raise GraphRAGCommandError(str(exc)) from exc
        self._runtime_validated = True

    def init_workspace(
        self,
        *,
        workspace_dir: Path,
        model: str,
        embedding: str,
        force: bool,
    ) -> GraphRAGCommandResult:  # pragma: no cover - exercised by GraphRAG package
        self._ensure_runtime()
        command = _init_command(workspace_dir, model, embedding, force)
        try:
            from graphrag.cli.initialize import initialize_project_at

            initialize_project_at(
                workspace_dir,
                force=force,
                model=model,
                embedding_model=embedding,
            )
        except Exception as exc:
            return _api_result(command, self.paths.root, returncode=1, stderr=str(exc))
        return _api_result(
            command,
            self.paths.root,
            stdout=f"Initialized GraphRAG workspace at {workspace_dir}\n",
        )

    def index(
        self,
        *,
        workspace_dir: Path,
        method: str,
        dry_run: bool,
        cache: bool,
        skip_validation: bool,
        verbose: bool,
        status_callback: StatusCallback,
    ) -> GraphRAGCommandResult:  # pragma: no cover - exercised by GraphRAG package
        self._ensure_runtime()
        command = _index_command(
            workspace_dir,
            method=method,
            dry_run=dry_run,
            cache=cache,
            skip_validation=skip_validation,
            verbose=verbose,
        )
        if status_callback is not None:
            status_callback("starting graph index")
        stdout = _ProgressCapture(status_callback)
        stderr = _ProgressCapture(status_callback)
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                _run_index_entrypoint(
                    workspace_dir=workspace_dir,
                    method=method,
                    dry_run=dry_run,
                    cache=cache,
                    skip_validation=skip_validation,
                    verbose=verbose,
                )
        except SystemExit as exc:
            stdout.flush()
            stderr.flush()
            returncode = _system_exit_code(exc)
            stderr_text = stderr.getvalue()
            if returncode and not stderr_text:
                stderr_text = str(exc)
            return _api_result(
                command,
                self.paths.root,
                returncode=returncode,
                stdout=stdout.getvalue(),
                stderr=stderr_text,
            )
        except Exception as exc:
            stdout.flush()
            stderr.flush()
            stderr_text = _append_error(stderr.getvalue(), str(exc))
            return _api_result(
                command,
                self.paths.root,
                returncode=1,
                stdout=stdout.getvalue(),
                stderr=stderr_text,
            )
        stdout.flush()
        stderr.flush()
        return _api_result(
            command,
            self.paths.root,
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
        )

    def query(
        self,
        *,
        workspace_dir: Path,
        question: str,
        method: str,
        data_dir: Path | None,
        community_level: int | None,
        dynamic_community_selection: bool | None,
        response_type: str | None,
        streaming: bool | None,
        verbose: bool,
    ) -> GraphRAGCommandResult:  # pragma: no cover - exercised by GraphRAG package
        self._ensure_runtime()
        command = _query_command(
            workspace_dir,
            question,
            method=method,
            data_dir=data_dir,
            community_level=community_level,
            dynamic_community_selection=dynamic_community_selection,
            response_type=response_type,
            streaming=streaming,
            verbose=verbose,
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                query_result = _run_query_entrypoint(
                    workspace_dir=workspace_dir,
                    data_dir=data_dir,
                    method=method,
                    community_level=community_level,
                    dynamic_community_selection=dynamic_community_selection,
                    response_type=response_type,
                    streaming=streaming,
                    question=question,
                    verbose=verbose,
                )
        except Exception as exc:
            stderr_text = _append_error(stderr.getvalue(), str(exc))
            return _api_result(
                command,
                self.paths.root,
                returncode=1,
                stdout=stdout.getvalue(),
                stderr=stderr_text,
            )
        stdout_text = stdout.getvalue()
        if not stdout_text.strip():
            returned_answer = _query_return_to_text(query_result)
            if returned_answer:
                stdout_text = f"{returned_answer}\n"
        return _api_result(
            command,
            self.paths.root,
            stdout=stdout_text,
            stderr=stderr.getvalue(),
        )


class _RunnerGraphRAGApiBackend:
    """Compatibility backend for tests that inject a fake process runner."""

    def __init__(self, paths: ProjectPaths, runner: Runner) -> None:
        self.paths = paths
        self.runner = runner

    def init_workspace(
        self,
        *,
        workspace_dir: Path,
        model: str,
        embedding: str,
        force: bool,
    ) -> GraphRAGCommandResult:
        command = _init_command(workspace_dir, model, embedding, force)
        completed = self.runner(
            command,
            cwd=self.paths.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_graphrag_subprocess_env(),
        )
        return _completed_process_result(command, self.paths.root, completed)

    def index(
        self,
        *,
        workspace_dir: Path,
        method: str,
        dry_run: bool,
        cache: bool,
        skip_validation: bool,
        verbose: bool,
        status_callback: StatusCallback,
    ) -> GraphRAGCommandResult:
        command = _index_command(
            workspace_dir,
            method=method,
            dry_run=dry_run,
            cache=cache,
            skip_validation=skip_validation,
            verbose=verbose,
        )
        if status_callback is not None:
            status_callback("starting graph index")
        completed = self.runner(
            command,
            cwd=self.paths.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_graphrag_subprocess_env(),
        )
        _emit_progress_lines(completed.stdout or "", status_callback)
        _emit_progress_lines(completed.stderr or "", status_callback)
        return _completed_process_result(command, self.paths.root, completed)

    def query(
        self,
        *,
        workspace_dir: Path,
        question: str,
        method: str,
        data_dir: Path | None,
        community_level: int | None,
        dynamic_community_selection: bool | None,
        response_type: str | None,
        streaming: bool | None,
        verbose: bool,
    ) -> GraphRAGCommandResult:
        command = _query_command(
            workspace_dir,
            question,
            method=method,
            data_dir=data_dir,
            community_level=community_level,
            dynamic_community_selection=dynamic_community_selection,
            response_type=response_type,
            streaming=streaming,
            verbose=verbose,
        )
        completed = self.runner(
            command,
            cwd=self.paths.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_graphrag_subprocess_env(),
        )
        return _completed_process_result(command, self.paths.root, completed)


class SubprocessGraphRAGApiBackend:
    """Compatibility backend that shells out to GraphRAG's documented CLI."""

    def __init__(self, paths: ProjectPaths, runner: Runner) -> None:
        self.paths = paths
        self.runner = runner

    def init_workspace(
        self,
        *,
        workspace_dir: Path,
        model: str,
        embedding: str,
        force: bool,
    ) -> GraphRAGCommandResult:
        command = _cli_init_command(workspace_dir, model, embedding, force)
        completed = self.runner(
            command,
            cwd=self.paths.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_graphrag_subprocess_env(),
        )
        return _completed_process_result(command, self.paths.root, completed)

    def index(
        self,
        *,
        workspace_dir: Path,
        method: str,
        dry_run: bool,
        cache: bool,
        skip_validation: bool,
        verbose: bool,
        status_callback: StatusCallback,
    ) -> GraphRAGCommandResult:
        command = _cli_index_command(
            workspace_dir,
            method=method,
            dry_run=dry_run,
            cache=cache,
            skip_validation=skip_validation,
            verbose=verbose,
        )
        if status_callback is not None:
            status_callback("starting graph index")
        completed = self.runner(
            command,
            cwd=self.paths.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_graphrag_subprocess_env(),
        )
        _emit_progress_lines(completed.stdout or "", status_callback)
        _emit_progress_lines(completed.stderr or "", status_callback)
        return _completed_process_result(command, self.paths.root, completed)

    def query(
        self,
        *,
        workspace_dir: Path,
        question: str,
        method: str,
        data_dir: Path | None,
        community_level: int | None,
        dynamic_community_selection: bool | None,
        response_type: str | None,
        streaming: bool | None,
        verbose: bool,
    ) -> GraphRAGCommandResult:
        command = _cli_query_command(
            workspace_dir,
            question,
            method=method,
            data_dir=data_dir,
            community_level=community_level,
            dynamic_community_selection=dynamic_community_selection,
            response_type=response_type,
            streaming=streaming,
            verbose=verbose,
        )
        completed = self.runner(
            command,
            cwd=self.paths.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_graphrag_subprocess_env(),
        )
        return _completed_process_result(command, self.paths.root, completed)


def _run_index_entrypoint(
    *,
    workspace_dir: Path,
    method: str,
    dry_run: bool,
    cache: bool,
    skip_validation: bool,
    verbose: bool,
) -> None:  # pragma: no cover - thin wrapper around GraphRAG package
    from graphrag.cli.index import index_cli, update_cli
    from graphrag.config.enums import IndexingMethod

    index_method, is_update_run = _split_index_method(method)
    enum_method = IndexingMethod(index_method)
    kwargs = {
        "root_dir": workspace_dir,
        "method": enum_method,
        "verbose": verbose,
        "memprofile": False,
        "cache": cache,
        "config_filepath": None,
        "dry_run": dry_run,
        "skip_validation": skip_validation,
        "output_dir": None,
    }
    if is_update_run:
        _call_graphrag_entrypoint(
            update_cli,
            "graphrag.cli.index.update_cli",
            **kwargs,
        )
        return
    _call_graphrag_entrypoint(
        index_cli,
        "graphrag.cli.index.index_cli",
        **kwargs,
    )


def _run_query_entrypoint(
    *,
    workspace_dir: Path,
    data_dir: Path | None,
    method: str,
    community_level: int | None,
    dynamic_community_selection: bool | None,
    response_type: str | None,
    streaming: bool | None,
    question: str,
    verbose: bool,
) -> Any:  # pragma: no cover - thin wrapper around GraphRAG package
    from graphrag.cli.query import (
        run_basic_search,
        run_drift_search,
        run_global_search,
        run_local_search,
    )

    kwargs: dict[str, Any] = {
        "config_filepath": None,
        "data_dir": data_dir,
        "root_dir": workspace_dir,
        "query": question,
        "verbose": verbose,
    }
    if community_level is not None:
        kwargs["community_level"] = community_level
    elif method in {"global", "local", "drift"}:
        kwargs["community_level"] = DEFAULT_QUERY_COMMUNITY_LEVEL
    if dynamic_community_selection is not None:
        kwargs["dynamic_community_selection"] = dynamic_community_selection
    if response_type is not None:
        kwargs["response_type"] = response_type
    elif method in {"global", "local", "drift"}:
        kwargs["response_type"] = DEFAULT_QUERY_RESPONSE_TYPE
    if streaming is not None:
        kwargs["streaming"] = streaming
    if method == "global":
        return _call_graphrag_entrypoint(
            run_global_search,
            "graphrag.cli.query.run_global_search",
            **kwargs,
        )
    elif method == "local":
        return _call_graphrag_entrypoint(
            run_local_search,
            "graphrag.cli.query.run_local_search",
            **kwargs,
        )
    elif method == "drift":
        return _call_graphrag_entrypoint(
            run_drift_search,
            "graphrag.cli.query.run_drift_search",
            **kwargs,
        )
    elif method == "basic":
        return _call_graphrag_entrypoint(
            run_basic_search,
            "graphrag.cli.query.run_basic_search",
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid GraphRAG query method: {method}")


def _api_result(
    command: tuple[str, ...],
    cwd: Path,
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> GraphRAGCommandResult:
    return GraphRAGCommandResult(
        command=command,
        cwd=cwd,
        returncode=returncode,
        stdout=stdout,
        stderr=_sanitize_stderr(command, stderr),
    )


def _call_graphrag_entrypoint(
    func: Callable[..., Any],
    name: str,
    **kwargs: Any,
) -> Any:
    signature = inspect.signature(func)
    parameters = signature.parameters
    has_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    call_kwargs: dict[str, Any] = dict(kwargs) if has_var_kwargs else {}
    missing_required: list[str] = []
    for parameter in parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if parameter.name in kwargs:
            call_kwargs[parameter.name] = kwargs[parameter.name]
            continue
        if parameter.name in _KNOWN_GRAPHRAG_ENTRYPOINT_DEFAULTS:
            call_kwargs[parameter.name] = _KNOWN_GRAPHRAG_ENTRYPOINT_DEFAULTS[
                parameter.name
            ]
            continue
        if parameter.default is inspect.Parameter.empty:
            missing_required.append(parameter.name)
    if missing_required:
        raise GraphRAGCompatibilityError(
            f"{name} has unsupported required parameter(s): "
            f"{', '.join(sorted(missing_required))}."
        )
    return func(**call_kwargs)


def _query_return_to_text(value: Any) -> str:
    if isinstance(value, tuple | list):
        if not value:
            return ""
        return _query_return_to_text(value[0])
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text.strip()
    if value is None:
        return ""
    return str(value).strip()


def _completed_process_result(
    command: tuple[str, ...],
    cwd: Path,
    completed: subprocess.CompletedProcess[str],
) -> GraphRAGCommandResult:
    return _api_result(
        command,
        cwd,
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


class _ProgressCapture(io.StringIO):
    """Capture GraphRAG output and surface status-worthy lines as callbacks."""

    def __init__(self, status_callback: StatusCallback = None) -> None:
        super().__init__()
        self.status_callback = status_callback
        self._pending_line = ""

    def write(self, value: str) -> int:
        written = super().write(value)
        if self.status_callback is not None:
            self._pending_line = _emit_progress_chunks(
                self._pending_line,
                value,
                self.status_callback,
            )
        return written

    def flush(self) -> None:
        if self.status_callback is not None and self._pending_line.strip():
            _emit_progress_line(self._pending_line, self.status_callback)
            self._pending_line = ""
        super().flush()


def _emit_progress_lines(value: str, status_callback: StatusCallback) -> None:
    if status_callback is None:
        return
    pending_line = _emit_progress_chunks("", value, status_callback)
    if pending_line.strip():
        _emit_progress_line(pending_line, status_callback)


def _emit_progress_chunks(
    pending_line: str,
    value: str,
    status_callback: Callable[[str], None],
) -> str:
    buffer = pending_line + value.replace("\r", "\n")
    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        _emit_progress_line(line, status_callback)
    return buffer


def _emit_progress_line(line: str, status_callback: Callable[[str], None]) -> None:
    label = _extract_progress_label(line)
    if label:
        status_callback(label)


def _append_error(stderr: str, detail: str) -> str:
    if not stderr:
        return detail
    if not detail:
        return stderr
    return f"{stderr.rstrip()}\n{detail}"


def _init_command(
    workspace_dir: Path,
    model: str,
    embedding: str,
    force: bool,
) -> tuple[str, ...]:
    args = [
        "graphrag.api",
        "initialize_project_at",
        "--root",
        str(workspace_dir),
        "--model",
        model,
        "--embedding",
        embedding,
    ]
    if force:
        args.append("--force")
    return tuple(args)


def _index_command(
    workspace_dir: Path,
    *,
    method: str,
    dry_run: bool,
    cache: bool,
    skip_validation: bool,
    verbose: bool,
) -> tuple[str, ...]:
    args = [
        "graphrag.api",
        "build_index",
        "--root",
        str(workspace_dir),
        "--method",
        method,
    ]
    if dry_run:
        args.append("--dry-run")
    if not cache:
        args.append("--no-cache")
    if skip_validation:
        args.append("--skip-validation")
    if verbose:
        args.append("--verbose")
    return tuple(args)


def _query_command(
    workspace_dir: Path,
    question: str,
    *,
    method: str,
    data_dir: Path | None,
    community_level: int | None,
    dynamic_community_selection: bool | None,
    response_type: str | None,
    streaming: bool | None,
    verbose: bool,
) -> tuple[str, ...]:
    args = [
        "graphrag.api",
        f"{method}_search",
        "--root",
        str(workspace_dir),
        "--method",
        method,
    ]
    if data_dir is not None:
        args.extend(["--data", str(data_dir)])
    if community_level is not None:
        args.extend(["--community-level", str(community_level)])
    if dynamic_community_selection is True:
        args.append("--dynamic-community-selection")
    elif dynamic_community_selection is False:
        args.append("--no-dynamic-selection")
    if response_type:
        args.extend(["--response-type", response_type])
    if streaming is True:
        args.append("--streaming")
    elif streaming is False:
        args.append("--no-streaming")
    if verbose:
        args.append("--verbose")
    args.append(question)
    return tuple(args)


def _cli_init_command(
    workspace_dir: Path,
    model: str,
    embedding: str,
    force: bool,
) -> tuple[str, ...]:
    args = [
        "graphrag",
        "init",
        "--root",
        str(workspace_dir),
        "--model",
        model,
        "--embedding",
        embedding,
    ]
    if force:
        args.append("--force")
    return tuple(args)


def _cli_index_command(
    workspace_dir: Path,
    *,
    method: str,
    dry_run: bool,
    cache: bool,
    skip_validation: bool,
    verbose: bool,
) -> tuple[str, ...]:
    args = [
        "graphrag",
        "index",
        "--root",
        str(workspace_dir),
        "--method",
        method,
    ]
    if dry_run:
        args.append("--dry-run")
    if not cache:
        args.append("--no-cache")
    if skip_validation:
        args.append("--skip-validation")
    if verbose:
        args.append("--verbose")
    return tuple(args)


def _cli_query_command(
    workspace_dir: Path,
    question: str,
    *,
    method: str,
    data_dir: Path | None,
    community_level: int | None,
    dynamic_community_selection: bool | None,
    response_type: str | None,
    streaming: bool | None,
    verbose: bool,
) -> tuple[str, ...]:
    args = [
        "graphrag",
        "query",
        "--root",
        str(workspace_dir),
        "--method",
        method,
    ]
    if data_dir is not None:
        args.extend(["--data", str(data_dir)])
    if community_level is not None:
        args.extend(["--community-level", str(community_level)])
    if dynamic_community_selection is True:
        args.append("--dynamic-community-selection")
    elif dynamic_community_selection is False:
        args.append("--no-dynamic-selection")
    if response_type:
        args.extend(["--response-type", response_type])
    if streaming is True:
        args.append("--streaming")
    elif streaming is False:
        args.append("--no-streaming")
    if verbose:
        args.append("--verbose")
    args.append(question)
    return tuple(args)


def _is_entrypoint_contract_error(message: str) -> bool:
    normalized = message.casefold()
    return any(
        marker in normalized
        for marker in (
            "cli entrypoints are unavailable",
            "unsupported required parameter",
            "missing expected parameter",
        )
    )


def _is_entrypoint_encoding_error(message: str) -> bool:
    normalized = message.casefold()
    return (
        "unicodedecodeerror" in normalized
        or "character maps to <undefined>" in normalized
    )


def _should_fallback_from_entrypoint(message: str) -> bool:
    return _is_entrypoint_contract_error(message) or _is_entrypoint_encoding_error(
        message
    )


def _graphrag_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    return env


def _split_index_method(method: str) -> tuple[str, bool]:
    if method.endswith("-update"):
        return method.removesuffix("-update"), True
    return method, False


def _system_exit_code(exc: SystemExit) -> int:
    code = exc.code
    if isinstance(code, bool):
        return int(code)
    if isinstance(code, int):
        return code
    if code is None:
        return 0
    if isinstance(code, str) and not code:
        return 0
    return 1


def _last_non_empty_line(value: str) -> str:
    for line in reversed(value.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _sanitize_stderr(command: tuple[str, ...], stderr: str) -> str:
    if "--dry-run" not in command:
        return stderr
    if _is_known_graphrag_dry_run_logging_error(stderr):
        return ""
    return stderr


def _extract_progress_label(line: str) -> str:
    """Extract a human-readable status fragment from a GraphRAG log line."""
    lowered = line.lower()
    if any(skip in lowered for skip in ("debug", "traceback", "warning:")):
        return ""
    for marker in ("running workflow:", "running verb:", "running step:"):
        idx = lowered.find(marker)
        if idx >= 0:
            return line[idx:].strip()
    if "%" in line and any(c.isdigit() for c in line):
        return line.strip()
    stripped = line.strip()
    if len(stripped) > MAX_PROGRESS_LABEL_LENGTH:
        return stripped[: MAX_PROGRESS_LABEL_LENGTH - 3] + "..."
    return stripped


def _is_known_graphrag_dry_run_logging_error(stderr: str) -> bool:
    stripped = stderr.strip()
    normalized = " ".join(stripped.split())
    return (
        stripped.startswith("--- Logging error ---")
        and "Dry run complete, exiting..." in stripped
        and "Arguments: (True,)" in stripped
        and ("logger.info" in normalized or "logging" in normalized.casefold())
    )
