"""Click command implementation for the kb ask command.

This module belongs to :mod:`graphwiki_kb.commands.ask` and keeps related
behavior close to the command, service, model, provider, storage, script,
or test surface that uses it.

``kb ask`` now defaults to the custom **WikiGraphRAG** backend because:

* On the 10-PDF capstone benchmark it produced provider-backed answers in
  3-6s vs GraphRAG's 5s-30min, with comparable citation count and 100%
  insufficient-evidence-behavior match.
* It runs without API keys (provider-free synthesis falls back gracefully).
* It is inspectable -- the whole point of the WikiGraphRAG comparator.

``--engine`` accepts a single value, a comma-separated list, or the
shortcut ``all`` (which expands to ``wikigraph,graphrag,legacy``). When
multiple engines are selected, the command runs each in turn and renders
their answers in clearly-labeled sections.
"""

from __future__ import annotations

from typing import Any

import click
from rich.markdown import Markdown as RichMarkdown

from graphwiki_kb.commands.common import (
    console,
    echo_bullet,
    echo_section,
    emit_json,
    live_status,
    require_initialized,
)
from graphwiki_kb.models.command_models import CommandContext, CommandSpec
from graphwiki_kb.providers import ProviderError
from graphwiki_kb.services.graph_ask_controller_service import GraphAskControllerError
from graphwiki_kb.services.graphrag_query_service import GraphRAGQueryError
from graphwiki_kb.services.query_router_service import GRAPH_ASK_METHODS
from graphwiki_kb.services.wikigraph_query_service import WikiGraphQueryError
from graphwiki_kb.wikigraph.models import QueryMethod, WikiGraphAnswer

SUMMARY = (
    "Ask the KB. Defaults to the WikiGraphRAG backend; use `--engine graphrag` "
    "for Microsoft GraphRAG, `--engine legacy` for the deprecated FTS path, or "
    "`--engine all` (or a comma-separated list) to compare side by side."
)

SINGLE_ENGINE_CHOICES = ("wikigraph", "graphrag", "legacy")
ALL_ENGINE_TOKEN = "all"
DEFAULT_ENGINE = "wikigraph"
WIKIGRAPH_METHODS = ("auto", "basic", "local", "global", "drift-lite")
ALL_METHOD_CHOICES = tuple(sorted(set(GRAPH_ASK_METHODS).union(WIKIGRAPH_METHODS)))
LEGACY_DEPRECATION_NOTE = (
    "Deprecated: SQLite FTS5 retrieval is legacy-only. "
    "GraphRAG and WikiGraphRAG are the active retrieval paths."
)


def build_spec(_: CommandContext | None = None) -> CommandSpec:
    """Build the command registry specification for this module."""
    return CommandSpec(name="ask", summary=SUMMARY)


def parse_engines(value: str) -> tuple[str, ...]:
    """Parse the ``--engine`` value into an ordered tuple of backend names.

    Accepts a single name, a comma-separated list, or the shortcut
    ``all``. Order is preserved (and de-duplicated) so users can control
    the section order in the rendered output.
    """
    raw = (value or "").strip().lower()
    if not raw:
        raise click.ClickException(
            "--engine requires a value. Choose one of: "
            f"{', '.join(SINGLE_ENGINE_CHOICES)}, {ALL_ENGINE_TOKEN}."
        )
    tokens = [item.strip() for item in raw.split(",") if item.strip()]
    if not tokens:
        raise click.ClickException("--engine list is empty.")
    expanded: list[str] = []
    for token in tokens:
        if token == ALL_ENGINE_TOKEN:
            for name in SINGLE_ENGINE_CHOICES:
                if name not in expanded:
                    expanded.append(name)
            continue
        if token not in SINGLE_ENGINE_CHOICES:
            raise click.ClickException(
                f"Unknown --engine {token!r}. Choose one of: "
                f"{', '.join(SINGLE_ENGINE_CHOICES)}, {ALL_ENGINE_TOKEN}."
            )
        if token not in expanded:
            expanded.append(token)
    return tuple(expanded)


def create_command() -> click.Command:
    """Create the Click command exposed by this module."""

    @click.command(name="ask", help=SUMMARY, short_help="Ask the KB.")
    @click.argument("question_terms", nargs=-1)
    @click.option(
        "--engine",
        type=str,
        default=DEFAULT_ENGINE,
        show_default=True,
        help=(
            "Backend(s) to query. Accepts a single value "
            "(wikigraph|graphrag|legacy), a comma-separated list, or 'all'. "
            "Default: wikigraph."
        ),
    )
    @click.option(
        "--method",
        type=click.Choice(ALL_METHOD_CHOICES),
        default="auto",
        show_default=True,
        help=(
            "Retrieval method. GraphRAG supports auto/basic/local/global/drift; "
            "WikiGraphRAG supports auto/basic/local/global/drift-lite; legacy "
            "ignores this flag."
        ),
    )
    @click.option(
        "--community-level",
        type=int,
        help="Forward GraphRAG's community level option.",
    )
    @click.option(
        "--dynamic-community-selection/--no-dynamic-selection",
        default=None,
        help="Forward GraphRAG dynamic community selection behavior.",
    )
    @click.option(
        "--response-type",
        help="Forward GraphRAG's response type option.",
    )
    @click.option("--limit", type=int, help="Deprecated; use GraphRAG routing instead.")
    @click.option(
        "--save",
        "save_answer",
        is_flag=True,
        help="Save the answer as an analysis page in the wiki.",
    )
    @click.option(
        "--save-as",
        "save_as_name",
        type=str,
        default=None,
        help=(
            "Save the answer as an analysis page with a custom slug. When "
            "multiple engines run, the slug is prefixed per engine."
        ),
    )
    @click.option(
        "--show-source-trace",
        "show_source_trace",
        is_flag=True,
        help="Print the graph source trace and routing metadata before the answer.",
    )
    @click.option(
        "--show-evidence",
        is_flag=True,
        hidden=True,
    )
    @click.option("--verbose", is_flag=True, help="Forward GraphRAG's verbose flag.")
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
    @click.pass_obj
    def command(
        command_context: CommandContext,
        question_terms: tuple[str, ...],
        engine: str,
        method: str,
        community_level: int | None,
        dynamic_community_selection: bool | None,
        response_type: str | None,
        limit: int | None,
        save_answer: bool,
        save_as_name: str | None,
        show_source_trace: bool,
        show_evidence: bool,
        verbose: bool,
        as_json: bool,
    ) -> None:
        """Run the ``kb ask`` command."""
        require_initialized(command_context)
        if not question_terms:
            raise click.ClickException(
                'Provide a question to answer, for example: kb ask "What changed?"'
            )
        if limit is not None:
            raise click.ClickException(
                "--limit is not supported by kb ask because retrieval is "
                "controlled by the selected backend."
            )
        if show_evidence and not as_json:
            console.print(
                "[yellow]--show-evidence is deprecated; use "
                "--show-source-trace.[/yellow]"
            )
        question = " ".join(question_terms).strip()
        engines = parse_engines(engine)
        for engine_name in engines:
            _validate_method_for_engine(engine_name, method)
        if "legacy" in engines and not as_json:
            console.print(f"[yellow]{LEGACY_DEPRECATION_NOTE}[/yellow]")

        results: dict[str, dict[str, Any]] = {}
        for engine_name in engines:
            slug_override = _engine_save_slug(save_as_name, engine_name, len(engines))
            if engine_name == "wikigraph":
                results[engine_name] = _run_wikigraph_ask(
                    command_context,
                    question,
                    method=method,
                    save_answer=save_answer,
                    save_as_name=slug_override,
                )
            elif engine_name == "graphrag":
                results[engine_name] = _run_graphrag_ask(
                    command_context,
                    question,
                    method=method,
                    community_level=community_level,
                    dynamic_community_selection=dynamic_community_selection,
                    response_type=response_type,
                    verbose=verbose,
                    save_answer=save_answer,
                    save_as_name=slug_override,
                )
            elif engine_name == "legacy":
                results[engine_name] = _run_legacy_ask(
                    command_context,
                    question,
                    save_answer=save_answer,
                    save_as_name=slug_override,
                )

        # When a single engine was selected and it failed, surface the
        # error as a ClickException so the exit code is non-zero -- matches
        # the pre-multi-engine UX.
        if len(engines) == 1:
            only = engines[0]
            payload = results[only]
            if not payload.get("ok", False):
                raise click.ClickException(
                    str(payload.get("error", f"{only} backend failed"))
                )

        if as_json:
            serializable = {
                name: _engine_payload_to_dict(name, payload)
                for name, payload in results.items()
            }
            if len(engines) == 1:
                emit_json(serializable[engines[0]])
            else:
                emit_json(
                    {
                        "question": question,
                        "engines": list(engines),
                        "results": serializable,
                    }
                )
            return

        for index, engine_name in enumerate(engines):
            if len(engines) > 1:
                if index > 0:
                    console.print("")
                echo_section(f"=== {engine_name.upper()} ===")
            _render_engine_result(
                engine_name,
                results[engine_name],
                show_source_trace=show_source_trace,
                show_evidence=show_evidence,
            )

    return command


# --------------------------------------------------------------------------- #
# Method validation                                                           #
# --------------------------------------------------------------------------- #


def _validate_method_for_engine(engine: str, method: str) -> None:
    """Raise a friendly error when ``--method`` is not valid for the engine.

    ``legacy`` ignores the method flag entirely.
    """
    if engine == "legacy":
        return
    if engine == "wikigraph":
        if method not in WIKIGRAPH_METHODS:
            raise click.ClickException(
                f"--method {method!r} is not valid for engine=wikigraph. "
                f"Choose one of: {', '.join(WIKIGRAPH_METHODS)}."
            )
        return
    if engine == "graphrag":
        if method not in GRAPH_ASK_METHODS:
            raise click.ClickException(
                f"--method {method!r} is not valid for engine=graphrag. "
                f"Choose one of: {', '.join(GRAPH_ASK_METHODS)}."
            )
        return
    raise click.ClickException(f"Unknown engine for method validation: {engine!r}")


# --------------------------------------------------------------------------- #
# Per-engine execution                                                        #
# --------------------------------------------------------------------------- #


def _engine_save_slug(
    save_as_name: str | None, engine: str, engine_count: int
) -> str | None:
    """Return a per-engine slug when saving the same question across engines."""
    if save_as_name is None:
        return None
    if engine_count <= 1:
        return save_as_name
    return f"{engine}-{save_as_name}"


def _run_wikigraph_ask(
    command_context: CommandContext,
    question: str,
    *,
    method: str,
    save_answer: bool,
    save_as_name: str | None,
) -> dict[str, Any]:
    query_service = command_context.services.wikigraph_query
    payload: dict[str, Any] = {"engine": "wikigraph"}
    try:
        with live_status("Querying WikiGraphRAG"):
            wikigraph_method: QueryMethod = method  # type: ignore[assignment]
            answer: WikiGraphAnswer = query_service.ask(
                question,
                method=wikigraph_method,
                save=save_answer,
                save_as=save_as_name,
            )
    except WikiGraphQueryError as exc:
        payload["ok"] = False
        payload["error"] = str(exc)
        payload["answer"] = ""
        return payload
    payload["ok"] = True
    payload["answer_obj"] = answer
    return payload


def _run_graphrag_ask(
    command_context: CommandContext,
    question: str,
    *,
    method: str,
    community_level: int | None,
    dynamic_community_selection: bool | None,
    response_type: str | None,
    verbose: bool,
    save_answer: bool,
    save_as_name: str | None,
) -> dict[str, Any]:
    controller = command_context.services.graph_ask_controller
    payload: dict[str, Any] = {"engine": "graphrag"}
    try:
        with live_status("Querying GraphRAG"):
            answer = controller.ask(
                question,
                method=method,
                community_level=community_level,
                dynamic_community_selection=dynamic_community_selection,
                response_type=response_type,
                streaming=None,
                verbose=verbose,
                save=save_answer,
                save_as=save_as_name,
            )
    except (GraphAskControllerError, GraphRAGQueryError) as exc:
        payload["ok"] = False
        payload["error"] = str(exc)
        payload["answer"] = ""
        return payload
    payload["ok"] = True
    payload["answer_obj"] = answer
    return payload


def _run_legacy_ask(
    command_context: CommandContext,
    question: str,
    *,
    save_answer: bool,
    save_as_name: str | None,
) -> dict[str, Any]:
    query_service = command_context.services.query
    payload: dict[str, Any] = {"engine": "legacy"}
    try:
        with live_status("Querying legacy FTS"):
            answer = query_service.answer_question(question)
    except ProviderError as exc:
        payload["ok"] = False
        payload["error"] = str(exc)
        payload["answer"] = ""
        return payload
    saved_path: str | None = None
    if save_answer and answer.mode != "no-matches":
        try:
            saved_path = query_service.save_answer(question, answer, slug=save_as_name)
        except Exception as exc:  # pragma: no cover - defensive
            payload["save_error"] = str(exc)
    payload["ok"] = True
    payload["answer_obj"] = answer
    payload["saved_path"] = saved_path
    return payload


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #


def _engine_payload_to_dict(engine: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Convert an internal engine payload into a JSON-friendly dict."""
    base: dict[str, Any] = {
        "engine": engine,
        "ok": payload.get("ok", False),
    }
    if payload.get("error"):
        base["error"] = payload["error"]
        base["answer"] = ""
        return base
    answer = payload.get("answer_obj")
    if engine == "wikigraph" and answer is not None:
        data = answer.model_dump()
        data["citation_count"] = len(answer.citations)
        base.update(data)
        return base
    if engine == "graphrag" and answer is not None:
        base.update(answer.to_dict())
        return base
    if engine == "legacy" and answer is not None:
        base["answer"] = answer.answer
        base["mode"] = answer.mode
        base["insufficient_evidence"] = answer.insufficient_evidence
        base["citation_count"] = len(answer.citations)
        base["citations"] = [
            {
                "title": c.title,
                "ref": c.citation_ref,
                "section": c.section,
            }
            for c in answer.citations
        ]
        base["saved_path"] = payload.get("saved_path")
        return base
    return base


def _render_engine_result(
    engine: str,
    payload: dict[str, Any],
    *,
    show_source_trace: bool,
    show_evidence: bool,
) -> None:
    if not payload.get("ok", False):
        console.print(f"[red]{engine}: {payload.get('error', 'unknown error')}[/red]")
        return
    if engine == "wikigraph":
        _render_wikigraph_answer(
            payload["answer_obj"], show_source_trace=show_source_trace
        )
        return
    if engine == "graphrag":
        _render_graphrag_answer(
            payload["answer_obj"],
            show_source_trace=show_source_trace or show_evidence,
        )
        return
    if engine == "legacy":
        _render_legacy_answer(
            payload["answer_obj"],
            saved_path=payload.get("saved_path"),
            show_source_trace=show_source_trace,
        )
        return


def _render_wikigraph_answer(
    answer: WikiGraphAnswer, *, show_source_trace: bool
) -> None:
    provider_mode = answer.provider_status.get("mode", "provider-free")
    console.print(
        f"[dim]\\[engine: wikigraph, method: {answer.method}, "
        f"provider: {provider_mode}][/dim]"
    )
    if answer.warnings:
        console.print("[yellow]Warnings: " + ", ".join(answer.warnings) + "[/yellow]")
    if show_source_trace:
        console.print("")
        echo_section("WikiGraphRAG Source Trace")
        seed_entities: list[str] = []
        communities: list[str] = []
        sub_questions: list[str] = []
        for step in answer.trace:
            if not isinstance(step, dict):
                continue
            seed_entities.extend(step.get("seed_entities") or [])
            communities.extend(step.get("communities") or [])
            sub_questions.extend(step.get("sub_questions") or [])
        if seed_entities:
            console.print("  Seed entities: " + ", ".join(dict.fromkeys(seed_entities)))
        if communities:
            console.print("  Communities: " + ", ".join(dict.fromkeys(communities)))
        if sub_questions:
            console.print("  Sub-questions:")
            for question_text in dict.fromkeys(sub_questions):
                echo_bullet(question_text)
        if answer.contexts:
            kind_counts: dict[str, int] = {}
            for ctx in answer.contexts:
                kind_counts[ctx.node_kind] = kind_counts.get(ctx.node_kind, 0) + 1
            kind_summary = ", ".join(
                f"{kind}={count}" for kind, count in sorted(kind_counts.items())
            )
            console.print(
                f"  Retrieved contexts: {len(answer.contexts)} "
                f"(by kind: {kind_summary})"
            )
            for ctx in answer.contexts:
                trace_label = ",".join(ctx.trace) if ctx.trace else "(none)"
                echo_bullet(f"{ctx.title} [{ctx.citation_ref}] trace={trace_label}")
        if answer.provider_status:
            console.print(
                "  Provider: "
                + ", ".join(
                    f"{key}={value}"
                    for key, value in answer.provider_status.items()
                    if value not in (None, "")
                )
            )
    console.print("")
    console.print(RichMarkdown(answer.answer or "No answer text returned."))
    if answer.contexts and not show_source_trace:
        console.print("")
        echo_section("Contexts")
        for ctx in answer.contexts:
            echo_bullet(f"{ctx.title} [{ctx.citation_ref}] (score={ctx.score:.3f})")
    if answer.saved_path:
        console.print(f"\nSaved analysis page: {answer.saved_path}")


def _render_graphrag_answer(answer: Any, *, show_source_trace: bool) -> None:
    console.print(
        f"[dim]\\[retriever: {answer.retriever}, method: {answer.method}, "
        f"planner: {answer.planner or 'none'}, "
        f"index_run_id: {answer.index_run_id or 'unknown'}][/dim]"
    )
    if answer.route_reason:
        route_detail = answer.route_reason
        if answer.route_confidence:
            route_detail += f" ({answer.route_confidence})"
        console.print(f"[dim]Route: {route_detail}[/dim]")
    if show_source_trace:
        console.print("")
        console.print("Source Trace")
        console.print(f"  GraphRAG input: {answer.source_trace.get('input_path')}")
        console.print(f"  GraphRAG output: {answer.source_trace.get('output_dir')}")
        console.print(f"  Route reason: {answer.route_reason or 'unknown'}")
        console.print(f"  Route confidence: {answer.route_confidence or 'unknown'}")
        if answer.route_matched_terms:
            console.print(
                "  Route matched terms: " + ", ".join(answer.route_matched_terms)
            )
        console.print(
            "  Parsed graph references: " f"{len(answer.graph_data_references)}"
        )
        console.print(f"  Support level: {answer.claim_support or 'unverified'}")
    if answer.staleness_warnings:
        console.print("")
        for warning in answer.staleness_warnings:
            console.print(f"[yellow]{warning}[/yellow]")
    console.print("")
    console.print(RichMarkdown(answer.answer or "No answer text returned."))
    if answer.saved_path:
        console.print(f"\nSaved analysis page: {answer.saved_path}")


def _render_legacy_answer(
    answer: Any, *, saved_path: str | None, show_source_trace: bool
) -> None:
    console.print(
        f"[dim]\\[engine: legacy, mode: {answer.mode or 'unknown'}, "
        f"insufficient_evidence: {answer.insufficient_evidence}][/dim]"
    )
    if show_source_trace and answer.citations:
        console.print("")
        echo_section("Evidence")
        for citation in answer.citations:
            echo_bullet(f"{citation.title} [{citation.citation_ref}]")
    console.print("")
    console.print(RichMarkdown(answer.answer or "No answer text returned."))
    if answer.citations and not show_source_trace:
        console.print("")
        echo_section("Citations")
        for citation in answer.citations:
            line = f"{citation.title} [{citation.citation_ref}]"
            if citation.section and citation.section != citation.title:
                line += f" - {citation.section}"
            echo_bullet(line)
    if saved_path:
        console.print(f"\nSaved analysis page: {saved_path}")
