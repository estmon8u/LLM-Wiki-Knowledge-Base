"""Additional WikiGraphRAG and agent-tool coverage for CI thresholds."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import click
import pytest

from graphwiki_kb.commands.retrieval_engines import (
    normalize_wikigraph_method,
    run_wikigraph_ask,
    run_wikigraph_find,
)
from graphwiki_kb.services.research_service import project_wikigraph_ask_kb_output
from graphwiki_kb.services.wikigraph_index_service import WikiGraphIndexService
from graphwiki_kb.wikigraph.deps import require_networkx
from graphwiki_kb.wikigraph.lexical_index import LexicalIndex
from graphwiki_kb.wikigraph.markdown_parser import ParsedChunk
from graphwiki_kb.wikigraph.models import WikiGraphAnswer, WikiGraphRetrievedContext

pytest.importorskip("networkx")


def _chunk(
    chunk_id: str,
    *,
    title: str = "RAG",
    text: str = "retrieval augmented generation improves knowledge tasks",
) -> ParsedChunk:
    return ParsedChunk(
        chunk_id=chunk_id,
        page_path="wiki/sources/rag.md",
        page_kind="source_page",
        title=title,
        heading="Overview",
        text=text,
        source_id="rag",
    )


def test_lexical_simple_backend_when_bm25s_unavailable(tmp_path: Path) -> None:
    """Fall back to the pure-Python lexical scorer when bm25s is absent."""
    chunks = [
        _chunk("c1"),
        _chunk("c2", title="Other", text="unrelated vocabulary only"),
    ]
    with patch(
        "graphwiki_kb.wikigraph.lexical_index.try_import_bm25s", return_value=None
    ):
        index = LexicalIndex(backend="bm25s", chunks=chunks, index_dir=tmp_path / "lex")
    assert index.backend == "simple"
    hits = index.search("retrieval generation", limit=5)
    assert hits
    assert hits[0].chunk_id == "c1"
    index.save()
    assert (tmp_path / "lex" / "index_meta.json").exists()


def test_lexical_empty_corpus_returns_no_hits(tmp_path: Path) -> None:
    index = LexicalIndex(backend="simple", chunks=[], index_dir=tmp_path / "empty")
    assert index.search("anything", limit=3) == []
    assert index.search("   ", limit=3) == []


def test_normalize_wikigraph_method_rejects_unknown() -> None:
    with pytest.raises(click.ClickException, match="Unsupported WikiGraphRAG method"):
        normalize_wikigraph_method("nope")


def test_run_wikigraph_find_prints_matched_entities(test_project, capsys) -> None:
    require_networkx()

    class _EntityFacade:
        def __init__(self, paths, config):
            pass

        def find(self, query, *, method, limit):
            return {
                "method": method,
                "matched_entities": [{"title": "RAG", "score": 0.95}],
                "contexts": [],
            }

    with patch(
        "graphwiki_kb.commands.retrieval_engines.WikiGraphQueryFacade",
        _EntityFacade,
    ):
        run_wikigraph_find(
            test_project.command_context,
            "RAG",
            method="local",
            limit=3,
            as_json=False,
        )
    assert "Matched entities" in capsys.readouterr().out


def test_run_wikigraph_find_empty_context_table(test_project, capsys) -> None:
    require_networkx()

    class _EmptyFacade:
        def __init__(self, paths, config):
            pass

        def find(self, query, *, method, limit):
            return {
                "method": method,
                "matched_entities": [],
                "contexts": [],
            }

    with patch(
        "graphwiki_kb.commands.retrieval_engines.WikiGraphQueryFacade",
        _EmptyFacade,
    ):
        run_wikigraph_find(
            test_project.command_context,
            "nothing",
            method="local",
            limit=3,
            as_json=False,
        )
    assert "No WikiGraphRAG contexts matched" in capsys.readouterr().out


def test_run_wikigraph_find_renders_entity_rows(test_project, capsys) -> None:
    require_networkx()
    test_project.write_file(
        "wiki/sources/realm.md",
        """---
title: REALM
type: source
source_id: realm
summary: REALM retrieval memory.
---
# REALM

## Overview

REALM uses retrieval during pretraining.
""",
    )
    WikiGraphIndexService(test_project.paths, test_project.config).build()
    run_wikigraph_find(
        test_project.command_context,
        "REALM",
        method="local",
        limit=5,
        as_json=False,
    )
    captured = capsys.readouterr()
    assert "WikiGraphRAG Contexts" in captured.out or "Matched entities" in captured.out


def test_run_wikigraph_find_import_error(test_project) -> None:
    class _BoomFacade:
        def __init__(self, paths, config):
            pass

        def find(self, *args, **kwargs):
            raise ImportError("networkx missing")

    with patch(
        "graphwiki_kb.commands.retrieval_engines.WikiGraphQueryFacade",
        _BoomFacade,
    ):
        with pytest.raises(click.ClickException, match="Install extras"):
            run_wikigraph_find(
                test_project.command_context,
                "q",
                method="local",
                limit=3,
                as_json=True,
            )


def test_run_wikigraph_ask_missing_index(test_project) -> None:
    require_networkx()
    with pytest.raises(click.ClickException, match="kb update"):
        run_wikigraph_ask(
            test_project.command_context,
            "What is REALM?",
            method="local",
            save_answer=False,
        )


def test_run_wikigraph_ask_import_error(test_project) -> None:
    class _BoomFacade:
        def __init__(self, paths, config, provider=None):
            pass

        def ask(self, *args, **kwargs):
            raise ImportError("networkx missing")

    with patch(
        "graphwiki_kb.commands.retrieval_engines.WikiGraphQueryFacade",
        _BoomFacade,
    ):
        with pytest.raises(click.ClickException, match="Install extras"):
            run_wikigraph_ask(
                test_project.command_context,
                "q",
                method="local",
                save_answer=False,
            )


def test_kb_ask_wikigraph_rejects_graphrag_flags(test_project) -> None:
    from graphwiki_kb.commands.ask import create_command

    require_networkx()
    test_project.write_file(
        "wiki/sources/rag.md",
        """---
title: RAG
type: source
source_id: rag
summary: RAG.
---
# RAG

## Overview

Retrieval augmented generation.
""",
    )
    WikiGraphIndexService(test_project.paths, test_project.config).build()
    from click.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(
        create_command(),
        [
            "What is RAG?",
            "--engine",
            "wikigraph",
            "--community-level",
            "2",
        ],
        obj=test_project.command_context,
    )
    assert result.exit_code != 0
    assert "GraphRAG-only flags" in result.output


def test_kb_ask_wikigraph_json_and_save(test_project) -> None:
    from graphwiki_kb.commands.ask import create_command
    from click.testing import CliRunner
    import json

    require_networkx()
    test_project.write_file(
        "wiki/sources/rag.md",
        """---
title: RAG
type: source
source_id: rag
summary: RAG.
---
# RAG

## Overview

Retrieval augmented generation.
""",
    )
    WikiGraphIndexService(test_project.paths, test_project.config).build()
    runner = CliRunner()
    result = runner.invoke(
        create_command(),
        [
            "What is RAG?",
            "--engine",
            "wikigraph",
            "--method",
            "local",
            "--json",
            "--save",
        ],
        obj=test_project.command_context,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.find("{") :])
    assert payload["engine"] == "wikigraph"
    assert list(test_project.paths.wiki_analysis_dir.glob("wikigraph-answer-*.md"))


def test_kb_ask_wikigraph_human_output(test_project) -> None:
    from graphwiki_kb.commands.ask import create_command
    from click.testing import CliRunner

    require_networkx()
    test_project.write_file(
        "wiki/sources/rag.md",
        """---
title: RAG
type: source
source_id: rag
summary: RAG.
---
# RAG

## Overview

Retrieval augmented generation.
""",
    )
    WikiGraphIndexService(test_project.paths, test_project.config).build()
    runner = CliRunner()
    result = runner.invoke(
        create_command(),
        ["What is RAG?", "--engine", "wikigraph", "--method", "local", "--verbose"],
        obj=test_project.command_context,
    )
    assert result.exit_code == 0, result.output
    assert "wikigraph" in result.output.lower()
    assert "ignored for WikiGraphRAG" in result.output


def test_update_service_wikigraph_import_error_allow_partial(test_project) -> None:
    from graphwiki_kb.services.update_service import UpdateOptions, UpdateService

    class _BoomIndex:
        def build(self, **kwargs):
            raise ImportError("networkx missing")

        def export_artifacts(self):
            return []

    service = UpdateService(
        ingest_service=test_project.services.ingest,
        compile_service=test_project.services.compile,
        concept_service=test_project.services.concepts,
        search_service=test_project.services.search,
        config=test_project.config,
        wikigraph_index_service=_BoomIndex(),
    )
    result = service._run_wikigraph_update(
        UpdateOptions(allow_partial=True),
    )
    assert result.skipped is True
    assert "WikiGraphRAG index skipped" in (result.warning or "")


def test_update_service_wikigraph_build_failure_allow_partial(test_project) -> None:
    from graphwiki_kb.services.update_service import UpdateOptions, UpdateService

    class _BoomIndex:
        def build(self, **kwargs):
            raise RuntimeError("disk full")

    service = UpdateService(
        ingest_service=test_project.services.ingest,
        compile_service=test_project.services.compile,
        concept_service=test_project.services.concepts,
        search_service=test_project.services.search,
        config=test_project.config,
        wikigraph_index_service=_BoomIndex(),
    )
    result = service._run_wikigraph_update(UpdateOptions(allow_partial=True))
    assert result.skipped is True
    assert "build failed" in (result.warning or "")


def test_update_service_wikigraph_build_and_export_callbacks(test_project) -> None:
    from graphwiki_kb.services.update_service import UpdateOptions, UpdateService

    require_networkx()
    test_project.write_file(
        "wiki/sources/rag.md",
        """---
title: RAG
type: source
source_id: rag
summary: RAG overview.
---
# RAG

## Overview

Retrieval augmented generation combines retrieval with generation.
""",
    )
    messages: list[str] = []
    service = UpdateService(
        ingest_service=test_project.services.ingest,
        compile_service=test_project.services.compile,
        concept_service=test_project.services.concepts,
        search_service=test_project.services.search,
        config=test_project.config,
        wikigraph_index_service=test_project.services.wikigraph_index,
    )
    result = service._run_wikigraph_update(
        UpdateOptions(export_wikigraph_artifacts=True),
        status_callback=messages.append,
    )
    assert result.build is not None
    assert result.exported_artifacts
    assert any("building WikiGraphRAG" in message for message in messages)
    assert any("exporting WikiGraphRAG" in message for message in messages)


def test_detect_communities_empty_graph_returns_no_communities() -> None:
    from graphwiki_kb.wikigraph.community_builder import detect_communities
    from graphwiki_kb.wikigraph.graph_store import WikiGraphStore

    require_networkx()
    assert detect_communities(WikiGraphStore(), []) == []


def test_detect_communities_rejects_unknown_algorithm(test_project) -> None:
    from graphwiki_kb.wikigraph.community_builder import detect_communities
    from graphwiki_kb.wikigraph.graph_store import WikiGraphStore
    from graphwiki_kb.wikigraph.models import WikiGraphNode

    require_networkx()
    store = WikiGraphStore()
    node = WikiGraphNode(
        id="n1",
        kind="entity",
        title="RAG",
        path="wiki/sources/rag.md",
        text="retrieval augmented generation",
    )
    store.add_node(node)
    with pytest.raises(ValueError, match="Unsupported community algorithm"):
        detect_communities(store, [node], algorithm="leiden")


def test_update_service_wikigraph_unavailable_service(test_project) -> None:
    from graphwiki_kb.services.update_service import UpdateOptions, UpdateService

    service = UpdateService(
        ingest_service=test_project.services.ingest,
        compile_service=test_project.services.compile,
        concept_service=test_project.services.concepts,
        search_service=test_project.services.search,
        config=test_project.config,
        wikigraph_index_service=None,
    )
    result = service._run_wikigraph_update(UpdateOptions())
    assert result.skipped is True
    assert "unavailable" in result.skip_reason


def test_run_wikigraph_ask_save_analysis_page(test_project) -> None:
    require_networkx()
    from tests.conftest import _StubProvider

    test_project.write_file(
        "wiki/sources/rag.md",
        """---
title: RAG
type: source
source_id: rag
summary: RAG overview.
---
# RAG

## Overview

Retrieval augmented generation combines retrieval with generation.
""",
    )
    WikiGraphIndexService(test_project.paths, test_project.config).build()
    test_project.services.query.provider = _StubProvider()
    answer = run_wikigraph_ask(
        test_project.command_context,
        "What is retrieval-augmented generation?",
        method="local",
        save_answer=True,
    )
    assert answer.answer
    analysis_dir = test_project.paths.wiki_analysis_dir
    saved = list(analysis_dir.glob("wikigraph-answer-*.md"))
    assert saved


def test_project_wikigraph_ask_kb_output_maps_fields() -> None:
    answer = WikiGraphAnswer(
        method="local",
        question="q",
        answer="RAG combines retrieval.",
        contexts=[
            WikiGraphRetrievedContext(
                node_id="n1",
                node_kind="chunk",
                title="RAG",
                path="wiki/sources/rag.md",
                text="body",
                score=0.9,
            )
        ],
        citations=[{"title": "RAG"}],
        trace=[{"step": "method", "value": "local"}],
        warnings=[],
    )
    projected = project_wikigraph_ask_kb_output(answer)
    assert projected.claim_support == "cited-graph-answer"
    assert projected.method == "local"
    assert projected.source_trace.get("method") == "local"

    empty = project_wikigraph_ask_kb_output(
        WikiGraphAnswer(
            method="auto",
            question="q",
            answer="",
            contexts=[],
            citations=[],
            trace=[],
        )
    )
    assert empty.claim_support == "no-answer"


def test_kb_update_command_prints_wikigraph_summary(test_project) -> None:
    from click.testing import CliRunner

    from graphwiki_kb.commands.update import create_command as create_update_command
    from tests.conftest import _StubProvider

    require_networkx()
    import yaml

    config = yaml.safe_load(test_project.paths.config_file.read_text(encoding="utf-8"))
    config["provider"] = {"name": "openai"}
    test_project.paths.config_file.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    test_project.config["provider"] = {"name": "openai"}
    test_project.write_file(
        "wiki/sources/rag.md",
        """---
title: RAG
type: source
source_id: rag
summary: RAG.
---
# RAG

## Overview

Retrieval augmented generation.
""",
    )
    runner = CliRunner()
    with patch(
        "graphwiki_kb.services.build_provider",
        return_value=_StubProvider(),
    ):
        result = runner.invoke(
            create_update_command(),
            ["--no-graph", "--export-wikigraph-artifacts"],
            obj=test_project.command_context,
        )
    assert result.exit_code == 0, result.output
    assert "WikiGraphRAG Summary" in result.output
    assert "nodes" in result.output
