"""Tests for deterministic wiki-priors artifact helpers."""

from __future__ import annotations

import json

from graphwiki_kb.models.source_models import RawSourceRecord
from graphwiki_kb.services.wiki_priors_service import (
    WikiPriorsResult,
    WikiPriorsService,
    _looks_like_alias,
    _looks_like_term,
    _SourceText,
    _term_count,
)


def _source_record(
    *,
    source_id: str = "src-rag",
    slug: str = "rag",
    title: str = "Graph RAG",
    normalized_path: str | None = "raw/normalized/rag.md",
    content_hash: str = "hash-rag",
) -> RawSourceRecord:
    return RawSourceRecord(
        source_id=source_id,
        slug=slug,
        title=title,
        origin=f"C:/sources/{slug}.pdf",
        source_type="file",
        raw_path=f"raw/sources/{slug}.pdf",
        normalized_path=normalized_path,
        content_hash=content_hash,
        origin_hash=f"origin-{content_hash}",
        ingested_at="2026-05-17T00:00:00+00:00",
        metadata={"converter": "test"},
    )


def test_term_count_uses_case_insensitive_boundaries() -> None:
    """Term counting should preserve exact term text with word boundaries."""
    text = "Graph RAG, graph rag. Paragraph RAGged graph ragged."

    assert _term_count(text, "Graph RAG") == 2
    assert _term_count(text, "RAG") == 2
    assert _term_count(text, "graph ragged") == 1


def test_result_counts_tolerate_missing_or_malformed_payloads(tmp_path) -> None:
    """Count properties should be defensive around absent artifact payloads."""
    missing = WikiPriorsResult(
        artifact_path=tmp_path / "wiki_priors.json",
        artifact=None,
        artifact_digest=None,
        enabled=True,
    )
    malformed = WikiPriorsResult(
        artifact_path=tmp_path / "wiki_priors.json",
        artifact={"glossary": {}, "entity_types": "concept"},
        artifact_digest="sha256:test",
        enabled=True,
    )
    populated = WikiPriorsResult(
        artifact_path=tmp_path / "wiki_priors.json",
        artifact={"glossary": [{}, {}], "entity_types": ["concept"]},
        artifact_digest="sha256:test",
        enabled=True,
    )

    assert missing.glossary_count == 0
    assert missing.entity_type_count == 0
    assert malformed.glossary_count == 0
    assert malformed.entity_type_count == 0
    assert populated.glossary_count == 2
    assert populated.entity_type_count == 1


def test_candidate_filters_reject_non_terms_and_bad_aliases() -> None:
    """Term and alias filters should reject generic or malformed hints."""
    assert _looks_like_term("") is False
    assert _looks_like_term("summary") is False
    assert _looks_like_term("1234") is False
    assert _looks_like_term("Source ID src-1") is False
    assert _looks_like_alias("A") is False
    assert _looks_like_alias("lowercase") is False


def test_glossary_counts_candidates_and_aliases_across_sources() -> None:
    """Glossary support counts include candidate aliases without changing output."""
    service = object.__new__(WikiPriorsService)
    source_texts = [
        _SourceText(
            source_id="s1",
            title="Graph RAG",
            text="Retrieval Augmented Generation (RAG) appears in Graph RAG systems.",
        ),
        _SourceText(
            source_id="s2",
            title="Dense Retrieval",
            text="Graph RAG uses RAG for corpus search.",
        ),
    ]

    rows = service._glossary(
        source_texts,
        max_terms=10,
        min_support_count=2,
    )

    rows_by_term = {str(row["term"]): row for row in rows}
    assert rows_by_term["Graph RAG"] == {
        "term": "Graph RAG",
        "aliases": [],
        "support_count": 2,
        "source_ids": ["s1", "s2"],
    }
    assert rows_by_term["Retrieval Augmented Generation"]["aliases"] == ["RAG"]
    assert rows_by_term["Retrieval Augmented Generation"]["source_ids"] == [
        "s1",
        "s2",
    ]


def test_candidate_collection_skips_invalid_alias_matches() -> None:
    """Bad parenthetical aliases should not create candidate aliases."""
    service = object.__new__(WikiPriorsService)

    candidates = service._candidate_terms(
        [_SourceText(source_id="s1", title="Graph RAG", text="bad alias (a)")]
    )

    assert set(candidates) == {"graph rag"}
    assert candidates["graph rag"].aliases == set()


def test_sync_writes_artifact_and_preserves_semantic_timestamp(test_project) -> None:
    """A repeated semantic match should not rewrite only because time changed."""
    test_project.config["graph"]["wiki_priors"] = {
        **test_project.config["graph"]["wiki_priors"],
        "max_glossary_terms": 5,
        "min_support_count": 1,
    }
    test_project.write_file(
        "wiki/sources/rag.md",
        "---\n"
        "title: Graph RAG\n"
        "summary: Graph retrieval augmented generation.\n"
        "---\n"
        "# Graph RAG\n\n"
        "Graph RAG connects graph retrieval and generation.\n",
    )
    test_project.write_file(
        "raw/normalized/rag.md",
        "# Graph RAG\n\nGraph RAG improves retrieval augmented generation.\n",
    )
    test_project.services["manifest"].save_source(_source_record())
    service = WikiPriorsService(
        test_project.paths,
        test_project.services["manifest"],
        config=test_project.config,
    )

    first = service.sync()
    generated_at = first.artifact["generated_at"] if first.artifact else None
    second = service.sync()

    assert first.enabled is True
    assert first.written is True
    assert first.artifact_digest is not None
    assert first.artifact_path.exists()
    assert second.written is False
    assert second.artifact is not None
    assert second.artifact["generated_at"] == generated_at
    persisted = json.loads(first.artifact_path.read_text(encoding="utf-8"))
    assert persisted["glossary"][0]["source_ids"] == ["src-rag"]


def test_sync_respects_disabled_and_preview_only(test_project) -> None:
    """Disabled priors should no-op; preview should build without writing."""
    disabled_config = {
        **test_project.config,
        "graph": {
            **test_project.config["graph"],
            "wiki_priors": {
                **test_project.config["graph"]["wiki_priors"],
                "enabled": False,
            },
        },
    }
    service = WikiPriorsService(
        test_project.paths,
        test_project.services["manifest"],
        config=disabled_config,
    )

    disabled = service.sync()

    assert disabled.enabled is False
    assert disabled.artifact is None
    assert disabled.artifact_digest is None
    assert disabled.artifact_path.exists() is False

    test_project.write_file("raw/normalized/rag.md", "Graph RAG preview text.")
    test_project.services["manifest"].save_source(_source_record())
    preview_service = WikiPriorsService(
        test_project.paths,
        test_project.services["manifest"],
        config=test_project.config,
    )

    preview = preview_service.sync(preview_only=True)

    assert preview.enabled is True
    assert preview.written is False
    assert preview.artifact is not None
    assert preview.artifact_path.exists() is False


def test_source_texts_reject_escape_paths_and_include_enabled_legacy_concepts(
    test_project,
) -> None:
    """Only safe normalized files and explicitly enabled concepts feed priors."""
    test_project.config["concepts"]["enabled"] = True
    test_project.config["graph"]["wiki_priors"] = {
        **test_project.config["graph"]["wiki_priors"],
        "include_legacy_concepts": True,
    }
    test_project.write_file("raw/normalized/safe.md", "Safe normalized text.")
    test_project.write_file("wiki/sources/safe.md", "# Safe Page\n\nSafe wiki text.")
    test_project.write_file(
        "wiki/concepts/graph-rag.md",
        "# Graph RAG Concept\n\nLegacy concept text.",
    )
    service = WikiPriorsService(
        test_project.paths,
        test_project.services["manifest"],
        config=test_project.config,
    )
    safe = _source_record(
        source_id="safe",
        slug="safe",
        title="Safe",
        normalized_path="raw/normalized/safe.md",
        content_hash="hash-safe",
    )
    unsafe = _source_record(
        source_id="unsafe",
        slug="unsafe",
        title="Unsafe",
        normalized_path="../outside.md",
        content_hash="hash-unsafe",
    )

    texts = service._source_texts([safe, unsafe])

    by_id = {text.source_id: text for text in texts}
    assert "Safe normalized text." in by_id["safe"].text
    assert "../outside.md" not in by_id["unsafe"].text
    assert by_id["concept:graph-rag"].title == "Graph Rag"


def test_preserve_generated_at_ignores_invalid_existing_artifacts(
    test_project,
) -> None:
    """Malformed or non-object artifacts should not affect the new payload."""
    service = WikiPriorsService(
        test_project.paths,
        test_project.services["manifest"],
        config=test_project.config,
    )
    artifact = {
        "schema_version": 1,
        "source_digest": "sha256:source",
        "wiki_compile_digest": "sha256:wiki",
        "generated_at": "new",
        "entity_types": [],
        "glossary": [],
    }

    service.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    service.artifact_path.write_text("{", encoding="utf-8")
    assert service._preserve_generated_at_when_semantically_equal(artifact) is artifact

    service.artifact_path.write_text("[]", encoding="utf-8")
    assert service._preserve_generated_at_when_semantically_equal(artifact) is artifact

    assert service._normalized_path(_source_record(normalized_path=None)) is None
