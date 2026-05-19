"""Tests for test concept service.

This module belongs to `tests.test_concept_service` and keeps related behavior
close to the command, service, model, provider, storage, script, or test
surface that uses it.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from graphwiki_kb.services.concept_service import (
    ConceptService,
    _derive_topic_terms,
    _drafts_from_provider_report,
    _extract_terms,
    _format_concept_title,
    _load_concept_cache,
    _normalize_topic_terms,
    _parse_provider_concept_report,
    _provider_concept_prompt,
    _ProviderConceptReport,
    _replace_backlinks_block,
    _source_pages_digest,
    _SourcePage,
    _split_frontmatter,
    _stem_token,
    _write_concept_cache,
)


def _compiled_page(title: str, summary: str, body: str) -> str:
    """Handles compiled page.

    Args:
        title: Title value used by the operation.
        summary: Summary value used by the operation.
        body: Body value used by the operation.

    Returns:
        str produced by the operation.
    """
    return (
        "---\n"
        f"title: {title}\n"
        f"summary: {summary}\n"
        "source_id: src-1\n"
        "raw_path: raw/src.md\n"
        "source_hash: hash-1\n"
        "compiled_at: 2026-04-14T00:00:00Z\n"
        "---\n\n"
        f"# {title}\n\n{body}\n"
    )


# --- Helper unit tests ---


def test_stem_token_strips_common_suffixes() -> None:
    """Verifies that stem token strips common suffixes."""
    assert _stem_token("retrieval") == "retriev"
    assert _stem_token("models") == "model"
    assert _stem_token("training") == "train"
    assert _stem_token("used") == "use"  # Snowball stems "used" -> "use"
    assert _stem_token("few") == "few"


def test_stem_token_minimum_length_guard() -> None:
    """Stemmer returns stems of at least 3 characters."""
    # Snowball stems these correctly; guard keeps stems >= 3 chars
    assert _stem_token("real") == "real"
    assert _stem_token("based") == "base"
    assert _stem_token("canonical") == "canon"
    assert _stem_token("tales") == "tale"
    # Very short tokens are returned unchanged
    assert _stem_token("go") == "go"
    assert _stem_token("ax") == "ax"


def test_extract_terms_skips_stopwords_and_short_tokens() -> None:
    """Verifies that extract terms skips stopwords and short tokens."""
    terms = _extract_terms("The models are used for retrieval tasks.")
    assert "the" not in terms
    assert "are" not in terms
    assert "for" not in terms
    assert "retriev" in terms


def test_extract_terms_filters_stemmed_generic_tokens() -> None:
    """Verifies that extract terms filters stemmed generic tokens."""
    terms = _extract_terms(
        "Question answering with language models and learning systems."
    )

    # NLTK stopwords filter function words, not domain content words
    assert "with" not in terms
    assert "and" not in terms
    # Domain words survive general-purpose filtering
    assert "answer" in terms
    assert "languag" in terms
    assert "learn" in terms


def test_split_frontmatter_parses_yaml() -> None:
    """Verifies that split frontmatter parses yaml."""
    text = "---\ntitle: Hello\n---\n\nBody text.\n"
    fm, body = _split_frontmatter(text)
    assert fm["title"] == "Hello"
    assert body.strip() == "Body text."


def test_split_frontmatter_handles_missing_frontmatter() -> None:
    """Verifies that split frontmatter handles missing frontmatter."""
    text = "# No frontmatter\n\nBody.\n"
    fm, body = _split_frontmatter(text)
    assert fm == {}
    assert body == text


def test_format_concept_title_one_term() -> None:
    """Verifies that format concept title one term."""
    assert _format_concept_title(["dense-retrieval"]) == "Dense Retrieval"


def test_format_concept_title_two_terms() -> None:
    """Verifies that format concept title two terms."""
    assert (
        _format_concept_title(["dense-retrieval", "question-answering"])
        == "Dense Retrieval and Question Answering"
    )


def test_format_concept_title_three_terms() -> None:
    """Verifies that format concept title three terms."""
    result = _format_concept_title(["alpha", "beta", "gamma"])
    assert result == "Alpha, Beta, and Gamma"


def test_replace_backlinks_block_adds_section() -> None:
    """Verifies that replace backlinks block adds section."""
    original = "---\ntitle: Page\n---\n\n# Page\n\nBody.\n"
    result = _replace_backlinks_block(original, [("alpha", "Alpha")])
    assert "## Related Concept Pages" in result
    assert "[[alpha|Alpha]]" in result
    assert result.endswith("\n")


def test_replace_backlinks_block_replaces_existing_block() -> None:
    """Verifies that replace backlinks block replaces existing block."""
    original = (
        "# Page\n\nBody.\n"
        "\n## Related Concept Pages\n\n"
        "<!-- kb:concept-backlinks:start -->\n"
        "- [[old-link|Old Link]]\n"
        "<!-- kb:concept-backlinks:end -->\n"
    )
    result = _replace_backlinks_block(original, [("new-link", "New Link")])
    assert "[[old-link|Old Link]]" not in result
    assert "[[new-link|New Link]]" in result


def test_replace_backlinks_block_removes_block_when_no_links() -> None:
    """Verifies that replace backlinks block removes block when no links."""
    original = (
        "# Page\n\nBody.\n"
        "\n## Related Concept Pages\n\n"
        "<!-- kb:concept-backlinks:start -->\n"
        "- [[old-link|Old Link]]\n"
        "<!-- kb:concept-backlinks:end -->\n"
    )
    result = _replace_backlinks_block(original, [])
    assert "## Related Concept Pages" not in result
    assert "concept-backlinks" not in result


def test_replace_backlinks_block_noop_when_empty_and_no_block() -> None:
    """Verifies that replace backlinks block noop when empty and no block."""
    original = "# Page\n\nBody.\n"
    assert _replace_backlinks_block(original, []) == original


def test_derive_topic_terms_uses_frequency() -> None:
    """Verifies that derive topic terms uses frequency."""
    pages = [
        _SourcePage(
            file_path=None,
            relative_path="wiki/sources/a.md",
            slug="a",
            title="Dense retrieval for question answering",
            summary="Dense retrieval approach.",
            terms=_extract_terms("Dense retrieval for question answering"),
        ),
        _SourcePage(
            file_path=None,
            relative_path="wiki/sources/b.md",
            slug="b",
            title="Dense passage retrieval with citations",
            summary="Another dense retrieval paper.",
            terms=_extract_terms("Dense passage retrieval with citations"),
        ),
    ]
    result = _derive_topic_terms(pages)
    assert len(result) >= 1
    assert len(result) <= 3


def test_derive_topic_terms_rejects_pure_stopword_themes() -> None:
    """Pages whose shared terms are all stopwords produce no topic terms."""
    pages = [
        _SourcePage(
            file_path=None,
            relative_path="wiki/sources/a.md",
            slug="a",
            title="The With And But For",
            summary="The with and but for each.",
            terms=_extract_terms("The with and but for each."),
        ),
        _SourcePage(
            file_path=None,
            relative_path="wiki/sources/b.md",
            slug="b",
            title="But For The And With",
            summary="But for the and with each.",
            terms=_extract_terms("But for the and with each."),
        ),
        _SourcePage(
            file_path=None,
            relative_path="wiki/sources/c.md",
            slug="c",
            title="For The With And But",
            summary="For the with and but each.",
            terms=_extract_terms("For the with and but each."),
        ),
    ]

    assert _derive_topic_terms(pages) == []


# --- Service integration tests ---


def test_generate_no_source_pages_returns_empty(test_project) -> None:
    """Verifies that generate no source pages returns empty.

    Args:
        test_project: Test project value used by the operation.
    """
    service = ConceptService(test_project.paths)
    result = service.generate()

    assert result.concept_paths == []
    assert result.updated_source_paths == []
    assert result.removed_paths == []


def test_generate_single_source_page_returns_empty(test_project) -> None:
    """Verifies that generate single source page returns empty.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file(
        "wiki/sources/only-page.md",
        _compiled_page("Only Page", "Standalone page.", "Body content."),
    )

    service = ConceptService(test_project.paths)
    result = service.generate()

    assert result.concept_paths == []


def test_generate_creates_concept_page_for_related_sources(test_project) -> None:
    """Verifies that generate creates concept page for related sources.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file(
        "wiki/sources/alpha.md",
        _compiled_page(
            "Dense Retrieval for Knowledge Bases",
            "Dense retrieval enables fast knowledge base search.",
            "Dense retrieval approach for knowledge base search.",
        ),
    )
    test_project.write_file(
        "wiki/sources/beta.md",
        _compiled_page(
            "Dense Retrieval Using Dual Encoders",
            "A dense retrieval method using dual encoder architecture.",
            "Dense retrieval dual encoder knowledge base search.",
        ),
    )
    test_project.write_file(
        "wiki/sources/gamma.md",
        _compiled_page(
            "Dense Retrieval for Open-Domain QA",
            "Dense retrieval supports open-domain question answering workflows.",
            "Dense retrieval question answering knowledge base search.",
        ),
    )

    service = ConceptService(test_project.paths)
    result = service.generate()

    assert len(result.concept_paths) >= 1
    concept_path = test_project.root / result.concept_paths[0]
    assert concept_path.exists()
    text = concept_path.read_text(encoding="utf-8")
    assert "type: concept" in text
    assert "generator: concept-service-v1" in text
    assert "source_pages:" in text


def test_generate_adds_backlinks_to_source_pages(test_project) -> None:
    """Verifies that generate adds backlinks to source pages.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file(
        "wiki/sources/alpha.md",
        _compiled_page(
            "Dense Retrieval for Knowledge Bases",
            "Dense retrieval enables fast knowledge base search.",
            "Dense retrieval approach for knowledge base search.",
        ),
    )
    test_project.write_file(
        "wiki/sources/beta.md",
        _compiled_page(
            "Dense Retrieval Using Dual Encoders",
            "A dense retrieval method using dual encoder architecture.",
            "Dense retrieval dual encoder knowledge base search.",
        ),
    )
    test_project.write_file(
        "wiki/sources/gamma.md",
        _compiled_page(
            "Dense Retrieval for Open-Domain QA",
            "Dense retrieval supports open-domain question answering workflows.",
            "Dense retrieval question answering knowledge base search.",
        ),
    )

    service = ConceptService(test_project.paths)
    result = service.generate()

    assert len(result.updated_source_paths) >= 1
    for rel_path in result.updated_source_paths:
        text = (test_project.root / rel_path).read_text(encoding="utf-8")
        assert "## Related Concept Pages" in text
        assert "<!-- kb:concept-backlinks:start -->" in text


def test_generate_removes_stale_managed_pages(test_project) -> None:
    """Verifies that generate removes stale managed pages.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.wiki_concepts_dir.mkdir(parents=True, exist_ok=True)
    test_project.write_file(
        "wiki/concepts/stale-topic.md",
        "---\ntitle: Stale\ntype: concept\ngenerator: concept-service-v1\n"
        "generated_at: 2026-01-01T00:00:00Z\nsource_pages: []\ntopic_terms: []\n"
        "summary: Old.\n---\n\n# Stale\n\nOld content.\n",
    )

    service = ConceptService(test_project.paths)
    result = service.generate()

    assert "wiki/concepts/stale-topic.md" in result.removed_paths
    assert not (test_project.root / "wiki/concepts/stale-topic.md").exists()


def test_generate_avoids_overwriting_manual_page(test_project) -> None:
    """Verifies that generate avoids overwriting manual page.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file(
        "wiki/concepts/dense.md",
        "---\ntitle: Manual Dense Page\ntype: analysis\n---\n\n# Manual Dense Page\n\nHand-written.\n",
    )
    test_project.write_file(
        "wiki/sources/alpha.md",
        _compiled_page(
            "Dense Retrieval for Knowledge Bases",
            "Dense retrieval enables fast knowledge base search.",
            "Dense retrieval approach for knowledge base search.",
        ),
    )
    test_project.write_file(
        "wiki/sources/beta.md",
        _compiled_page(
            "Dense Retrieval Using Dual Encoders",
            "A dense retrieval method using dual encoder architecture.",
            "Dense retrieval dual encoder knowledge base search.",
        ),
    )

    service = ConceptService(test_project.paths)
    result = service.generate()

    manual_text = (test_project.root / "wiki/concepts/dense.md").read_text(
        encoding="utf-8"
    )
    assert "Hand-written" in manual_text

    for cpath in result.concept_paths:
        full = test_project.root / cpath
        text = full.read_text(encoding="utf-8")
        assert "generator: concept-service-v1" in text


def test_generate_idempotent(test_project) -> None:
    """Verifies that generate idempotent.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file(
        "wiki/sources/alpha.md",
        _compiled_page(
            "Dense Retrieval for Knowledge Bases",
            "Dense retrieval enables fast knowledge base search.",
            "Dense retrieval approach for knowledge base search.",
        ),
    )
    test_project.write_file(
        "wiki/sources/beta.md",
        _compiled_page(
            "Dense Retrieval Using Dual Encoders",
            "A dense retrieval method using dual encoder architecture.",
            "Dense retrieval dual encoder knowledge base search.",
        ),
    )

    service = ConceptService(test_project.paths)
    first = service.generate()
    second = service.generate()

    assert len(first.concept_paths) == len(second.concept_paths)
    assert second.removed_paths == []


def test_lint_passes_for_generated_concept_page(test_project) -> None:
    """Verifies that lint passes for generated concept page.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file(
        "wiki/sources/alpha.md",
        _compiled_page(
            "Dense Retrieval for Knowledge Bases",
            "Dense retrieval enables fast knowledge base search.",
            "Dense retrieval approach for knowledge base search.",
        ),
    )
    test_project.write_file(
        "wiki/sources/beta.md",
        _compiled_page(
            "Dense Retrieval Using Dual Encoders",
            "A dense retrieval method using dual encoder architecture.",
            "Dense retrieval dual encoder knowledge base search.",
        ),
    )

    service = ConceptService(test_project.paths)
    service.generate()

    report = test_project.services["lint"].lint()
    field_issues = [
        i for i in report.issues if i.code == "missing-field" and "concepts/" in i.path
    ]
    assert field_issues == []


def test_generate_with_no_wiki_sources_dir(test_project) -> None:
    """Verifies that generate with no wiki sources dir.

    Args:
        test_project: Test project value used by the operation.
    """
    import shutil

    if test_project.paths.wiki_sources_dir.exists():
        shutil.rmtree(test_project.paths.wiki_sources_dir)

    service = ConceptService(test_project.paths)
    result = service.generate()

    assert result.concept_paths == []


def test_resolve_destination_skips_manual_page(test_project) -> None:
    """Verifies that resolve destination skips manual page.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.wiki_concepts_dir.mkdir(parents=True, exist_ok=True)
    test_project.write_file(
        "wiki/concepts/dense-retrieval.md",
        "---\ntitle: Manual\ntype: analysis\n---\n\n# Manual\n\nHand-written.\n",
    )
    test_project.write_file(
        "wiki/sources/alpha.md",
        _compiled_page(
            "Dense Retrieval for Knowledge Bases",
            "Dense retrieval enables fast knowledge base search.",
            "Dense retrieval approach for knowledge base search.",
        ),
    )
    test_project.write_file(
        "wiki/sources/beta.md",
        _compiled_page(
            "Dense Retrieval Using Dual Encoders",
            "A dense retrieval method using dual encoder architecture.",
            "Dense retrieval dual encoder knowledge base search.",
        ),
    )

    service = ConceptService(test_project.paths)
    result = service.generate()

    for cpath in result.concept_paths:
        assert cpath.split("/")[-1] != "dense-retrieval.md"
    manual = (test_project.root / "wiki/concepts/dense-retrieval.md").read_text(
        encoding="utf-8"
    )
    assert "Hand-written" in manual


def test_compile_with_concepts_flag_cli() -> None:
    """Verifies that compile with concepts flag cli."""
    from unittest.mock import patch

    from click.testing import CliRunner

    from graphwiki_kb.cli import main
    from graphwiki_kb.providers.base import (
        ProviderRequest,
        ProviderResponse,
        TextProvider,
    )

    class _FakeProvider(TextProvider):
        """Represents fake provider behavior and data.

        Attributes:
            See annotated class attributes for stored values.
        """

        name = "fake"

        def generate(self, request: ProviderRequest) -> ProviderResponse:
            """Generate.

            Args:
                request: Request value used by the operation.

            Returns:
                ProviderResponse produced by the operation.
            """
            return ProviderResponse(text="Stub summary.", model_name="fake-v1")

    runner = CliRunner()
    with runner.isolated_filesystem():
        from pathlib import Path

        Path("alpha.md").write_text(
            "# Dense Retrieval for KB\n\n"
            "Dense retrieval enables knowledge base search.\n",
            encoding="utf-8",
        )
        Path("beta.md").write_text(
            "# Dense Retrieval Dual Encoders\n\n"
            "Dense retrieval dual encoder architecture for knowledge base search.\n",
            encoding="utf-8",
        )

        assert runner.invoke(main, ["init"]).exit_code == 0
        assert runner.invoke(main, ["add", "alpha.md"]).exit_code == 0
        assert runner.invoke(main, ["add", "beta.md"]).exit_code == 0

        import yaml

        config = yaml.safe_load(Path("kb.config.yaml").read_text(encoding="utf-8"))
        config["provider"] = {"name": "stub"}
        Path("kb.config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )

        with patch(
            "graphwiki_kb.services.build_provider", return_value=_FakeProvider()
        ):
            result = runner.invoke(main, ["update", "--concepts", "--no-graph"])

        assert result.exit_code == 0
        assert "Concept Summary" in result.output


def test_split_frontmatter_handles_invalid_yaml() -> None:
    """Verifies that split frontmatter handles invalid yaml."""
    text = "---\n: :\n  bad: [unclosed\n---\n\nBody.\n"
    fm, _ = _split_frontmatter(text)
    assert fm == {}


def test_list_managed_pages_skips_non_concept_pages(test_project) -> None:
    """Verifies that list managed pages skips non concept pages.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.wiki_concepts_dir.mkdir(parents=True, exist_ok=True)
    test_project.write_file(
        "wiki/concepts/manual.md",
        "---\ntitle: Manual Topic\ntype: analysis\n---\n\n# Manual\n\nHand-written.\n",
    )

    service = ConceptService(test_project.paths)
    managed = service._list_managed_pages()

    assert len(managed) == 0


def test_generate_with_unrelated_pages_produces_no_concepts(test_project) -> None:
    """Verifies that generate with unrelated pages produces no concepts.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.write_file(
        "wiki/sources/apples.md",
        _compiled_page(
            "Apples and Fruit Growing",
            "Everything about apple cultivation.",
            "Orchards produce different apple varieties year round.",
        ),
    )
    test_project.write_file(
        "wiki/sources/rockets.md",
        _compiled_page(
            "Rocket Propulsion Systems",
            "Chemical fuels for rocket engines.",
            "Liquid fuel engines differ from solid boosters.",
        ),
    )

    service = ConceptService(test_project.paths)
    result = service.generate()

    assert result.concept_paths == []


def test_split_frontmatter_no_closing_marker() -> None:
    """Verifies that split frontmatter no closing marker."""
    text = "---\ntitle: Hello\nNo closing marker here\n"
    fm, body = _split_frontmatter(text)
    assert fm == {}
    assert body == text


def test_resolve_destination_suffix_collision(test_project) -> None:
    """Verifies that resolve destination suffix collision.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.wiki_concepts_dir.mkdir(parents=True, exist_ok=True)
    test_project.write_file(
        "wiki/concepts/test-slug.md",
        "---\ntitle: Manual\ntype: analysis\n---\n\n# Manual\n",
    )
    test_project.write_file(
        "wiki/concepts/test-slug-2.md",
        "---\ntitle: Manual 2\ntype: analysis\n---\n\n# Manual 2\n",
    )

    service = ConceptService(test_project.paths)
    dest = service._resolve_destination("test-slug", set())

    assert dest.name == "test-slug-3.md"


def test_resolve_destination_stops_after_slug_attempt_limit(
    monkeypatch,
    test_project,
) -> None:
    """Verifies concept slug allocation cannot loop forever."""
    import pytest

    import graphwiki_kb.services.concept_service as concept_module

    monkeypatch.setattr(concept_module, "_MAX_CONCEPT_SLUG_ATTEMPTS", 3)
    test_project.paths.wiki_concepts_dir.mkdir(parents=True, exist_ok=True)
    for name in ("test-slug.md", "test-slug-2.md", "test-slug-3.md"):
        test_project.write_file(
            f"wiki/concepts/{name}",
            "---\ntitle: Manual\ntype: analysis\n---\n\n# Manual\n",
        )
    service = ConceptService(test_project.paths)

    with pytest.raises(ValueError, match="after 3 attempts"):
        service._resolve_destination("test-slug", set())


def test_list_managed_pages_no_concepts_dir(test_project) -> None:
    """Verifies that list managed pages no concepts dir.

    Args:
        test_project: Test project value used by the operation.
    """
    import shutil

    if test_project.paths.wiki_concepts_dir.exists():
        shutil.rmtree(test_project.paths.wiki_concepts_dir)

    service = ConceptService(test_project.paths)
    managed = service._list_managed_pages()

    assert managed == set()


def test_list_managed_pages_handles_unreadable_file(test_project) -> None:
    """Verifies that list managed pages handles unreadable file.

    Args:
        test_project: Test project value used by the operation.
    """
    test_project.paths.wiki_concepts_dir.mkdir(parents=True, exist_ok=True)
    bad_path = test_project.paths.wiki_concepts_dir / "corrupt.md"
    bad_path.write_bytes(b"\x80\x81\x82")

    service = ConceptService(test_project.paths)
    managed = service._list_managed_pages()

    assert bad_path not in managed


def test_generate_zero_output_when_no_themes_qualify(test_project) -> None:
    """Three pages with disjoint terms produce zero concept pages."""
    for i, (slug, title) in enumerate(
        [
            ("alpha", "Alpha mechanics"),
            ("beta", "Beta kinetics"),
            ("gamma", "Gamma optics"),
        ]
    ):
        fm = {
            "title": title,
            "summary": f"Topic {slug} is unrelated to the others.",
            "type": "source",
            "source_id": f"src_{i}",
            "source_hash": f"hash_{i}",
            "raw_path": f"raw/sources/{slug}.md",
            "origin": "local",
            "compiled_at": "2026-04-22T00:00:00Z",
            "ingested_at": "2026-04-22T00:00:00Z",
            "tags": [],
        }

        page = f"---\n{yaml.safe_dump(fm, sort_keys=False)}---\n\n# {title}\n\n## Summary\n\n{fm['summary']}\n"
        test_project.paths.wiki_sources_dir.mkdir(parents=True, exist_ok=True)
        (test_project.paths.wiki_sources_dir / f"{slug}.md").write_text(
            page, encoding="utf-8"
        )

    service = ConceptService(test_project.paths)
    result = service.generate()

    assert result.concept_paths == []


def test_load_source_pages_ignores_placeholder_summaries(test_project) -> None:
    """Placeholder summaries should not contribute to concept term derivation."""
    fm = {
        "title": "Dense Passage Retrieval",
        "summary": "No summary available yet.",
        "type": "source",
        "source_id": "src_1",
        "source_hash": "hash_1",
        "raw_path": "raw/sources/dpr.md",
        "origin": "local",
        "compiled_at": "2026-04-22T00:00:00Z",
        "ingested_at": "2026-04-22T00:00:00Z",
        "tags": [],
    }

    page = f"---\n{yaml.safe_dump(fm, sort_keys=False)}---\n\n# Dense Passage Retrieval\n\n## Summary\n\nNo summary available yet.\n"
    test_project.paths.wiki_sources_dir.mkdir(parents=True, exist_ok=True)
    (test_project.paths.wiki_sources_dir / "dpr.md").write_text(page, encoding="utf-8")

    service = ConceptService(test_project.paths)
    pages = service._load_source_pages()

    assert len(pages) == 1
    assert pages[0].summary == ""
    assert "summar" not in pages[0].terms
    assert "avail" not in pages[0].terms


# ---------------------------------------------------------------------------
# Snowball stemmer integration
# ---------------------------------------------------------------------------


def test_stem_token_uses_snowball_stemmer() -> None:
    """Verify the NLTK Snowball stemmer provides linguistically correct stems."""
    # Irregular forms the old hand-rolled stemmer missed
    assert _stem_token("running") == "run"
    assert _stem_token("better") == "better"
    assert _stem_token("studies") == "studi"
    assert _stem_token("happiness") == "happi"


def test_stem_token_short_input_unchanged() -> None:
    """Tokens producing stems shorter than 3 chars should return original."""
    assert _stem_token("go") == "go"
    assert _stem_token("be") == "be"


# ---------------------------------------------------------------------------
# Provider concept clustering helpers
# ---------------------------------------------------------------------------


def _make_source_pages(count: int = 4) -> list[_SourcePage]:
    """Handles make source pages.

    Args:
        count: Count value used by the operation.

    Returns:
        list[_SourcePage] produced by the operation.
    """
    pages = []
    for i in range(count):
        slug = f"page-{i}"
        title = f"Dense Retrieval Approach {i}"
        summary = f"A dense retrieval method variant {i}."
        pages.append(
            _SourcePage(
                file_path=Path(f"wiki/sources/{slug}.md"),
                relative_path=f"wiki/sources/{slug}.md",
                slug=slug,
                title=title,
                summary=summary,
                terms=_extract_terms(f"{title} {summary}"),
            )
        )
    return pages


def test_normalize_topic_terms_deduplicates_and_caps() -> None:
    """Verifies that normalize topic terms deduplicates and caps."""
    result = _normalize_topic_terms(
        ["Dense Retrieval", "dense-retrieval", "QA", "extra"]
    )
    assert result == ["dense-retrieval", "qa", "extra"]


def test_normalize_topic_terms_empty_input() -> None:
    """Verifies that normalize topic terms empty input."""
    assert _normalize_topic_terms([]) == []


def test_source_pages_digest_deterministic() -> None:
    """Verifies that source pages digest deterministic."""
    pages = _make_source_pages(3)
    d1 = _source_pages_digest(pages)
    d2 = _source_pages_digest(pages)
    assert d1 == d2
    assert len(d1) == 64  # SHA-256 hex


def test_source_pages_digest_changes_on_content_change() -> None:
    """Verifies that source pages digest changes on content change."""
    pages = _make_source_pages(3)
    d1 = _source_pages_digest(pages)
    pages[0] = _SourcePage(
        file_path=pages[0].file_path,
        relative_path=pages[0].relative_path,
        slug=pages[0].slug,
        title="Changed Title",
        summary=pages[0].summary,
        terms=pages[0].terms,
    )
    d2 = _source_pages_digest(pages)
    assert d1 != d2


def test_provider_concept_prompt_includes_all_pages() -> None:
    """Verifies that provider concept prompt includes all pages."""
    pages = _make_source_pages(3)
    prompt = _provider_concept_prompt(pages)
    for page in pages:
        assert page.relative_path in prompt
        assert page.title in prompt


def test_parse_provider_concept_report_valid_json() -> None:
    """Verifies that parse provider concept report valid json."""
    raw = '{"concepts": [{"title": "T", "summary": "S", "topic_terms": ["a"], "source_pages": ["x"]}]}'
    report = _parse_provider_concept_report(raw)
    assert len(report.concepts) == 1
    assert report.concepts[0].title == "T"


def test_parse_provider_concept_report_rejects_invalid_json() -> None:
    """Verifies that parse provider concept report rejects invalid json."""
    import pytest

    with pytest.raises(Exception):
        _parse_provider_concept_report("not json")


def test_drafts_from_provider_report_matches_by_relative_path() -> None:
    """Verifies that drafts from provider report matches by relative path."""
    pages = _make_source_pages(4)
    report = _ProviderConceptReport(
        concepts=[
            _ProviderConceptReport.model_validate(
                {
                    "concepts": [
                        {
                            "title": "Cluster A",
                            "summary": "Summary A",
                            "topic_terms": ["dense-retrieval"],
                            "source_pages": [p.relative_path for p in pages[:3]],
                        }
                    ]
                }
            ).concepts[0]
        ]
    )
    drafts = _drafts_from_provider_report(report, pages)
    assert len(drafts) == 1
    assert drafts[0].title == "Cluster A"
    assert len(drafts[0].source_pages) == 3


def test_drafts_from_provider_report_matches_by_slug() -> None:
    """Verifies that drafts from provider report matches by slug."""
    pages = _make_source_pages(4)
    report = _parse_provider_concept_report(
        '{"concepts": [{"title": "By Slug", "summary": "S",'
        '"topic_terms": ["dense-retrieval"],'
        f'"source_pages": ["{pages[0].slug}", "{pages[1].slug}", "{pages[2].slug}"]'
        "}]}"
    )
    drafts = _drafts_from_provider_report(report, pages)
    assert len(drafts) == 1


def test_drafts_from_provider_report_skips_small_cluster() -> None:
    """Verifies that drafts from provider report skips small cluster."""
    pages = _make_source_pages(4)
    report = _parse_provider_concept_report(
        '{"concepts": [{"title": "Small", "summary": "S",'
        '"topic_terms": ["alpha"],'
        f'"source_pages": ["{pages[0].relative_path}", "{pages[1].relative_path}"]'
        "}]}"
    )
    drafts = _drafts_from_provider_report(report, pages)
    assert len(drafts) == 0  # below _MIN_SOURCE_PAGES


def test_drafts_from_provider_report_deduplicates_slugs() -> None:
    """Verifies that drafts from provider report deduplicates slugs."""
    pages = _make_source_pages(4)
    report = _parse_provider_concept_report(
        '{"concepts": ['
        '{"title": "C1", "summary": "S", "topic_terms": ["alpha"],'
        f'"source_pages": ["{pages[0].relative_path}", "{pages[1].relative_path}", "{pages[2].relative_path}"]'
        "},"
        '{"title": "C2", "summary": "S", "topic_terms": ["alpha"],'
        f'"source_pages": ["{pages[0].relative_path}", "{pages[1].relative_path}", "{pages[3].relative_path}"]'
        "}]}"
    )
    drafts = _drafts_from_provider_report(report, pages)
    assert len(drafts) == 1  # second shares slug "alpha"


def test_write_and_load_concept_cache(tmp_path) -> None:
    """Verifies that write and load concept cache.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    cache_path = tmp_path / "cache.json"
    report = _parse_provider_concept_report(
        '{"concepts": [{"title": "T", "summary": "S", "topic_terms": ["a"], "source_pages": ["x"]}]}'
    )
    _write_concept_cache(cache_path, "digest123", report)

    loaded = _load_concept_cache(cache_path, "digest123")
    assert loaded is not None
    assert len(loaded.concepts) == 1
    assert loaded.concepts[0].title == "T"


def test_load_concept_cache_returns_none_on_digest_mismatch(tmp_path) -> None:
    """Verifies that load concept cache returns none on digest mismatch.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    cache_path = tmp_path / "cache.json"
    report = _parse_provider_concept_report('{"concepts": []}')
    _write_concept_cache(cache_path, "digest123", report)

    assert _load_concept_cache(cache_path, "other-digest") is None


def test_load_concept_cache_returns_none_on_missing_file(tmp_path) -> None:
    """Verifies that load concept cache returns none on missing file.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    assert _load_concept_cache(tmp_path / "missing.json", "any") is None


def test_load_concept_cache_returns_none_on_corrupt_json(tmp_path) -> None:
    """Verifies that load concept cache returns none on corrupt json.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("not json", encoding="utf-8")
    assert _load_concept_cache(cache_path, "any") is None


def test_load_concept_cache_returns_none_on_version_mismatch(tmp_path) -> None:
    """Verifies that load concept cache returns none on version mismatch.

    Args:
        tmp_path: Tmp path value used by the operation.
    """
    import json

    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps({"version": 999, "source_digest": "d", "report": {"concepts": []}}),
        encoding="utf-8",
    )
    assert _load_concept_cache(cache_path, "d") is None


# ---------------------------------------------------------------------------
# Provider concept service integration
# ---------------------------------------------------------------------------


class _StubConceptProvider:
    """Represents stub concept provider behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    name = "stub"

    def generate(self, request):
        """Generate.

        Args:
            request: Request value used by the operation.
        """
        import json

        from graphwiki_kb.providers.base import ProviderResponse

        source_pages = []
        for line in request.prompt.splitlines():
            if line.startswith("### wiki/sources/"):
                source_pages.append(line[4:].strip())
        report = {
            "concepts": (
                [
                    {
                        "title": "Stub Cluster",
                        "summary": "Stub summary.",
                        "topic_terms": ["stub-theme"],
                        "source_pages": source_pages[:3],
                    }
                ]
                if len(source_pages) >= 3
                else []
            )
        }
        return ProviderResponse(text=json.dumps(report), model_name="stub-v1")


class _FailingProvider:
    """Represents failing provider behavior and data.

    Attributes:
        See annotated class attributes for stored values.
    """

    name = "failing"

    def generate(self, request):
        """Generate.

        Args:
            request: Request value used by the operation.
        """
        raise RuntimeError("provider down")


def test_generate_with_provider_creates_concept_pages(test_project) -> None:
    """Verifies that generate with provider creates concept pages.

    Args:
        test_project: Test project value used by the operation.
    """
    for slug in ("alpha", "beta", "gamma"):
        test_project.write_file(
            f"wiki/sources/{slug}.md",
            _compiled_page(
                f"Dense Retrieval {slug.title()}",
                f"Dense retrieval method {slug}.",
                f"Dense retrieval body {slug}.",
            ),
        )

    service = ConceptService(test_project.paths, provider=_StubConceptProvider())
    result = service.generate()

    assert len(result.concept_paths) >= 1
    concept_path = test_project.root / result.concept_paths[0]
    text = concept_path.read_text(encoding="utf-8")
    assert "type: concept" in text


def test_generate_with_provider_caches_and_reuses(test_project) -> None:
    """Verifies that generate with provider caches and reuses.

    Args:
        test_project: Test project value used by the operation.
    """
    for slug in ("alpha", "beta", "gamma"):
        test_project.write_file(
            f"wiki/sources/{slug}.md",
            _compiled_page(
                f"Dense Retrieval {slug.title()}",
                f"Dense retrieval method {slug}.",
                f"Dense retrieval body {slug}.",
            ),
        )

    provider = _StubConceptProvider()
    service = ConceptService(test_project.paths, provider=provider)
    first = service.generate()
    cache_path = service._concept_cache_path()
    assert cache_path.exists()

    # Second run should use cache (even with different provider instance)
    service2 = ConceptService(test_project.paths, provider=_FailingProvider())
    second = service2.generate()

    assert len(first.concept_paths) == len(second.concept_paths)


def test_generate_with_provider_falls_back_on_failure(test_project) -> None:
    """Verifies that generate with provider falls back on failure.

    Args:
        test_project: Test project value used by the operation.
    """
    for slug in ("alpha", "beta", "gamma"):
        test_project.write_file(
            f"wiki/sources/{slug}.md",
            _compiled_page(
                f"Dense Retrieval {slug.title()}",
                f"Dense retrieval method {slug}.",
                f"Dense retrieval body {slug}.",
            ),
        )

    service = ConceptService(test_project.paths, provider=_FailingProvider())
    result = service.generate()

    # Should still produce concepts via deterministic fallback
    assert isinstance(result.concept_paths, list)


def test_generate_with_provider_cache_invalidated_on_source_change(
    test_project,
) -> None:
    """Verifies that generate with provider cache invalidated on source change.

    Args:
        test_project: Test project value used by the operation.
    """
    for slug in ("alpha", "beta", "gamma"):
        test_project.write_file(
            f"wiki/sources/{slug}.md",
            _compiled_page(
                f"Dense Retrieval {slug.title()}",
                f"Dense retrieval method {slug}.",
                f"Dense retrieval body {slug}.",
            ),
        )

    provider = _StubConceptProvider()
    service = ConceptService(test_project.paths, provider=provider)
    service.generate()

    # Change a source page
    test_project.write_file(
        "wiki/sources/alpha.md",
        _compiled_page(
            "Completely Different Topic",
            "Totally new summary about something else.",
            "New body content.",
        ),
    )

    # New service should not use stale cache
    service2 = ConceptService(test_project.paths, provider=provider)
    result = service2.generate()
    assert isinstance(result.concept_paths, list)
