from __future__ import annotations

from pathlib import Path

import yaml

from src.services.concept_service import (
    ConceptService,
    _derive_topic_terms,
    _extract_terms,
    _format_concept_title,
    _replace_backlinks_block,
    _SourcePage,
    _split_frontmatter,
    _stem_token,
)


def _compiled_page(title: str, summary: str, body: str) -> str:
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
    assert _stem_token("retrieval") == "retriev"
    assert _stem_token("models") == "model"
    assert _stem_token("training") == "train"
    assert _stem_token("used") == "used"  # too short to strip
    assert _stem_token("few") == "few"


def test_stem_token_minimum_length_guard() -> None:
    """Stemmer must not produce stems shorter than 4 characters."""
    assert _stem_token("canonical") == "canonical"
    assert _stem_token("canonic") == "canonic"
    assert _stem_token("real") == "real"
    assert _stem_token("based") == "based"


def test_extract_terms_skips_stopwords_and_short_tokens() -> None:
    terms = _extract_terms("The models are used for retrieval tasks.")
    assert "the" not in terms
    assert "are" not in terms
    assert "for" not in terms
    assert "retriev" in terms


def test_extract_terms_filters_stemmed_generic_tokens() -> None:
    terms = _extract_terms(
        "Question answering with language models and learning systems."
    )

    assert "answer" not in terms
    assert "languag" not in terms
    assert "learn" not in terms


def test_split_frontmatter_parses_yaml() -> None:
    text = "---\ntitle: Hello\n---\n\nBody text.\n"
    fm, body = _split_frontmatter(text)
    assert fm["title"] == "Hello"
    assert body.strip() == "Body text."


def test_split_frontmatter_handles_missing_frontmatter() -> None:
    text = "# No frontmatter\n\nBody.\n"
    fm, body = _split_frontmatter(text)
    assert fm == {}
    assert body == text


def test_format_concept_title_one_term() -> None:
    assert _format_concept_title(["dense-retrieval"]) == "Dense Retrieval"


def test_format_concept_title_two_terms() -> None:
    assert (
        _format_concept_title(["dense-retrieval", "question-answering"])
        == "Dense Retrieval and Question Answering"
    )


def test_format_concept_title_three_terms() -> None:
    result = _format_concept_title(["alpha", "beta", "gamma"])
    assert result == "Alpha, Beta, and Gamma"


def test_replace_backlinks_block_adds_section() -> None:
    original = "---\ntitle: Page\n---\n\n# Page\n\nBody.\n"
    result = _replace_backlinks_block(original, [("alpha", "Alpha")])
    assert "## Related Concept Pages" in result
    assert "[[alpha|Alpha]]" in result
    assert result.endswith("\n")


def test_replace_backlinks_block_replaces_existing_block() -> None:
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
    original = "# Page\n\nBody.\n"
    assert _replace_backlinks_block(original, []) == original


def test_derive_topic_terms_uses_frequency() -> None:
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


def test_derive_topic_terms_rejects_broad_generic_themes() -> None:
    pages = [
        _SourcePage(
            file_path=None,
            relative_path="wiki/sources/a.md",
            slug="a",
            title="Question Answering with Language Models",
            summary="Knowledge intensive language model workflows.",
            terms=_extract_terms(
                "Question Answering with Language Models "
                "Knowledge intensive language model workflows."
            ),
        ),
        _SourcePage(
            file_path=None,
            relative_path="wiki/sources/b.md",
            slug="b",
            title="Language Models for Question Answering",
            summary="Knowledge intensive tasks for question answering.",
            terms=_extract_terms(
                "Language Models for Question Answering "
                "Knowledge intensive tasks for question answering."
            ),
        ),
        _SourcePage(
            file_path=None,
            relative_path="wiki/sources/c.md",
            slug="c",
            title="Knowledge in Language Models for Question Answering",
            summary="Question answering tasks depend on language model knowledge.",
            terms=_extract_terms(
                "Knowledge in Language Models for Question Answering "
                "Question answering tasks depend on language model knowledge."
            ),
        ),
    ]

    assert _derive_topic_terms(pages) == []


# --- Service integration tests ---


def test_generate_no_source_pages_returns_empty(test_project) -> None:
    service = ConceptService(test_project.paths)
    result = service.generate()

    assert result.concept_paths == []
    assert result.updated_source_paths == []
    assert result.removed_paths == []


def test_generate_single_source_page_returns_empty(test_project) -> None:
    test_project.write_file(
        "wiki/sources/only-page.md",
        _compiled_page("Only Page", "Standalone page.", "Body content."),
    )

    service = ConceptService(test_project.paths)
    result = service.generate()

    assert result.concept_paths == []


def test_generate_creates_concept_page_for_related_sources(test_project) -> None:
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
    import shutil

    if test_project.paths.wiki_sources_dir.exists():
        shutil.rmtree(test_project.paths.wiki_sources_dir)

    service = ConceptService(test_project.paths)
    result = service.generate()

    assert result.concept_paths == []


def test_resolve_destination_skips_manual_page(test_project) -> None:
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
        assert "dense-retrieval.md" != cpath.split("/")[-1]
    manual = (test_project.root / "wiki/concepts/dense-retrieval.md").read_text(
        encoding="utf-8"
    )
    assert "Hand-written" in manual


def test_compile_with_concepts_flag_cli() -> None:
    from click.testing import CliRunner
    from unittest.mock import patch
    from src.cli import main
    from src.providers.base import ProviderRequest, ProviderResponse, TextProvider

    class _FakeProvider(TextProvider):
        name = "fake"

        def generate(self, request: ProviderRequest) -> ProviderResponse:
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

        with patch("src.services.build_provider", return_value=_FakeProvider()):
            result = runner.invoke(main, ["update"])

        assert result.exit_code == 0
        assert "Concept Summary" in result.output


def test_split_frontmatter_handles_invalid_yaml() -> None:
    text = "---\n: :\n  bad: [unclosed\n---\n\nBody.\n"
    fm, body = _split_frontmatter(text)
    assert fm == {}


def test_list_managed_pages_skips_non_concept_pages(test_project) -> None:
    test_project.paths.wiki_concepts_dir.mkdir(parents=True, exist_ok=True)
    test_project.write_file(
        "wiki/concepts/manual.md",
        "---\ntitle: Manual Topic\ntype: analysis\n---\n\n# Manual\n\nHand-written.\n",
    )

    service = ConceptService(test_project.paths)
    managed = service._list_managed_pages()

    assert len(managed) == 0


def test_generate_with_unrelated_pages_produces_no_concepts(test_project) -> None:
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
    text = "---\ntitle: Hello\nNo closing marker here\n"
    fm, body = _split_frontmatter(text)
    assert fm == {}
    assert body == text


def test_resolve_destination_suffix_collision(test_project) -> None:
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


def test_list_managed_pages_no_concepts_dir(test_project) -> None:
    import shutil

    if test_project.paths.wiki_concepts_dir.exists():
        shutil.rmtree(test_project.paths.wiki_concepts_dir)

    service = ConceptService(test_project.paths)
    managed = service._list_managed_pages()

    assert managed == set()


def test_list_managed_pages_handles_unreadable_file(test_project) -> None:
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
        import yaml

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
    import yaml

    page = f"---\n{yaml.safe_dump(fm, sort_keys=False)}---\n\n# Dense Passage Retrieval\n\n## Summary\n\nNo summary available yet.\n"
    test_project.paths.wiki_sources_dir.mkdir(parents=True, exist_ok=True)
    (test_project.paths.wiki_sources_dir / "dpr.md").write_text(page, encoding="utf-8")

    service = ConceptService(test_project.paths)
    pages = service._load_source_pages()

    assert len(pages) == 1
    assert pages[0].summary == ""
    assert "summar" not in pages[0].terms
    assert "avail" not in pages[0].terms
