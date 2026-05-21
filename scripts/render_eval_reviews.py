"""Render per-question and per-PDF review markdown from evaluator artifacts.

Reads the JSON written by :file:`scripts/evaluate_backends.py` (the
``backend_runs_<label>.json`` artifact) plus the WikiGraphRAG index
JSON to produce:

* ``eval/results/per_question_review.md`` — side-by-side answer +
  retrieval for every benchmark question across WGR and GraphRAG.
* ``eval/results/per_pdf_review.md`` — for each of the 10 PDFs, the
  curated wiki source page, the WGR TextUnit count, and the number
  of GraphRAG entities that reference the paper title.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=(
            REPO_ROOT
            / "eval"
            / "results"
            / "artifacts"
            / "backend_runs_real_pdf_answers_v2.json"
        ),
        help="Path to the backend_runs_<label>.json artifact.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("~/wgr-eval-project").expanduser(),
        help="The KB project root used to generate the artifact.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "eval" / "results",
        help="Directory to write per_question_review.md / per_pdf_review.md.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.artifact.read_text(encoding="utf-8"))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_per_question(payload, out_dir / "per_question_review.md")
    _write_per_pdf(args.project_root, out_dir / "per_pdf_review.md")
    print(f"Wrote {out_dir / 'per_question_review.md'}")
    print(f"Wrote {out_dir / 'per_pdf_review.md'}")
    return 0


def _write_per_question(payload: dict[str, Any], dest: Path) -> None:
    answers_by_qid: dict[str, dict[str, dict[str, Any]]] = {}
    for row in payload["results"]["answers"]:
        qid = row["run"]["question_id"]
        backend = row["run"]["backend"]
        answers_by_qid.setdefault(qid, {})[backend] = row

    retrieval_by_qid: dict[str, dict[str, dict[str, Any]]] = {}
    for row in payload["results"]["retrieval"]:
        qid = row["run"]["question_id"]
        backend = row["run"]["backend"]
        retrieval_by_qid.setdefault(qid, {})[backend] = row

    lines: list[str] = [
        "# Per-question side-by-side review (WGR vs GraphRAG)",
        "",
        "Run label: `real_pdf_answers_v2` — see "
        "`eval/results/backend_summary_real_pdf_answers_v2.md` for the headline numbers.",
        "Each section shows the question text, both backends' answers, citation counts, "
        "and what the retrieval surfaced.",
        "",
    ]
    for qid in sorted(answers_by_qid):
        wgr = answers_by_qid[qid].get("wikigraph", {})
        gr = answers_by_qid[qid].get("graphrag", {})
        wgr_run = wgr.get("run", {})
        gr_run = gr.get("run", {})
        wgr_metrics = wgr.get("metrics", {})
        gr_metrics = gr.get("metrics", {})
        question_text = wgr_run.get("question") or gr_run.get("question") or qid
        lines.append(f"## {qid}")
        lines.append("")
        lines.append(f"**Question:** {question_text}")
        lines.append("")
        lines.append(
            f"| Metric | WGR | GraphRAG |\n|---|---|---|\n"
            f"| answer_quality_score | "
            f"{wgr_metrics.get('answer_quality_score', '-')} | "
            f"{gr_metrics.get('answer_quality_score', '-')} |\n"
            f"| grounded_entity_rate | "
            f"{wgr_metrics.get('grounded_entity_rate', '-')} | "
            f"{gr_metrics.get('grounded_entity_rate', '-')} |\n"
            f"| insufficient_evidence | "
            f"{wgr_run.get('insufficient_evidence', '-')} | "
            f"{gr_run.get('insufficient_evidence', '-')} |\n"
            f"| citation_count | {wgr_run.get('citation_count', '-')} | "
            f"{gr_run.get('citation_count', '-')} |\n"
            f"| citation_ref_valid_strict | "
            f"{wgr_run.get('citation_ref_strict_rate', '-')} | "
            f"{gr_run.get('citation_ref_strict_rate', '-')} |\n"
            f"| latency_seconds | {wgr_run.get('latency_seconds', '-')} | "
            f"{gr_run.get('latency_seconds', '-')} |"
        )
        lines.append("")
        lines.append("### WGR answer")
        lines.append("")
        lines.append("```text")
        lines.append((wgr_run.get("answer") or "(no answer)")[:1500])
        lines.append("```")
        lines.append("")
        lines.append("### GraphRAG answer")
        lines.append("")
        lines.append("```text")
        lines.append((gr_run.get("answer") or "(no answer)")[:1500])
        lines.append("```")
        lines.append("")
        wgr_r = retrieval_by_qid.get(qid, {}).get("wikigraph", {}).get("run", {})
        gr_r = retrieval_by_qid.get(qid, {}).get("graphrag", {}).get("run", {})
        lines.append("### Retrieval — top 8 (title | path)")
        lines.append("")
        lines.append("**WGR:**")
        for title, path in zip(
            wgr_r.get("retrieved_titles", []),
            wgr_r.get("retrieved_paths", []),
            strict=False,
        ):
            lines.append(f"- `{title[:80]}` ← `{path[:80]}`")
        lines.append("")
        lines.append("**GraphRAG:**")
        for title, path in zip(
            gr_r.get("retrieved_titles", []),
            gr_r.get("retrieved_paths", []),
            strict=False,
        ):
            lines.append(f"- `{title[:80]}` ← `{path[:80]}`")
        lines.append("")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_per_pdf(project_root: Path, dest: Path) -> None:
    project_root = project_root.expanduser().resolve()
    normalized = project_root / "raw" / "normalized"
    wikigraph_dir = project_root / "graph" / "wikigraph"

    # Load wikigraph index to count text units per source document.
    text_unit_counts: dict[str, int] = {}
    entity_paths_to_titles: dict[str, list[str]] = {}
    try:
        nodes = json.loads((wikigraph_dir / "nodes.json").read_text())
        for node in nodes:
            if node["kind"] == "text_unit":
                path = node.get("path", "")
                text_unit_counts[path] = text_unit_counts.get(path, 0) + 1
            elif node["kind"] == "entity":
                path = node.get("path", "")
                entity_paths_to_titles.setdefault(path, []).append(node["title"])
    except FileNotFoundError:
        pass

    # Load GraphRAG entities for paper-title mention counts.
    graphrag_entities: list[str] = []
    try:
        import pyarrow.parquet as pq

        entity_path = (
            project_root / "graph" / "graphrag" / "output" / "entities.parquet"
        )
        if entity_path.exists():
            table = pq.read_table(entity_path, columns=["title"])
            graphrag_entities = [str(t) for t in table.column("title").to_pylist() if t]
    except ImportError:
        pass

    pdfs = sorted(normalized.glob("*.md"))
    lines = [
        "# Per-PDF inspection (10-paper arXiv corpus)",
        "",
        "For each of the 10 PDFs we record:",
        "- Normalized markdown path + length",
        "- Curated wiki source page path",
        "- WGR TextUnit count derived from the normalized body",
        "- WGR curated entities whose `path` points to the wiki source page",
        "- Number of GraphRAG entity-title mentions of the paper short name",
        "",
    ]
    short_name_map = {
        "atlas-few-shot-learning-with-retrieval-augmented-language-models": "Atlas",
        "dense-passage-retrieval-for-open-domain-question-answering": "DPR",
        "from-local-to-global-a-graphrag-approach-to-query-focused-summarization": (
            "GraphRAG"
        ),
        "in-context-retrieval-augmented-language-models": "In-Context RALM",
        "latent-retrieval-for-weakly-supervised-open-domain-question-answering": (
            "ORQA"
        ),
        "leveraging-passage-retrieval-with-generative-models-for-open-domain-question-answering": (
            "FiD"
        ),
        "realm-retrieval-augmented-language-model-pre-training": "REALM",
        "replug-retrieval-augmented-black-box-language-models": "REPLUG",
        "retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks": "RAG",
        "self-rag-learning-to-retrieve-generate-and-critique-through-self-reflection": (
            "Self-RAG"
        ),
    }
    for path in pdfs:
        slug = path.stem
        normalized_chars = path.stat().st_size
        rel_normalized = f"raw/normalized/{path.name}"
        text_units = text_unit_counts.get(rel_normalized, 0)
        wgr_entities = entity_paths_to_titles.get(f"wiki/sources/{slug}.md", [])
        short_name = short_name_map.get(slug, slug.split("-")[0])
        # Case-sensitive word-boundary match against entity titles to
        # avoid noisy matches like "IN" being inside many titles.
        import re as _re

        pattern = _re.compile(
            r"(?<![A-Za-z0-9_])" + _re.escape(short_name) + r"(?![A-Za-z0-9_])",
            _re.IGNORECASE,
        )
        graphrag_mentions = sum(1 for t in graphrag_entities if pattern.search(str(t)))
        lines.append(f"## {slug}")
        lines.append("")
        lines.append(
            f"- Normalized markdown: `{rel_normalized}` ({normalized_chars} bytes)"
        )
        lines.append(f"- Wiki source page: `wiki/sources/{slug}.md`")
        lines.append(f"- WGR TextUnit count for this paper: **{text_units}**")
        if wgr_entities:
            preview = ", ".join(f"`{e}`" for e in wgr_entities[:6])
            lines.append(
                f"- WGR entities anchored to wiki page ({len(wgr_entities)}): {preview}"
            )
        lines.append(
            f"- GraphRAG entity titles word-matching `{short_name}`: "
            f"**{graphrag_mentions}**"
        )
        lines.append("")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
