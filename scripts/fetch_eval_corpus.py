"""Download a fixed corpus of 10 retrieval-augmented-generation papers from arXiv.

Used by the Phase 5/6 end-to-end evaluation of WikiGraphRAG vs
Microsoft GraphRAG. The corpus was chosen to mirror the benchmark
questions in :file:`eval/benchmark.yaml` and to cover the spectrum of
methods the benchmark mentions (REALM, RAG, DPR, FiD, REPLUG, Self-RAG,
Atlas, In-Context RALM, ORQA, Microsoft GraphRAG).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CorpusPaper:
    """Single arXiv paper specification."""

    slug: str
    arxiv_id: str
    title: str


CORPUS: tuple[CorpusPaper, ...] = (
    CorpusPaper(
        slug="realm",
        arxiv_id="2002.08909",
        title="REALM: Retrieval-Augmented Language Model Pre-Training",
    ),
    CorpusPaper(
        slug="rag",
        arxiv_id="2005.11401",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    ),
    CorpusPaper(
        slug="dpr",
        arxiv_id="2004.04906",
        title="Dense Passage Retrieval for Open-Domain Question Answering",
    ),
    CorpusPaper(
        slug="fid",
        arxiv_id="2007.01282",
        title="Leveraging Passage Retrieval with Generative Models for Open Domain QA (FiD)",
    ),
    CorpusPaper(
        slug="replug",
        arxiv_id="2301.12652",
        title="REPLUG: Retrieval-Augmented Black-Box Language Models",
    ),
    CorpusPaper(
        slug="self-rag",
        arxiv_id="2310.11511",
        title="Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
    ),
    CorpusPaper(
        slug="atlas",
        arxiv_id="2208.03299",
        title="Atlas: Few-shot Learning with Retrieval Augmented Language Models",
    ),
    CorpusPaper(
        slug="in-context-ralm",
        arxiv_id="2302.00083",
        title="In-Context Retrieval-Augmented Language Models",
    ),
    CorpusPaper(
        slug="orqa",
        arxiv_id="1906.00300",
        title="Latent Retrieval for Weakly Supervised Open Domain Question Answering (ORQA)",
    ),
    CorpusPaper(
        slug="graphrag",
        arxiv_id="2404.16130",
        title="From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "eval-pdfs",
        help="Directory to write the PDFs into (default: ~/eval-pdfs).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Number of download retries per PDF (default: 4).",
    )
    return parser


def fetch_pdf(arxiv_id: str, dest: Path, *, retries: int) -> bool:
    """Download an arXiv PDF, retrying with exponential backoff."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    request = urllib.request.Request(
        url,
        headers={
            # arXiv blocks the default Python UA; identify ourselves so
            # the polite-bot path is taken instead of a 403.
            "User-Agent": (
                "graphwiki-kb-eval/0.1 (+https://github.com/estmon8u/"
                "LLM-Wiki-Knowledge-Base) python-urllib"
            ),
            "Accept": "application/pdf,*/*",
        },
    )
    backoff = 2.0
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
            print(
                f"  attempt {attempt}/{retries} for {arxiv_id} failed: {exc}",
                file=sys.stderr,
            )
            if attempt == retries:
                break
            time.sleep(backoff)
            backoff *= 2
            continue
        if not payload.startswith(b"%PDF"):
            last_err = RuntimeError(f"Did not receive PDF magic for {arxiv_id}")
            time.sleep(backoff)
            backoff *= 2
            continue
        if len(payload) < 10 * 1024:
            last_err = RuntimeError(
                f"PDF for {arxiv_id} is suspiciously small ({len(payload)} bytes)."
            )
            time.sleep(backoff)
            backoff *= 2
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()[:12]
        print(
            f"  ok: wrote {dest.name} ({len(payload) // 1024} KiB, sha256[:12]={digest})"
        )
        return True
    print(f"  FAILED to download {arxiv_id}: {last_err}", file=sys.stderr)
    return False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir: Path = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(CORPUS)} papers into {out_dir}")
    failures: list[str] = []
    for paper in CORPUS:
        dest = out_dir / f"{paper.slug}.pdf"
        if dest.exists() and dest.stat().st_size > 10 * 1024:
            print(f"- {paper.slug} ({paper.arxiv_id}): already present")
            continue
        print(f"- {paper.slug} ({paper.arxiv_id}): {paper.title}")
        ok = fetch_pdf(paper.arxiv_id, dest, retries=args.retries)
        if not ok:
            failures.append(paper.slug)
        # Polite pacing between requests to avoid hammering arXiv.
        time.sleep(1.0)
    if failures:
        print(f"Failed: {failures}", file=sys.stderr)
        return 1
    print(f"Done. All {len(CORPUS)} PDFs in {out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
