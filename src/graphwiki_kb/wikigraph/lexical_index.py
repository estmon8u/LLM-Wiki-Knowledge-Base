"""Lexical retrieval over WikiGraphRAG chunks."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from graphwiki_kb.wikigraph.deps import try_import_bm25s
from graphwiki_kb.wikigraph.markdown_parser import ParsedChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass(frozen=True)
class LexicalHit:
    """One lexical retrieval hit."""

    chunk_id: str
    score: float


class LexicalIndex:
    """BM25S-backed or simple lexical index over chunk text."""

    def __init__(
        self,
        *,
        backend: str,
        chunks: list[ParsedChunk],
        index_dir: Path,
    ) -> None:
        self.backend = backend
        self.chunks = chunks
        self.index_dir = index_dir
        self._chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self._bm25 = None
        self._corpus_tokens: list[list[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._avg_doc_len = 0.0
        self._build()

    def search(self, query: str, *, limit: int = 12) -> list[LexicalHit]:
        """Return top lexical hits for a query."""
        if not self.chunks:
            return []
        if self._bm25 is not None:
            return self._search_bm25s(query, limit=limit)
        return self._search_simple(query, limit=limit)

    def save(self) -> None:
        """Persist index metadata for inspection."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.backend,
            "chunk_count": len(self.chunks),
            "chunk_ids": [chunk.chunk_id for chunk in self.chunks],
        }
        (self.index_dir / "index_meta.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        if self._bm25 is not None:
            self._bm25.save(str(self.index_dir / "bm25s_index"), show_progress=False)

    @classmethod
    def load(
        cls,
        *,
        backend: str,
        chunks: list[ParsedChunk],
        index_dir: Path,
    ) -> LexicalIndex:
        """Load or rebuild a lexical index."""
        return cls(backend=backend, chunks=chunks, index_dir=index_dir)

    def _build(self) -> None:
        corpus = [_chunk_text(chunk) for chunk in self.chunks]
        if self.backend == "bm25s":
            bm25s = try_import_bm25s()
            if bm25s is not None:
                tokenized = bm25s.tokenize(corpus, show_progress=False)
                retriever = bm25s.BM25()
                retriever.index(tokenized, show_progress=False)
                self._bm25 = retriever
                self._bm25_corpus = tokenized
                return
        self.backend = "simple"
        self._corpus_tokens = [_tokenize(text) for text in corpus]
        lengths = [len(tokens) for tokens in self._corpus_tokens]
        self._avg_doc_len = sum(lengths) / max(len(lengths), 1)
        for tokens in self._corpus_tokens:
            self._doc_freq.update(set(tokens))

    def _search_bm25s(self, query: str, *, limit: int) -> list[LexicalHit]:
        assert self._bm25 is not None
        if not self.chunks:
            return []
        bm25s = try_import_bm25s()
        assert bm25s is not None
        query_tokens = bm25s.tokenize([query], show_progress=False)
        indices, scores = self._bm25.retrieve(
            query_tokens,
            k=min(limit, len(self.chunks)),
            show_progress=False,
        )
        hits: list[LexicalHit] = []
        for row_index, row_scores in zip(indices, scores, strict=False):
            for doc_index, score in zip(row_index, row_scores, strict=False):
                if doc_index < 0:
                    continue
                chunk = self.chunks[int(doc_index)]
                hits.append(LexicalHit(chunk_id=chunk.chunk_id, score=float(score)))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]

    def _search_simple(self, query: str, *, limit: int) -> list[LexicalHit]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        query_counts = Counter(query_tokens)
        total_docs = len(self._corpus_tokens)
        hits: list[LexicalHit] = []
        k1 = 1.5
        b = 0.75
        for chunk, doc_tokens in zip(self.chunks, self._corpus_tokens, strict=True):
            if not doc_tokens:
                continue
            doc_counts = Counter(doc_tokens)
            doc_len = len(doc_tokens)
            score = 0.0
            for term, qf in query_counts.items():
                if term not in doc_counts:
                    continue
                df = self._doc_freq.get(term, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                tf = doc_counts[term]
                denom = tf + k1 * (1 - b + b * doc_len / max(self._avg_doc_len, 1))
                score += idf * (tf * (k1 + 1)) / denom * qf
            if score > 0:
                hits.append(LexicalHit(chunk_id=chunk.chunk_id, score=score))
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]


def _chunk_text(chunk: ParsedChunk) -> str:
    return f"{chunk.title}\n{chunk.heading}\n{chunk.text}"


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
