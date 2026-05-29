"""Entry point for the research-grounded RAG evaluation harness.

See :mod:`scripts.rag_eval.cli` for usage. Compares direct / legacy / graphrag /
wikigraph with rank-aware retrieval metrics, RAGAS, a bias-mitigated judge, and
bootstrap confidence intervals.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.rag_eval.cli import main  # noqa: E402

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
