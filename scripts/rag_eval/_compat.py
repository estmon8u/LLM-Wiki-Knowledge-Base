"""Import-time compatibility shims so `ragas` imports in this environment.

`ragas 0.2.x` imports ``langchain_community.chat_models.vertexai.ChatVertexAI``
at module load, but the langchain version resolved in this project removed that
module (it only mattered for an optional VertexAI wrapper we never use). We
register a tiny stub so `import ragas` succeeds. Importing this module before
`ragas` is required.
"""

from __future__ import annotations

import sys
import types


def install_ragas_compat_shims() -> None:
    """Install stubs required for `import ragas` to succeed (idempotent)."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return
    try:  # If the real module exists, do nothing.
        __import__(module_name)
        return
    except Exception:
        pass

    stub = types.ModuleType(module_name)

    class ChatVertexAI:  # minimal stub; the VertexAI path is never exercised.
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError(
                "ChatVertexAI is a compatibility stub and is not supported."
            )

    stub.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
    sys.modules[module_name] = stub


install_ragas_compat_shims()
