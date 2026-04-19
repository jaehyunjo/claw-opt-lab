"""Backend factory — picks between Azure OpenAI and the Gemma (HF) adapter.

Both backends expose the same duck-typed surface the agent loop uses::

    client.chat.completions.create(model=..., messages=..., tools=...)
        → resp.choices[0].message.{content, tool_calls}

so ``run_conversation`` doesn't need to know which one it's driving.
"""

from __future__ import annotations

from typing import Tuple

BACKEND_AZURE = "azure"
BACKEND_GEMMA = "gemma"
ALL_BACKENDS = (BACKEND_AZURE, BACKEND_GEMMA)


class UnknownBackendError(ValueError):
    pass


def get_backend(name: str) -> Tuple[object, str]:
    """Return ``(client, deployment)`` for the requested backend."""
    if name == BACKEND_AZURE:
        from core.azure import get_client, get_deployment
        return get_client(), get_deployment()
    if name == BACKEND_GEMMA:
        from core.gemma import get_client, get_deployment
        return get_client(), get_deployment()
    raise UnknownBackendError(
        f"unknown backend: {name!r} (choose from: {', '.join(ALL_BACKENDS)})"
    )
