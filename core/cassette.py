"""Record and replay Azure OpenAI responses.

Both ``RecordingClient`` and ``ReplayClient`` duck-type the small subset of
the ``AzureOpenAI`` interface the agent loop uses::

    client.chat.completions.create(model=..., messages=..., tools=...)

and return an object with::

    resp.choices[0].message.{content, tool_calls}
    tc.id, tc.function.name, tc.function.arguments

Cassettes are JSON files named ``cassettes/<scenario>.json``::

    {
      "recorded_at": "...",
      "model": "gpt-5-mini",
      "calls": [
        {
          "request": {"n_messages": 2, "last_user": "…", "tool_count": 47},
          "response": {
            "content": null,
            "tool_calls": [
              {"id": "call_1", "type": "function",
               "function": {"name": "clarify", "arguments": "{...}"}}
            ]
          }
        },
        ...
      ]
    }
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


# ── SDK-shape adapters ─────────────────────────────────────────────────────

class _Endpoint:
    """Minimal ``.completions`` object with a ``.create(**kwargs)`` method."""
    def __init__(self, create_fn):
        self.create = create_fn


class _Chat:
    def __init__(self, create_fn):
        self.completions = _Endpoint(create_fn)


def _serialize_response(resp) -> Dict[str, Any]:
    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None) or []
    return {
        "content": getattr(msg, "content", None),
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in tool_calls
        ] or None,
    }


def _deserialize_response(data: Dict[str, Any]):
    tool_calls = []
    for tc in data.get("tool_calls") or []:
        tool_calls.append(SimpleNamespace(
            id=tc["id"],
            type=tc.get("type", "function"),
            function=SimpleNamespace(
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"],
            ),
        ))
    msg = SimpleNamespace(
        content=data.get("content"),
        tool_calls=tool_calls or None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _request_signature(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    messages = kwargs.get("messages") or []
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content") or ""
            last_user = content if isinstance(content, str) else str(content)
            last_user = last_user[:160]
            break
    return {
        "n_messages": len(messages),
        "last_user": last_user,
        "tool_count": len(kwargs.get("tools") or []),
        "model": kwargs.get("model"),
    }


# ── Clients ────────────────────────────────────────────────────────────────

class RecordingClient:
    """Wraps a real client; also appends each response to an in-memory list.

    Call :meth:`save` after your run to persist the cassette.
    """

    def __init__(self, real_client, cassette_path: Path, model_hint: str = ""):
        self._real = real_client
        self._path = Path(cassette_path)
        self._calls: List[Dict[str, Any]] = []
        self._model_hint = model_hint
        self.chat = _Chat(self._create)

    def _create(self, **kwargs):
        resp = self._real.chat.completions.create(**kwargs)
        self._calls.append({
            "request": _request_signature(kwargs),
            "response": _serialize_response(resp),
        })
        return resp

    def save(self) -> Path:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "recorded_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": self._model_hint,
            "calls": self._calls,
        }
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return self._path


class ReplayClient:
    """Serves responses from a cassette file. No network."""

    def __init__(self, cassette_path: Path):
        self._path = Path(cassette_path)
        if not self._path.exists():
            raise CassetteMissingError(self._path)
        data = json.loads(self._path.read_text(encoding="utf-8"))
        self._calls: List[Dict[str, Any]] = list(data.get("calls") or [])
        self._idx = 0
        self.chat = _Chat(self._create)

    def _create(self, **_kwargs):
        if self._idx >= len(self._calls):
            raise CassetteExhaustedError(self._path, len(self._calls))
        call = self._calls[self._idx]
        self._idx += 1
        return _deserialize_response(call["response"])


class CassetteMissingError(FileNotFoundError):
    def __init__(self, path: Path):
        super().__init__(f"Cassette not found: {path}. "
                         f"Run with --mode record to create it.")
        self.path = path


class CassetteExhaustedError(RuntimeError):
    def __init__(self, path: Path, length: int):
        super().__init__(
            f"Cassette exhausted: {path} has {length} recorded call(s) but "
            f"the loop requested more. The system prompt, tool schemas, or "
            f"the scenario prompt may have changed since recording — re-record "
            f"with --mode record."
        )
        self.path = path
