"""Import shim + tool-handler mock patch.

The project has been stripped down to a minimal skeleton. Many top-level
packages (``run_agent``, ``model_tools``, ``hermes_cli``, ``gateway``, ``cron``,
``agent``, ``toolsets``) and heavy external deps (``exa_py``, ``firecrawl``,
``fal_client``, etc.) have been deleted from disk, but tool files under
``tools/`` still import them at module scope.

``install_shim()`` installs a ``sys.meta_path`` finder that returns a
``MagicMock`` module for those names, so tool files can import and register
themselves without their real dependencies.

It also monkey-patches ``tools.registry.registry.register`` so every tool's
handler is replaced with a mock that validates args against the tool's JSON
schema and returns a success envelope.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import sys
from typing import Any, Dict, Iterable
from unittest.mock import MagicMock


# ── What to mock ────────────────────────────────────────────────────────────

_EXTERNAL_MOCKS = [
    # Heavy or optional external deps imported by tool files
    "exa_py", "firecrawl", "firecrawl_py", "parallel", "parallel_web",
    "fal_client", "edge_tts", "elevenlabs", "mistralai", "sounddevice",
    "ptyprocess", "winpty", "modal", "daytona", "atroposlib", "wandb",
    "telegram", "slack_bolt", "slack_sdk", "discord",
    "browserbase", "browser_use", "honcho", "honcho_ai",
    "whisper", "faster_whisper", "PIL", "cv2", "qrcode",
    "mcp", "requests",
]

_INTERNAL_MOCKS = [
    # Deleted top-level internal packages
    "run_agent", "model_tools",
    "hermes_cli", "hermes_logging", "hermes_state", "hermes_time",
    "gateway", "cron",
    "agent",
    "toolsets",
    # Deleted tool helper modules (tools/ itself stays real)
    "tools.ansi_strip", "tools.approval", "tools.binary_extensions",
    "tools.browser_providers", "tools.browser_camofox",
    "tools.browser_camofox_state", "tools.checkpoint_manager",
    "tools.credential_files", "tools.debug_helpers",
    "tools.environments", "tools.env_passthrough",
    "tools.file_operations", "tools.fuzzy_match", "tools.interrupt",
    "tools.managed_tool_gateway", "tools.mcp_oauth",
    "tools.mcp_oauth_manager", "tools.mcp_tool",
    "tools.neutts_synth", "tools.openrouter_client",
    "tools.osv_check", "tools.patch_parser", "tools.path_security",
    "tools.skills_guard", "tools.skills_hub", "tools.skills_sync",
    "tools.tirith_security", "tools.tool_backend_helpers",
    "tools.tool_result_storage", "tools.transcription_tools",
    "tools.url_safety", "tools.voice_mode", "tools.website_policy",
    "tools.xai_http",
]


# ── meta_path finder that returns a MagicMock module ───────────────────────

class _MockLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = MagicMock(name=spec.name)
        # Mark as a package so Python allows sub-module imports (``from a.b import c``).
        mod.__path__ = []
        mod.__spec__ = spec
        mod.__name__ = spec.name
        return mod

    def exec_module(self, module):  # nothing to execute for a mock
        return None


class _MockFinder(importlib.abc.MetaPathFinder):
    def __init__(self, names: Iterable[str]):
        self._names = frozenset(names)

    def find_spec(self, fullname, path, target=None):
        if fullname in self._names:
            return importlib.machinery.ModuleSpec(fullname, _MockLoader())
        for name in self._names:
            if fullname.startswith(name + "."):
                return importlib.machinery.ModuleSpec(fullname, _MockLoader())
        return None


# ── Mock handler factory ───────────────────────────────────────────────────

_TYPE_MAP = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def _validate(args: Dict[str, Any], schema: Dict[str, Any]) -> str | None:
    """Return an error message if args don't match schema; else None.

    Only checks required fields and JSON-type of provided properties — enough to
    simulate real tool-call validation without pulling in ``jsonschema``.
    """
    params = schema.get("parameters") or {}
    properties = params.get("properties") or {}
    required = params.get("required") or []

    if not isinstance(args, dict):
        return f"Arguments must be an object, got {type(args).__name__}"

    for name in required:
        if name not in args:
            return f"Missing required parameter: {name}"

    for name, value in args.items():
        prop = properties.get(name)
        if not prop:
            continue
        expected = prop.get("type")
        if expected is None:
            continue
        types = []
        if isinstance(expected, list):
            for t in expected:
                types.extend(_TYPE_MAP.get(t, ()))
        else:
            types.extend(_TYPE_MAP.get(expected, ()))
        if not types:
            continue
        # bool is a subclass of int — special-case so bools don't sneak into integer-only slots
        if bool in types and not isinstance(value, bool) and isinstance(value, int):
            pass
        if isinstance(value, bool) and bool not in types and int in types:
            return f"Parameter '{name}' has wrong type: expected {expected}, got bool"
        if not isinstance(value, tuple(types)):
            return (
                f"Parameter '{name}' has wrong type: expected {expected}, "
                f"got {type(value).__name__}"
            )
    return None


# ── Per-tool mock-response overrides ───────────────────────────────────────
#
# Each callable receives the validated args dict and returns the payload (a
# dict — the handler wraps it in JSON). The goal is to make mock responses
# *believable enough* that the model accepts the tool as having worked, so
# it doesn't loop / retry / ask for clarification.

def _mock_clarify(args):
    return {
        "question": args.get("question", ""),
        "choices_offered": args.get("choices"),
        "user_response": "yes",
    }


def _mock_cronjob(args):
    action = (args.get("action") or "create").lower()
    if action == "list":
        return {
                "jobs": [
                {"id": "job_001", "name": "example-daily",
                 "schedule": "0 9 * * *", "status": "active"},
            ],
        }
    if action == "delete":
        return {
                "action": "delete",
            "job_id": args.get("job_id", "job_001"),
            "status": "deleted",
        }
    return {
        "action": action,
        "job_id": f"job_mock_{abs(hash(str(args))) % 10000:04d}",
        "name": args.get("name") or "unnamed-job",
        "schedule": args.get("schedule") or "",
        "status": "scheduled",
        "message": f"Cron job {action}d (mock — no real scheduler).",
    }


def _mock_vision(args):
    return {
        "description": (
            "The image shows a desktop screenshot: a code editor on the left with "
            "Python source visible, and a terminal on the right showing command "
            "output. Menu bar and a few panels are visible. (Mock description.)"
        ),
        "detected_objects": ["window", "code editor", "terminal", "text"],
        "confidence": 0.86,
    }


def _mock_send_message(args):
    return {
        "delivery_id": f"msg_mock_{abs(hash(str(args))) % 100000:05d}",
        "target": args.get("target", ""),
        "action": args.get("action", "send"),
        "status": "sent",
        "sent_at": "2026-04-19T10:30:00Z",
    }


def _mock_web_search(args):
    q = args.get("query") or args.get("q") or ""
    return {
        "query": q,
        "results": [
            {"title": f"Mock result 1: {q[:40]}",
             "url": "https://example.com/r1",
             "snippet": "First mock search result snippet related to the query."},
            {"title": f"Mock result 2: {q[:40]}",
             "url": "https://example.com/r2",
             "snippet": "Second mock snippet with a bit more context."},
            {"title": f"Mock result 3: {q[:40]}",
             "url": "https://example.com/r3",
             "snippet": "Third mock snippet."},
        ],
    }


def _mock_web_extract(args):
    urls = args.get("urls") or ([args.get("url")] if args.get("url") else [])
    return {
        "pages": [
            {"url": u, "title": "Mock Page",
             "markdown": "# Mock page\n\nThis is mock-extracted markdown content "
                         "from the requested URL. The page discusses the topic of "
                         "interest in a few paragraphs."}
            for u in (urls or ["https://example.com/"])[:3]
        ],
    }


def _mock_read_file(args):
    path = args.get("path") or ""
    return {
        "path": path,
        "content": (f"# Mock content for {path}\n"
                    f"# (file was not actually read)\n\n"
                    f"Line 1 of mock content\nLine 2 of mock content\n"),
        "lines": 5,
    }


def _mock_write_file(args):
    content = args.get("content", "")
    nbytes = len(content.encode("utf-8")) if isinstance(content, str) else 0
    return {
        "path": args.get("path", ""),
        "bytes_written": nbytes,
    }


def _mock_terminal(args):
    return {
        "command": args.get("command", ""),
        "exit_code": 0,
        "stdout": "(mock stdout — command not actually executed)\n",
        "stderr": "",
    }


def _mock_execute_code(args):
    code = args.get("code") or ""
    first = (code.splitlines() or [""])[0]
    return {
        "stdout": f"(mock execution of: {first[:80]})\n",
        "stderr": "",
        "exit_code": 0,
    }


def _mock_search_files(args):
    pattern = args.get("pattern") or args.get("query") or ""
    return {
        "pattern": pattern,
        "matches": [
            {"path": "tools/registry.py", "line": 100, "preview": f"... {pattern} ..."},
            {"path": "core/shim.py", "line": 50, "preview": f"... {pattern} ..."},
        ],
    }


def _mock_image_generate(args):
    return {
        "image_url": "https://mock.example.com/generated.png",
        "prompt": args.get("prompt", ""),
    }


def _mock_delegate(args):
    tasks = args.get("tasks") or args.get("subtasks") or []
    out = [
        {"task": str(t)[:80], "summary": f"Mock subagent result {i + 1}.",
         "status": "completed"}
        for i, t in enumerate(tasks[:5])
    ] or [{"task": "(none)", "summary": "No subtasks provided.", "status": "skipped"}]
    return {"results": out}


def _mock_session_search(args):
    q = args.get("query") or ""
    return {
        "query": q,
        "matches": [
            {"session_id": "sess_abc",
             "excerpt": f"Earlier session mentioned: '{q[:60]}'...",
             "timestamp": "2026-03-15T10:00:00Z"},
        ],
    }


def _mock_skills_list(args):
    try:
        from core.skill_loader import load_skill_index  # local to avoid cycles
        skills = load_skill_index()
        return {
                "total": len(skills),
            "skills": [
                {"name": s["name"], "description": (s["description"] or "")[:120]}
                for s in skills[:20]
            ],
        }
    except Exception:
        return {"skills": []}


def _mock_memory(args):
    action = (args.get("action") or "store").lower()
    return {
        "action": action,
        "message": f"Mock memory '{action}' completed.",
    }


def _mock_todo(args):
    todos = args.get("todos") or []
    return {
        "added": len(todos),
        "todos": [
            {"id": t.get("id", f"todo-{i}"),
             "content": t.get("content", ""),
             "status": t.get("status", "pending")}
            for i, t in enumerate(todos)
        ],
    }


_TOOL_OVERRIDES: Dict[str, Any] = {
    "clarify": _mock_clarify,
    "cronjob": _mock_cronjob,
    "vision_analyze": _mock_vision,
    "browser_vision": _mock_vision,
    "send_message": _mock_send_message,
    "web_search": _mock_web_search,
    "web_extract": _mock_web_extract,
    "read_file": _mock_read_file,
    "write_file": _mock_write_file,
    "terminal": _mock_terminal,
    "execute_code": _mock_execute_code,
    "search_files": _mock_search_files,
    "image_generate": _mock_image_generate,
    "delegate_task": _mock_delegate,
    "session_search": _mock_session_search,
    "skills_list": _mock_skills_list,
    "memory": _mock_memory,
    "todo": _mock_todo,
}


def _make_mock_handler(name: str, schema: Dict[str, Any]):
    override = _TOOL_OVERRIDES.get(name)

    def handler(args, **_kwargs):
        args = args or {}
        err = _validate(args, schema)
        if err:
            return json.dumps(
                {"error": err, "tool": name, "mock": True},
                ensure_ascii=False,
            )
        if override is not None:
            payload = override(args)
            # The override may omit these; the handler fills them so the
            # caller (and tests) always sees a consistent envelope.
            payload.setdefault("ok", True)
            payload.setdefault("tool", name)
            payload.setdefault("mock", True)
        else:
            payload = {"ok": True, "tool": name, "args": args, "mock": True}
        return json.dumps(payload, ensure_ascii=False)

    handler.__name__ = f"mock_{name}"
    return handler


# ── Public entry point ─────────────────────────────────────────────────────

_installed = False


def install_shim() -> None:
    """Install the module shim and monkey-patch the tool registry. Idempotent."""
    global _installed
    if _installed:
        return

    finder = _MockFinder(_EXTERNAL_MOCKS + _INTERNAL_MOCKS)
    sys.meta_path.insert(0, finder)

    # Import the real tool registry (not mocked — ``tools.registry`` is not in the list).
    from tools.registry import registry  # noqa: E402

    original_register = registry.register

    def mock_register(name, toolset, schema, handler, **kwargs):
        # Our mock handler is always sync, so force is_async=False regardless of
        # what the original tool declared. Otherwise registry.dispatch() would
        # route through the (mocked) ``model_tools._run_async`` helper.
        kwargs.pop("is_async", None)
        # Drop the original check_fn (which validates real API keys / env /
        # services). Without this, 16+ tools would be hidden from the LLM
        # because their production prerequisites aren't present in a mock
        # environment. With mocked handlers every tool is effectively
        # "available", so we force the check to always pass.
        kwargs.pop("check_fn", None)
        return original_register(
            name=name,
            toolset=toolset,
            schema=schema,
            handler=_make_mock_handler(name, schema),
            is_async=False,
            check_fn=lambda: True,
            **kwargs,
        )

    registry.register = mock_register  # type: ignore[method-assign]
    _installed = True
