"""Gemma (mobile E2B) backend — loads the on-device ``.litertlm`` bundle
through the desktop ``litert-lm-api`` Python wheel (macOS / Linux, CPU
inference). No phone required.

The bundle comes from ``litert-community/gemma-4-E2B-it-litert-lm``; the
default file is ``gemma-4-E2B-it.litertlm`` (2.58 GB). First run downloads
it via ``huggingface_hub.hf_hub_download`` using ``HUGGINGFACE_TOKEN``
(gated model — license must be accepted on the HF UI). Set
``GEMMA_MODEL_PATH`` to skip the download and point at a local copy.

The agent loop in ``core/agent.py`` only touches this duck-typed surface::

    client.chat.completions.create(model=..., messages=..., tools=..., ...)
        → resp.choices[0].message.{content, tool_calls}
        → tc.id, tc.function.name, tc.function.arguments

Tool-calling path: Gemma-4 was trained for function calling via a
``<|tool_call>call:name{args}<tool_call|>`` format that
``litert_lm.Engine.create_conversation(tools=[...])`` renders automatically
from OpenAI-schema-shaped ``Tool`` objects. We wrap each JSON schema in a
``_SchemaTool`` and attach a ``ToolEventHandler`` that **intercepts** the
call (returns False in ``approve_tool_call``) so the model's tool choice
surfaces up to our agent loop — which then dispatches through the mock
registry and comes back for another turn. This mirrors the Azure path:
one ``create_conversation`` call per agent iteration, tool_calls appear
in the assistant message, the outer loop handles dispatch.

Env vars:

    HUGGINGFACE_TOKEN       — required for HF download (also accepts HF_TOKEN)
    GEMMA_HF_REPO           — default ``litert-community/gemma-4-E2B-it-litert-lm``
    GEMMA_MODEL_FILE        — default ``gemma-4-E2B-it.litertlm``
    GEMMA_MODEL_PATH        — optional absolute path to a local bundle
    GEMMA_MAX_NUM_TOKENS    — Engine context budget (default 16384)
    GEMMA_BACKEND           — CPU | GPU (default CPU; GPU uses Metal/WebGPU on macOS)
    GEMMA_TOOL_DESC_MAX     — per-tool description char budget for synthesized
                              callables (default 120). Lower = faster prefill,
                              less tool-choice precision. 0 disables truncation.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_HF_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
_DEFAULT_MODEL_FILE = "gemma-4-E2B-it.litertlm"
_DEFAULT_MAX_NUM_TOKENS = 16384

_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _ROOT / "models"


class GemmaConfigError(RuntimeError):
    """Raised when required HuggingFace env vars are missing."""


# ── message flattening for LiteRT-LM ───────────────────────────────────────

# LiteRT-LM's conversation preface expects OpenAI-style messages with
# roles {"system","user","assistant"}. The agent loop also emits "tool"
# role messages carrying a tool-call result. We fold those into the
# following user turn so the model sees them as conversation context
# (rather than trying to rely on a "tool" role the chat template may not
# handle identically to OpenAI's rendering).

def _flatten_for_litertlm(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pending_tool_text: List[str] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "tool":
            pending_tool_text.append(f"[tool {m.get('name','?')} result]\n{content}")
            continue
        if role == "assistant":
            tool_calls = m.get("tool_calls") or []
            if not content and tool_calls:
                tc = tool_calls[0]
                fn = (tc or {}).get("function") or {}
                content = json.dumps(
                    {"tool": fn.get("name"), "args": _safe_json(fn.get("arguments"))},
                    ensure_ascii=False,
                )
        if pending_tool_text and role == "user":
            content = "\n\n".join(pending_tool_text + [content])
            pending_tool_text = []
        elif pending_tool_text:
            out.append({"role": "user", "content": "\n\n".join(pending_tool_text)})
            pending_tool_text = []
        out.append({"role": role or "user", "content": content})
    if pending_tool_text:
        out.append({"role": "user", "content": "\n\n".join(pending_tool_text)})
    return out


def _safe_json(s: Any) -> Any:
    if isinstance(s, str):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return s
    return s


def _extract_text(resp: Any) -> str:
    """Normalize what ``Conversation.send_message`` returns into plain text."""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        parts = resp.get("content")
        if isinstance(parts, str):
            return parts
        if isinstance(parts, list):
            pieces: List[str] = []
            for p in parts:
                if isinstance(p, dict) and p.get("type") == "text":
                    pieces.append(p.get("text") or "")
                elif isinstance(p, str):
                    pieces.append(p)
            return "".join(pieces)
    return str(resp or "")


def _to_openai_response(content: Optional[str],
                        captured_tool_call: Optional[Dict[str, Any]]):
    """Build a ``resp.choices[0].message`` object matching the OpenAI SDK shape."""
    tool_calls = None
    if captured_tool_call is not None:
        args_dict = captured_tool_call.get("args") or captured_tool_call.get("arguments") or {}
        if isinstance(args_dict, str):
            args_dict = _safe_json(args_dict) or {}
        tc = SimpleNamespace(
            id=f"call_{uuid.uuid4().hex[:12]}",
            type="function",
            function=SimpleNamespace(
                name=captured_tool_call["name"],
                arguments=json.dumps(args_dict, ensure_ascii=False),
            ),
        )
        tool_calls = [tc]
    msg = SimpleNamespace(
        content=None if tool_calls else (content or ""),
        tool_calls=tool_calls,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


# ── SDK-shape wrapper ──────────────────────────────────────────────────────

class _Endpoint:
    def __init__(self, create_fn):
        self.create = create_fn


class _Chat:
    def __init__(self, create_fn):
        self.completions = _Endpoint(create_fn)


_SHARED_ENGINE: Any = None
_SHARED_ENGINE_PATH: Optional[str] = None


def _get_shared_engine(model_path: str):
    """Load (or return the cached) ``litert_lm.Engine``.

    Reloading a 2.58 GB bundle between scenarios is prohibitively slow, so
    the engine lives for the life of the process. If the caller requests
    a different path than the one currently loaded, we rebuild it.
    """
    global _SHARED_ENGINE, _SHARED_ENGINE_PATH
    if _SHARED_ENGINE is not None and _SHARED_ENGINE_PATH == model_path:
        return _SHARED_ENGINE
    import litert_lm
    max_tokens = int(os.environ.get("GEMMA_MAX_NUM_TOKENS", _DEFAULT_MAX_NUM_TOKENS))
    backend_name = os.environ.get("GEMMA_BACKEND", "").upper().strip()
    kwargs: Dict[str, Any] = {"max_num_tokens": max_tokens}
    if backend_name in ("CPU", "GPU"):
        kwargs["backend"] = getattr(litert_lm.Backend, backend_name)
    logger.info("Loading LiteRT-LM bundle (max_num_tokens=%d, backend=%s): %s",
                max_tokens, backend_name or "default", model_path)
    _SHARED_ENGINE = litert_lm.Engine(model_path, **kwargs)
    _SHARED_ENGINE_PATH = model_path
    return _SHARED_ENGINE


_JSON_TYPE_TO_PY = {
    "string": str, "integer": int, "number": float,
    "boolean": bool, "array": list, "object": dict,
}


class _ToolCallIntercepted(RuntimeError):
    """Raised from a synthesized callable to abort ``send_message`` the
    moment the engine tries to dispatch the model's chosen tool. The
    outer adapter catches it and surfaces the captured call up to the
    agent loop as an OpenAI-shape ``tool_calls`` entry."""


def _truncate_desc(text: str, limit: int) -> str:
    text = (text or "").strip().splitlines()[0] if text else ""
    if limit and len(text) > limit:
        return text[: max(0, limit - 1)].rstrip() + "…"
    return text


def _synthesize_callable(schema: Dict[str, Any]):
    """Build a no-op Python callable whose signature + docstring match
    ``schema['function']`` closely enough that
    ``litert_lm.tools.tool_from_function`` produces an equivalent OpenAI
    schema when introspecting it. The function body is never executed —
    our ``ToolEventHandler`` cancels dispatch before the engine calls it.

    Tool descriptions and per-parameter descriptions dominate prefill cost
    on Gemma E2B (on the order of 8K tokens across the 47-tool registry),
    so we truncate each to ``GEMMA_TOOL_DESC_MAX`` chars (default 120) and
    keep only the first line. Set ``GEMMA_TOOL_DESC_MAX=0`` to disable.
    """
    import inspect

    desc_limit = int(os.environ.get("GEMMA_TOOL_DESC_MAX", "120"))

    fn = (schema or {}).get("function") or {}
    name = fn.get("name") or "tool"
    description = _truncate_desc(fn.get("description") or "", desc_limit) \
        or f"{name} (mocked)"
    params = (fn.get("parameters") or {}).get("properties") or {}
    required = set((fn.get("parameters") or {}).get("required") or [])

    py_params: List[inspect.Parameter] = []
    arg_doc_lines: List[str] = []
    for pname, pschema in params.items():
        ptype = _JSON_TYPE_TO_PY.get((pschema or {}).get("type"), str)
        default = inspect.Parameter.empty if pname in required else None
        py_params.append(inspect.Parameter(
            pname,
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=default,
            annotation=ptype,
        ))
        pdesc = _truncate_desc((pschema or {}).get("description") or "", desc_limit)
        if pdesc:
            arg_doc_lines.append(f"    {pname}: {pdesc}")

    doc_parts = [description]
    if arg_doc_lines:
        doc_parts += ["", "Args:"] + arg_doc_lines

    def _impl(**_kwargs):
        # Our ToolEventHandler captures the call before this runs; raise so
        # the engine aborts instead of retrying after a denial.
        raise _ToolCallIntercepted(name)

    _impl.__name__ = name
    _impl.__qualname__ = name
    _impl.__doc__ = "\n".join(doc_parts)
    _impl.__signature__ = inspect.Signature(parameters=py_params)
    return _impl


def _make_handler_class():
    """Build a ``litert_lm.ToolEventHandler`` subclass lazily (the abstract
    base is only available once the wheel is loaded)."""
    import litert_lm

    class _CapturingHandler(litert_lm.ToolEventHandler):
        def __init__(self):
            super().__init__()
            self.captured: Optional[Dict[str, Any]] = None

        def approve_tool_call(self, tool_call) -> bool:
            self.captured = _normalize_tool_call(tool_call)
            # Return True so the engine proceeds to ``execute`` — the
            # synthesized callable raises ``_ToolCallIntercepted`` there,
            # which propagates out of ``send_message`` cleanly. Returning
            # False instead makes the engine inject a "tool denied"
            # message and retry, which produces malformed tool calls.
            return True

        def process_tool_response(self, tool_response):
            return tool_response

    return _CapturingHandler


_GEMMA_QUOTE_TOKEN = '<|"|>'


def _scrub_gemma_tokens(value: Any) -> Any:
    """Strip Gemma's internal ``<|"|>`` quote tokens that can leak into
    tool-call arguments when the chat template's quoting markers survive
    detokenization."""
    if isinstance(value, str):
        return value.replace(_GEMMA_QUOTE_TOKEN, "").strip()
    if isinstance(value, dict):
        return {k: _scrub_gemma_tokens(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_gemma_tokens(v) for v in value]
    return value


def _normalize_tool_call(tc: Any) -> Dict[str, Any]:
    """Normalize what ``approve_tool_call`` hands us to ``{name, args}``."""
    if isinstance(tc, dict):
        name = tc.get("name") or tc.get("tool") or (tc.get("function") or {}).get("name")
        args = (
            tc.get("args")
            or tc.get("arguments")
            or (tc.get("function") or {}).get("arguments")
            or {}
        )
    else:
        name = getattr(tc, "name", None) or getattr(tc, "tool", None)
        args = getattr(tc, "args", None) or getattr(tc, "arguments", {}) or {}
    return {"name": name, "args": _scrub_gemma_tokens(args)}


class GemmaClient:
    """Adapter over ``litert_lm.Engine`` that exposes the slice of the
    OpenAI SDK the agent loop calls (``client.chat.completions.create``)."""

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._engine = _get_shared_engine(model_path)
        self.chat = _Chat(self._create)

    def _create(self, **kwargs):
        messages: List[Dict[str, Any]] = list(kwargs.get("messages") or [])
        tools_schemas: List[Dict[str, Any]] = list(kwargs.get("tools") or [])
        if not messages:
            raise ValueError("messages must not be empty")

        messages = _flatten_for_litertlm(messages)
        preface = messages[:-1]
        last = messages[-1]

        tool_callables: List[Any] = []
        handler = None
        if tools_schemas:
            tool_callables = [_synthesize_callable(s) for s in tools_schemas]
            handler = _make_handler_class()()

        logger.debug(
            "Gemma call: preface=%d last_role=%s tools=%d",
            len(preface), last.get("role"), len(tools_schemas),
        )
        conv = self._engine.create_conversation(
            messages=preface,
            tools=tool_callables or None,
            tool_event_handler=handler,
        )

        try:
            raw = conv.send_message(last.get("content") or "")
        except _ToolCallIntercepted:
            # Expected path: our synthesized callable aborted the engine
            # once the model committed to a tool call; the handler has
            # the details.
            return _to_openai_response("", handler.captured if handler else None)
        except RuntimeError as e:
            # LiteRT-LM occasionally raises ``Failed to parse tool calls``
            # when the model emits mixed prose + a malformed ``<|tool_call>``
            # block. If we already captured a call via the handler, use it.
            # Otherwise, treat the generation as empty text so the outer
            # agent loop can finish the scenario cleanly rather than
            # aborting mid-record.
            if handler is not None and handler.captured:
                return _to_openai_response("", handler.captured)
            logger.warning("LiteRT-LM generation failed; returning empty text. %s", e)
            return _to_openai_response("", None)
        text = _extract_text(raw)

        captured = handler.captured if handler else None
        return _to_openai_response(text, captured)


# ── model resolution ───────────────────────────────────────────────────────

def _resolve_model_path() -> str:
    """Return a local path to the ``.litertlm`` bundle, downloading if needed."""
    local = os.environ.get("GEMMA_MODEL_PATH")
    if local:
        p = Path(local).expanduser()
        if not p.exists():
            raise GemmaConfigError(f"GEMMA_MODEL_PATH does not exist: {p}")
        return str(p)

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise GemmaConfigError(
            "HUGGINGFACE_TOKEN (or HF_TOKEN) must be set to download the "
            "Gemma LiteRT-LM bundle, or set GEMMA_MODEL_PATH to a local file."
        )

    repo_id = os.environ.get("GEMMA_HF_REPO", _DEFAULT_HF_REPO)
    filename = os.environ.get("GEMMA_MODEL_FILE", _DEFAULT_MODEL_FILE)
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download
    logger.info("Resolving %s / %s via HuggingFace (cache: %s)",
                repo_id, filename, _MODELS_DIR)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        cache_dir=str(_MODELS_DIR),
    )
    return path


# ── module-level entry points (mirror core/azure.py) ───────────────────────

def get_client() -> GemmaClient:
    return GemmaClient(_resolve_model_path())


def get_deployment() -> str:
    return os.environ.get("GEMMA_MODEL_FILE", _DEFAULT_MODEL_FILE)
