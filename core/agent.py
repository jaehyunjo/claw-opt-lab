"""Minimal agent loop: call LLM, dispatch tool calls via the registry, repeat."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from core.azure import get_client, get_deployment
from tools.registry import registry

logger = logging.getLogger(__name__)

Callback = Optional[Callable[..., None]]


def _assistant_message(msg) -> Dict[str, Any]:
    """Serialize the SDK ChatCompletionMessage into a plain dict for replay."""
    out: Dict[str, Any] = {"role": "assistant", "content": msg.content}
    if getattr(msg, "tool_calls", None):
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in msg.tool_calls
        ]
    return out


def run_conversation(
    user_message: str,
    system_prompt: str,
    enabled_tools: List[str],
    max_iterations: int = 10,
    max_completion_tokens: Optional[int] = None,
    on_assistant_text: Callback = None,
    on_tool_call: Callback = None,
    on_tool_result: Callback = None,
    client=None,
    deployment: Optional[str] = None,
) -> Dict[str, Any]:
    """Drive one user turn through the agent loop.

    Returns ``{"final": str, "messages": [...], "iterations": int}``.

    ``client`` / ``deployment`` overrides exist for tests that inject a fake
    Azure client.
    """
    cli = client or get_client()
    model = deployment or get_deployment()

    tool_schemas = registry.get_definitions(set(enabled_tools)) if enabled_tools else []

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    iterations = 0
    final_text = ""

    while iterations < max_iterations:
        iterations += 1
        kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        if tool_schemas:
            kwargs["tools"] = tool_schemas
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens
        resp = cli.chat.completions.create(**kwargs)

        msg = resp.choices[0].message
        messages.append(_assistant_message(msg))

        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            for tc in tool_calls:
                tool_name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
                if on_tool_call:
                    on_tool_call(tool_name, args)
                result = registry.dispatch(tool_name, args)
                if on_tool_result:
                    on_tool_result(tool_name, result)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            continue  # let the model react to the tool results

        final_text = msg.content or ""
        if on_assistant_text:
            on_assistant_text(final_text)
        break
    else:
        logger.warning("Agent loop hit max_iterations=%d", max_iterations)

    return {"final": final_text, "messages": messages, "iterations": iterations}
