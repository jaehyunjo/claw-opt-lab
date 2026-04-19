"""Gemma adapter tests — fake the ``litert_lm.Engine`` to keep it hermetic.

The adapter passes tools natively into ``create_conversation`` and relies
on a ``ToolEventHandler`` to intercept the model's tool-call decision.
These tests simulate both branches (tool-call intercepted vs. plain-text
reply) without touching the real runtime.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from core import gemma


class _FakeConversation:
    """Stand-in for ``litert_lm.Conversation.send_message``.

    Mimics the real flow: when the engine decides to call a tool, it first
    asks the handler for approval (we capture, return True) and then
    invokes the synthesized callable — which raises ``_ToolCallIntercepted``
    to abort generation. If no tool call is scripted, returns canned text.
    """

    def __init__(self, *, handler, tools, tool_call: Optional[Dict[str, Any]],
                 reply_text: str, log: List[Dict[str, Any]]):
        self._handler = handler
        self._tools = tools or []
        self._tool_call = tool_call
        self._reply_text = reply_text
        self._log = log

    def send_message(self, message):
        self._log.append({"kind": "send_message", "message": message})
        if self._tool_call and self._handler is not None:
            self._handler.approve_tool_call(self._tool_call)
            # Find the synthesized callable matching the requested name
            # and call it — the real engine would do this internally.
            name = self._tool_call.get("name")
            for t in self._tools:
                if getattr(t, "__name__", None) == name:
                    # Trigger the abort exactly like litert_lm would.
                    t()  # will raise _ToolCallIntercepted
            # If no callable matched, still raise the sentinel.
            raise gemma._ToolCallIntercepted(name or "unknown")
        return {"role": "assistant", "content": [{"type": "text", "text": self._reply_text}]}


class _FakeEngine:
    def __init__(self, *, reply_text: str = "",
                 tool_call: Optional[Dict[str, Any]] = None):
        self.reply_text = reply_text
        self.tool_call = tool_call
        self.calls: List[Dict[str, Any]] = []

    def create_conversation(self, messages=None, tools=None,
                            tool_event_handler=None, **_):
        tools_list = list(tools or [])
        self.calls.append({
            "kind": "create_conversation",
            "preface": list(messages or []),
            "tools": tools_list,
            "handler": tool_event_handler,
        })
        return _FakeConversation(
            handler=tool_event_handler,
            tools=tools_list,
            tool_call=self.tool_call,
            reply_text=self.reply_text,
            log=self.calls,
        )


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch):
    monkeypatch.setattr(gemma, "_SHARED_ENGINE", None)
    monkeypatch.setattr(gemma, "_SHARED_ENGINE_PATH", None)


@pytest.fixture(autouse=True)
def _stub_handler_class(monkeypatch):
    """Replace the litert_lm-backed ToolEventHandler factory with a plain
    Python stand-in so tests don't need the native runtime."""

    class _CapturingHandler:
        def __init__(self):
            self.captured = None

        def approve_tool_call(self, tool_call):
            self.captured = gemma._normalize_tool_call(tool_call)
            return True

        def process_tool_response(self, tool_response):
            return tool_response

    monkeypatch.setattr(gemma, "_make_handler_class",
                        lambda: _CapturingHandler)


def _install_fake_engine(monkeypatch, *, reply_text="", tool_call=None):
    fake = _FakeEngine(reply_text=reply_text, tool_call=tool_call)

    def _fake_loader(_path):
        return fake

    monkeypatch.setattr(gemma, "_get_shared_engine", _fake_loader)
    return fake


def test_plain_text_reply_becomes_message_content(monkeypatch):
    fake = _install_fake_engine(monkeypatch, reply_text="hi there")
    client = gemma.GemmaClient("/fake/model.litertlm")

    resp = client.chat.completions.create(
        model="gemma",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ],
    )

    msg = resp.choices[0].message
    assert msg.content == "hi there"
    assert msg.tool_calls is None

    create_call = fake.calls[0]
    assert create_call["kind"] == "create_conversation"
    assert [m["role"] for m in create_call["preface"]] == ["system"]
    assert create_call["tools"] == []
    send_call = fake.calls[1]
    assert send_call["kind"] == "send_message"
    assert send_call["message"] == "hello"


def test_intercepted_tool_call_surfaces_as_tool_calls(monkeypatch):
    captured_call = {"name": "send_message",
                     "args": {"recipient": "a@b", "body": "hi"}}
    fake = _install_fake_engine(monkeypatch, tool_call=captured_call)
    client = gemma.GemmaClient("/fake/model.litertlm")

    tool_schema = {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send an email",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["recipient", "body"],
            },
        },
    }

    resp = client.chat.completions.create(
        model="gemma",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "email bob"},
        ],
        tools=[tool_schema],
    )

    msg = resp.choices[0].message
    assert msg.content is None
    assert msg.tool_calls is not None and len(msg.tool_calls) == 1

    tc = msg.tool_calls[0]
    assert tc.id.startswith("call_")
    assert tc.type == "function"
    assert tc.function.name == "send_message"
    parsed = json.loads(tc.function.arguments)
    assert parsed == {"recipient": "a@b", "body": "hi"}

    # The engine received a synthesized callable whose signature matches
    # the JSON schema (so the chat template can render the tool).
    synth = fake.calls[0]["tools"][0]
    assert callable(synth)
    assert synth.__name__ == "send_message"
    import inspect
    params = inspect.signature(synth).parameters
    assert set(params) == {"recipient", "body"}
    assert fake.calls[0]["handler"] is not None


def test_tool_role_messages_are_folded_into_next_user_turn(monkeypatch):
    fake = _install_fake_engine(monkeypatch, reply_text="ack")
    client = gemma.GemmaClient("/fake/model.litertlm")

    client.chat.completions.create(
        model="gemma",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "email bob"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "send_message",
                                          "arguments": "{\"recipient\":\"bob\"}"}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "send_message",
             "content": "{\"ok\": true}"},
            {"role": "user", "content": "thanks"},
        ],
    )

    preface = fake.calls[0]["preface"]
    roles = [m["role"] for m in preface]
    assert roles == ["system", "user", "assistant"]
    last = fake.calls[1]["message"]
    assert "tool send_message result" in last
    assert "thanks" in last


def test_args_come_through_even_when_handler_gets_string_args(monkeypatch):
    _install_fake_engine(monkeypatch,
                        tool_call={"name": "clarify",
                                   "args": "{\"question\": \"?\"}"})
    client = gemma.GemmaClient("/fake/model.litertlm")
    resp = client.chat.completions.create(
        model="gemma",
        messages=[{"role": "user", "content": "x"}],
        tools=[{"type": "function",
                "function": {"name": "clarify", "parameters": {}}}],
    )
    tc = resp.choices[0].message.tool_calls[0]
    assert tc.function.name == "clarify"
    assert json.loads(tc.function.arguments) == {"question": "?"}


def test_gemma_quote_tokens_are_stripped_from_args(monkeypatch):
    leaky = {"name": "send_message",
             "args": {"target": "<|\"|>gmail:a@b<|\"|>",
                      "message": "<|\"|>hi<|\"|>"}}
    _install_fake_engine(monkeypatch, tool_call=leaky)
    client = gemma.GemmaClient("/fake/model.litertlm")
    resp = client.chat.completions.create(
        model="gemma",
        messages=[{"role": "user", "content": "x"}],
        tools=[{"type": "function",
                "function": {"name": "send_message",
                             "parameters": {"type": "object",
                                            "properties": {
                                                "target": {"type": "string"},
                                                "message": {"type": "string"}}}}}],
    )
    parsed = json.loads(resp.choices[0].message.tool_calls[0].function.arguments)
    assert parsed == {"target": "gmail:a@b", "message": "hi"}


def test_empty_messages_raises(monkeypatch):
    _install_fake_engine(monkeypatch, reply_text="")
    client = gemma.GemmaClient("/fake/model.litertlm")
    with pytest.raises(ValueError):
        client.chat.completions.create(model="gemma", messages=[])
