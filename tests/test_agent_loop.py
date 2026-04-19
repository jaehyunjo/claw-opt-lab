"""Agent loop runs tool calls to completion using a fake Azure client."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.agent import run_conversation
from tools.registry import registry


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _completion(msg):
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _tool_call(name: str, args: dict, tc_id: str = "call_1"):
    return SimpleNamespace(
        id=tc_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def test_loop_returns_plain_text_without_tool_calls():
    client = MagicMock()
    client.chat.completions.create.return_value = _completion(_msg(content="hello"))
    result = run_conversation(
        user_message="hi",
        system_prompt="You are a bot.",
        enabled_tools=[],
        client=client,
        deployment="gpt-5-mini",
    )
    assert result["final"] == "hello"
    assert result["iterations"] == 1
    # 3 messages: system, user, assistant
    assert len(result["messages"]) == 3
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][1]["role"] == "user"
    assert result["messages"][2]["role"] == "assistant"


def test_loop_dispatches_tool_call_then_responds():
    """Model asks for read_file, gets a mock result, then produces a final answer."""
    tool_name = "read_file"
    assert registry.get_entry(tool_name), "read_file must be registered"

    args = {"path": "/tmp/foo.txt"}
    first = _completion(_msg(tool_calls=[_tool_call(tool_name, args)]))
    second = _completion(_msg(content="file read complete"))
    client = MagicMock()
    client.chat.completions.create.side_effect = [first, second]

    observed = {"tool": None, "result": None, "text": None}

    def on_tool_call(name, a):
        observed["tool"] = (name, a)

    def on_tool_result(name, r):
        observed["result"] = (name, r)

    def on_assistant_text(t):
        observed["text"] = t

    result = run_conversation(
        user_message="read /tmp/foo.txt",
        system_prompt="You are a bot.",
        enabled_tools=[tool_name],
        client=client,
        deployment="gpt-5-mini",
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
        on_assistant_text=on_assistant_text,
    )

    assert result["final"] == "file read complete"
    assert result["iterations"] == 2
    assert observed["tool"] == (tool_name, args)
    assert observed["text"] == "file read complete"

    # The tool result should be a mock success payload for this tool.
    # Tools with overrides return richer shapes (no ``args`` echo), so only
    # the envelope fields are guaranteed.
    parsed = json.loads(observed["result"][1])
    assert parsed["ok"] is True
    assert parsed["tool"] == tool_name
    assert parsed["mock"] is True

    # Ensure the message log contains a tool-role message
    roles = [m["role"] for m in result["messages"]]
    assert roles.count("tool") == 1


def test_loop_respects_max_iterations():
    """If the model never stops asking for tools, we bail out."""
    tc = _tool_call("read_file", {"path": "/tmp/x"}, tc_id=f"call_x")
    endless = _completion(_msg(tool_calls=[tc]))
    client = MagicMock()
    client.chat.completions.create.return_value = endless

    result = run_conversation(
        user_message="loop forever",
        system_prompt="You are a bot.",
        enabled_tools=["read_file"],
        max_iterations=3,
        client=client,
        deployment="gpt-5-mini",
    )
    assert result["iterations"] == 3
    assert result["final"] == ""
