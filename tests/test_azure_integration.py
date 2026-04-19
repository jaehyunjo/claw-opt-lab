"""End-to-end smoke test against the real Azure OpenAI deployment.

Runs only when ``AZURE_OPENAI_API_KEY`` is set. Marked ``integration`` so a
standard ``pytest`` invocation skips it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("AZURE_OPENAI_API_KEY"),
        reason="AZURE_OPENAI_API_KEY not set",
    ),
]


def test_azure_client_plain_chat():
    """A trivial round-trip: the model must return a non-empty string."""
    from core.azure import get_client, get_deployment

    client = get_client()
    resp = client.chat.completions.create(
        model=get_deployment(),
        messages=[
            {"role": "system", "content": "Reply with a single short sentence."},
            {"role": "user", "content": "Say hi."},
        ],
    )
    content = resp.choices[0].message.content or ""
    assert content.strip(), "expected non-empty assistant response"


def test_agent_loop_real_azure_no_tools():
    """Full agent loop end-to-end without any tool calls."""
    from core.agent import run_conversation

    result = run_conversation(
        user_message="Reply with exactly: READY",
        system_prompt="You are a terse test bot. Reply as instructed.",
        enabled_tools=[],
        max_iterations=3,
    )
    assert result["final"].strip(), f"empty final response: {result}"
    assert result["iterations"] >= 1


def test_agent_loop_real_azure_with_mock_tool():
    """End-to-end with a tool available; model may or may not call it, but the loop must terminate."""
    from core.agent import run_conversation
    from tools.registry import registry

    assert registry.get_entry("read_file") is not None

    result = run_conversation(
        user_message=(
            "Please call the read_file tool with path='/etc/hostname', "
            "then briefly summarize what it returned."
        ),
        system_prompt="You are a test bot. Use the provided tools when asked.",
        enabled_tools=["read_file"],
        max_iterations=5,
    )
    assert result["final"].strip(), f"empty final response: {result}"
    assert result["iterations"] <= 5
