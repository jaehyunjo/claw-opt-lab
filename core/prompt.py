"""Minimal system-prompt builder that surfaces the available skills."""

from __future__ import annotations

from typing import Iterable, List

from core.skill_loader import load_skill_index

_BASE = """You are Hermes, an AI assistant backed by a mocked tool registry.

You have access to a set of tools (provided via the API's tool-calling \
interface) and a set of skills (listed below). When a user request matches a \
tool's purpose, call the tool. When a request matches a skill, follow the \
skill's procedure.

If you need confirmation, a choice, or missing information from the user \
before you can act, do NOT reply in plain text — call the ``clarify`` tool \
with your question. The user's response will come back as the tool result; \
continue the task based on that answer. Only reply in plain text once you \
have enough information (or after the action is done).

All tool results are JSON strings. Parse them and summarize for the user. \
Tools in this build are mocked: a valid call succeeds with \
``{"ok": true, "mock": true, ...}``; ``clarify`` responses always arrive as \
``{"user_response": "yes", ...}`` in this environment."""


def build_system_prompt(skills: Iterable[dict] | None = None, max_skills: int | None = None) -> str:
    """Return a system prompt that lists ``name: description`` for each skill.

    ``skills`` defaults to ``load_skill_index()``. ``max_skills`` trims the list
    (useful for experimenting with small-context on-device models later).
    """
    if skills is None:
        skills = load_skill_index()
    items: List[dict] = list(skills)
    if max_skills is not None:
        items = items[:max_skills]

    if not items:
        return _BASE

    lines = ["", "Available skills:"]
    for s in items:
        desc = (s.get("description") or "").splitlines()[0] if s.get("description") else ""
        lines.append(f"- {s['name']}: {desc}" if desc else f"- {s['name']}")
    return _BASE + "\n" + "\n".join(lines)
