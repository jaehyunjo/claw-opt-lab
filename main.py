"""REPL entry point for the Hermes skeleton.

Usage:
    python main.py

Requires the four Azure env vars (see ``.env.example``). Load ``.env`` in the
current directory automatically.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from core import shim  # noqa: E402
shim.install_shim()

from tools.registry import registry, discover_builtin_tools  # noqa: E402
from core.agent import run_conversation  # noqa: E402
from core.azure import AzureConfigError  # noqa: E402
from core.prompt import build_system_prompt  # noqa: E402


def _print_tool_call(name: str, args: dict) -> None:
    print(f"  → {name}({json.dumps(args, ensure_ascii=False)})")


def _print_tool_result(name: str, result: str) -> None:
    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        parsed = result
    print(f"  ← {json.dumps(parsed, ensure_ascii=False)}")


def _print_assistant(text: str) -> None:
    print(f"\n[assistant]\n{text}\n")


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(name)s: %(message)s")

    discover_builtin_tools()
    system_prompt = build_system_prompt()
    tool_names = registry.get_all_tool_names()

    print(f"Hermes skeleton — {len(tool_names)} tools registered")
    print(f"System prompt: {len(system_prompt)} chars")
    print("Type a message; Ctrl-D or 'exit' to quit.\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line or line in {"exit", "quit"}:
            return 0
        try:
            run_conversation(
                user_message=line,
                system_prompt=system_prompt,
                enabled_tools=tool_names,
                on_assistant_text=_print_assistant,
                on_tool_call=_print_tool_call,
                on_tool_result=_print_tool_result,
            )
        except AzureConfigError as e:
            print(f"Config error: {e}", file=sys.stderr)
            return 2
        except Exception as e:  # surfacing any runtime error cleanly for the REPL
            print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
