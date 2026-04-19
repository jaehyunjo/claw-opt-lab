"""REPL entry point for the Hermes skeleton.

Usage:
    python main.py                    # Azure (default)
    python main.py --backend gemma    # HuggingFace Gemma

Requires either the four ``AZURE_OPENAI_*`` env vars (see ``.env.example``)
or ``HUGGINGFACE_TOKEN`` for the Gemma backend. ``.env`` in the current
directory is loaded automatically.
"""

from __future__ import annotations

import argparse
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
from core.backends import ALL_BACKENDS, BACKEND_AZURE, get_backend  # noqa: E402
from core.gemma import GemmaConfigError  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Hermes skeleton REPL")
    parser.add_argument("--backend", choices=ALL_BACKENDS, default=BACKEND_AZURE,
                        help="LLM backend: azure (default) or gemma (HuggingFace Inference)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(name)s: %(message)s")

    discover_builtin_tools()
    system_prompt = build_system_prompt()
    tool_names = registry.get_all_tool_names()

    try:
        client, deployment = get_backend(args.backend)
    except (AzureConfigError, GemmaConfigError) as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2

    print(f"Hermes skeleton — {len(tool_names)} tools registered")
    print(f"Backend: {args.backend}  ({deployment})")
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
                client=client,
                deployment=deployment,
                on_assistant_text=_print_assistant,
                on_tool_call=_print_tool_call,
                on_tool_result=_print_tool_result,
            )
        except Exception as e:  # surfacing any runtime error cleanly for the REPL
            print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
