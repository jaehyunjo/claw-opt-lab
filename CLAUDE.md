# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**claw-opt-lab** — minimal agent skeleton for LLM optimization experiments (routing, prompt assembly, RAG, cost). Forked from `NousResearch/hermes-agent` and stripped to ~1% of its code.

- 47 tool schemas preserved from upstream (realistic LLM tool surface)
- All tool handlers mocked (no real filesystem / network / process effects)
- 79 skills indexed into the system prompt
- 16 scenario demos with record/replay cassette pattern

Cloud backend: **Azure OpenAI** (`gpt-5-mini`), configured via `.env`.

## Common commands

```bash
./run.sh setup                          # create venv + install deps
./run.sh test                           # pytest (16 tests)
./run.sh demo                           # replay 16 scenarios (~0s)
./run.sh demo --mode record             # re-record every cassette against Azure
./run.sh demo --mode record --scenario vision,cronjob
./run.sh demo --mode live --prompt "..."
./run.sh demo --list                    # see scenarios + cassette presence
./run.sh repl                           # interactive main.py
```

Direct pytest also works: `source venv/bin/activate && pytest tests/ -v`.

## Bootstrap order (critical)

Enforced in `main.py`, `demo.py`, `tests/conftest.py`:

1. `core.shim.install_shim()` — meta_path finder for deleted packages + monkey-patches `tools.registry.registry.register` so every handler becomes a mock, `is_async=False`, `check_fn=lambda: True` (otherwise 16+ tools would be hidden because their real `check_requirements` fails).
2. `tools.registry.discover_builtin_tools()` — AST-scans `tools/*.py` and imports self-registering modules. Populates 47 tools.
3. Agent loop (`core.agent.run_conversation`) uses the registry and the client you pass in — `AzureOpenAI` for live/record, `ReplayClient` for replay.

If you add a new entry point, call `shim.install_shim()` before any import of `tools.*` tool modules.

## Architecture

| Path | Role |
|------|------|
| `core/shim.py`         | Mock import finder + registry patch + `_TOOL_OVERRIDES` (per-tool mock payloads for realistic returns). The single load-bearing file. |
| `core/agent.py`        | Sync agent loop: LLM → dispatch tool_calls → append results → repeat. Accepts `client` + `deployment` injection. |
| `core/azure.py`        | `AzureOpenAI` bootstrap from 4 env vars. |
| `core/cassette.py`     | `RecordingClient` wraps real client + saves responses; `ReplayClient` serves responses from a JSON file — both duck-type `client.chat.completions.create(...)`. |
| `core/prompt.py`       | System prompt assembly + skill listing. |
| `core/skill_loader.py` | `skills/**/SKILL.md` frontmatter scanner. |
| `tools/registry.py`    | Unchanged from upstream — tool registry, dispatch, schema lookup. |
| `tools/*.py` (21)      | Self-registering tool modules. Handlers get replaced at register time. |
| `hermes_constants.py`  | Path helpers that some tool files import. Kept verbatim; safe to trim later. |
| `demo.py`              | 16 Korean scenarios + summary table; `--mode replay|record|live`. |
| `cassettes/<n>.json`   | Recorded Azure response sequences — committed so replay works offline. |

## Things that will bite you

- **Shim must run first.** Importing any `tools.<foo>` before `install_shim()` causes `ImportError` because most tool files reference deleted modules.
- **Tool list differs from upstream.** The shim force-overrides `check_fn` to `True`, so `cronjob`, `vision_analyze`, `ha_*`, and `rl_*` are exposed even without their production prerequisites. This is deliberate — routing experiments need a fixed tool surface regardless of env.
- **Cassette drift.** If you change `core/prompt.py`, `demo.py` scenario prompts, or tool schemas, old cassettes may be out of sync (model decisions depend on those inputs). `ReplayClient` raises `CassetteExhaustedError` if the loop asks for more calls than were recorded — re-record with `--mode record`.
- **Azure rate limits.** `gpt-5-mini` has low TPM/RPM — the openai SDK silently retries 429 with `retry-after` delays, which can stretch a single call to 3+ minutes. Cassettes eliminate this in day-to-day use.
- **`.env` is gitignored.** Don't commit real Azure keys. `.env.example` has placeholders only.
- **No tool has real side effects.** Every `terminal`, `write_file`, `send_message`, `cronjob` call returns mock success in ~0ms without touching the host. Verified by test_shim.py + manual timing (e.g. `curl http://httpbin.org/delay/3` returns instantly via the mock).

## Adding / modifying

- **New scenario**: append a `Scenario(name, expect, expected_tools, prompt)` to `SCENARIOS` in `demo.py`, then `./run.sh demo --mode record --scenario <name>`.
- **Richer mock for a tool**: add a callable to `_TOOL_OVERRIDES` in `core/shim.py`. Returned dict is wrapped with `{ok, tool, mock}` defaults by the handler.
- **Add a tool**: drop a new file in `tools/` that calls `registry.register(...)` at module scope. AST-based discovery picks it up automatically — no import list to maintain.
- **Add a new LLM backend** (on-device): new `core/<backend>.py` exposing `get_client()` returning an object with `.chat.completions.create(**kwargs)`. Pass it to `run_conversation(client=...)`.

## Tests

- `test_shim.py` — mock handler validation; MagicMock import resolution; envelope shape
- `test_skill_loader.py` — frontmatter parsing; real `skills/` load
- `test_agent_loop.py` — loop with fake Azure client (plain / tool-call / max-iterations)
- `test_azure_integration.py` — real Azure round-trip (skipped without `AZURE_OPENAI_API_KEY`)

All tests run fast (<1s except the 3 integration tests ~15s total).
