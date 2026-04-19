# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**claw-opt-lab** — minimal agent skeleton for LLM optimization experiments (routing, prompt assembly, RAG, cost). Forked from `NousResearch/hermes-agent` and stripped to ~1% of its code.

- 47 tool schemas preserved from upstream (realistic LLM tool surface)
- All tool handlers mocked (no real filesystem / network / process effects)
- 79 skills indexed into the system prompt
- 16 scenario demos with record/replay cassette pattern

Backends (switchable via `--backend {azure,gemma}`):

- **`azure`** — Azure OpenAI `gpt-5-mini` via the `openai` SDK. Default.
- **`gemma`** — local CPU inference of `gemma-4-E2B-it.litertlm` (2.58 GB) through `litert-lm-api`. Same weights `gemmaclaw` ships on-device. First run downloads the gated bundle from `litert-community/gemma-4-E2B-it-litert-lm` via `huggingface_hub` using `HUGGINGFACE_TOKEN` (user must accept the Gemma license on the HF UI).

Both backends expose the same OpenAI-shape `chat.completions.create` so the agent loop, cassettes, and tests don't care which one is driving.

## Common commands

```bash
./run.sh setup                          # create venv + install deps
./run.sh test                           # pytest (22 tests)
./run.sh demo                           # Azure replay — 16 scenarios (~0s)
./run.sh demo --backend gemma           # Gemma replay (offline)
./run.sh demo --mode record             # re-record every Azure cassette
./run.sh demo --backend gemma --mode record   # re-record every Gemma cassette
./run.sh demo --backend gemma --mode live --scenario email
./run.sh demo --mode live --prompt "..."
./run.sh demo --list                    # see scenarios + cassette presence (per --backend)
./run.sh repl [--backend gemma]         # interactive main.py
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
| `core/gemma.py`        | `GemmaClient` — loads `gemma-4-E2B-it.litertlm` via `litert-lm-api`, synthesizes Python callables from the 47 JSON tool schemas so the Gemma chat template renders native `<\|tool_call>` tokens, and uses a `ToolEventHandler` returning `False` to capture the tool choice as OpenAI-shape `tool_calls`. Shared `Engine` singleton (per-scenario reload is too slow at 2.58 GB). |
| `core/backends.py`     | Factory `get_backend("azure"|"gemma") → (client, deployment)`. |
| `core/cassette.py`     | `RecordingClient` wraps any client + saves responses; `ReplayClient` serves responses from a JSON file — both duck-type `client.chat.completions.create(...)`. Backend-agnostic. |
| `core/prompt.py`       | System prompt assembly + skill listing (`max_skills=` trims the list; Gemma uses 10). |
| `core/skill_loader.py` | `skills/**/SKILL.md` frontmatter scanner. |
| `tools/registry.py`    | Unchanged from upstream — tool registry, dispatch, schema lookup. |
| `tools/*.py` (21)      | Self-registering tool modules. Handlers get replaced at register time. |
| `hermes_constants.py`  | Path helpers that some tool files import. Kept verbatim; safe to trim later. |
| `demo.py`              | 16 Korean scenarios + summary table; `--mode replay|record|live`, `--backend azure|gemma`. |
| `cassettes/<backend>/<n>.json` | Per-backend recorded response sequences — committed so replay works offline. |
| `models/` (gitignored) | Cached `.litertlm` bundles (2.58 GB for the default Gemma). |

## Things that will bite you

- **Shim must run first.** Importing any `tools.<foo>` before `install_shim()` causes `ImportError` because most tool files reference deleted modules.
- **Tool list differs from upstream.** The shim force-overrides `check_fn` to `True`, so `cronjob`, `vision_analyze`, `ha_*`, and `rl_*` are exposed even without their production prerequisites. This is deliberate — routing experiments need a fixed tool surface regardless of env.
- **Cassette drift is per-backend.** Cassettes live under `cassettes/azure/` or `cassettes/gemma/` — different backends produce different responses, so they're not interchangeable. If you change `core/prompt.py`, `demo.py` scenario prompts, or tool schemas, old cassettes may be out of sync. `ReplayClient` raises `CassetteExhaustedError` when the loop asks for more calls than were recorded — re-record with `--mode record`.
- **Azure rate limits.** `gpt-5-mini` has low TPM/RPM — the openai SDK silently retries 429 with `retry-after` delays, which can stretch a single call to 3+ minutes. Cassettes eliminate this in day-to-day use.
- **Gemma context window.** `gemma-4-E2B-it.litertlm` is trained at 4K; we set `Engine(max_num_tokens=16384)` but the model degrades past its trained length. For Gemma the system prompt is trimmed to 10 skills (see `demo.py`'s `max_skills` / `GEMMA_MAX_SKILLS`). The full 79-skill listing stays on Azure. If Gemma emits gibberish, check `core/prompt.py` didn't grow and that tool descriptions are concise.
- **Gemma model load is slow (~60 s, 2.58 GB → RAM).** `core/gemma.py` keeps a module-level `_SHARED_ENGINE` so scenario N+1 reuses the Engine from scenario N. A process that creates two different model paths rebuilds the Engine. Don't import `core/gemma.py` at module load time — lazy-import via `core/backends.py` so Azure-only users don't pay the `litert-lm-api` import cost.
- **Gemma prefill is the long pole.** Tool description text dominates — 47 tools × full descriptions ≈ 8K tokens. `_synthesize_callable` truncates per-field descriptions to `GEMMA_TOOL_DESC_MAX` chars (default 120) to keep prefill under ~5K tokens. On CPU a full 16-scenario record takes ~50 min; each prefill is 60–90 s.
- **GPU backend has a parsing bug.** `GEMMA_BACKEND=GPU` runs on Apple Silicon Metal at ~6× the throughput, but the current `litert-lm-api` WebGPU sampler path occasionally produces a truncated `<|tool_call>` that the engine parser rejects. Stick with CPU for recordings; flip GPU on for interactive REPL where a retry is cheap.
- **Gemma parser can crash on malformed output.** `litert_lm.cc:761` raises `INVALID_ARGUMENT: Failed to parse tool calls from response: …` when the model emits mixed prose + a malformed tool-call block (seen on the `todo` scenario). The adapter catches `RuntimeError` in `_create`, logs a warning, and returns empty text so the agent loop can still finish the turn. The replay cassette then contains an empty assistant message — harmless, but the scenario will show `pass=✗` because no tool was called.
- **`.env` is gitignored.** Don't commit real Azure keys or the HuggingFace token. `.env.example` has placeholders only. `models/` is also gitignored — the 2.58 GB `.litertlm` bundle lives there.
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
- `test_gemma_adapter.py` — fakes `litert_lm.Engine`; verifies OpenAI-shape output, tool-call interception, tool-role message folding
- `test_azure_integration.py` — real Azure round-trip (skipped without `AZURE_OPENAI_API_KEY`)

All tests run fast (<1s except the Azure integration test ~15s). No live Gemma test in CI — gate one on `GEMMA_MODEL_PATH` if you add one.
