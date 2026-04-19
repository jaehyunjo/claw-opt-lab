# claw-opt-lab

A stripped-down agent skeleton for **LLM optimization experiments** — routing (on-device vs cloud), small-context prompt assembly, RAG, cost measurement. Forked from [hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT) and reduced to the bare bones needed to observe agent-loop + tool-call behavior end-to-end.

- **47 tool schemas** preserved from upstream so the LLM sees a realistic tool surface
- **All tool handlers mocked** — no real filesystem / network / process side effects
- **79 skills** indexed into the system prompt for skill-aware behavior
- **Cassette record/replay** — hit the backend once, replay instantly thereafter
- **Switchable backends** — cloud Azure OpenAI (`gpt-5-mini`) or on-desktop Gemma (`gemma-4-E2B-it.litertlm` via LiteRT-LM)
- **16 Korean scenario demos** covering email, web research, code exec, file ops, cron, vision, and more

## Quick start

```bash
git clone <this repo>
cd claw-opt-lab
./run.sh setup                         # python -m venv + pip install -e ".[dev]"
cp .env.example .env                   # fill in Azure and/or HuggingFace creds
./run.sh demo                          # replay 16 pre-recorded Azure scenarios (~0s)
./run.sh demo --backend gemma          # replay Gemma cassettes (offline)
./run.sh test                          # pytest
```

```
Hermes skeleton scenario demo
[config]
  mode                         replay
  cassette dir                 cassettes

Tools: 47   Skills: 79   System prompt: 20495 chars   max_iterations=6

── [1/16] email ──
expect:   clarify → send_message
[user]    Gmail로 jaehyun.jo@gmail.com에 메일 한 통 보내줘. ...
  → call  clarify({"question": "지금 바로 메일을 전송해도 될까요?", ...})
  ← result {"ok": true, "tool": "clarify", "user_response": "yes", ...}
  → call  send_message({"action": "send", "target": "gmail:jaehyun.jo@gmail.com", ...})
  ← result {"ok": true, "tool": "send_message", "status": "sent", ...}
[assistant]
  메일을 보냈습니다. ...
[stats]  iterations=3  tools=2  pass=✓
```

## Layout

```
claw-opt-lab/
├── main.py                  # REPL entry point
├── demo.py                  # 16-scenario demo with record/replay
├── run.sh                   # bash wrapper (setup / demo / test / repl)
├── core/
│   ├── shim.py              # MagicMock deleted modules; monkey-patch registry
│   ├── azure.py             # AzureOpenAI client (env-driven)
│   ├── gemma.py             # LiteRT-LM Gemma-4 E2B client
│   ├── backends.py          # Backend factory: azure | gemma → (client, deployment)
│   ├── agent.py             # Agent loop: LLM → tool dispatch → repeat
│   ├── cassette.py          # Record/replay backend responses (backend-agnostic)
│   ├── prompt.py            # System prompt + skills index
│   └── skill_loader.py      # Scan skills/**/SKILL.md
├── tools/                   # 21 self-registering tool modules (handlers mocked)
│   ├── registry.py          # Tool registry (unchanged from upstream)
│   └── ...
├── skills/                  # 79 bundled skills (SKILL.md + helpers)
├── cassettes/
│   ├── azure/               # Recorded Azure responses — one JSON per scenario
│   └── gemma/               # Recorded Gemma LiteRT-LM responses
├── models/                  # Cached .litertlm bundles (gitignored, ~2.6 GB)
├── hermes_constants.py      # Path helpers used by some tool files
└── tests/                   # pytest suite
```

## Two backends

| Backend  | Where it runs           | What drives inference                                  |
|----------|-------------------------|--------------------------------------------------------|
| `azure`  | cloud                   | Azure OpenAI `gpt-5-mini` via the `openai` SDK         |
| `gemma`  | local CPU (macOS/Linux) | `litert-lm-api` loads `gemma-4-E2B-it.litertlm` directly from the `litert-community` HF repo |

Both expose the same OpenAI-shape `chat.completions.create` to the agent loop, so the record/replay cassette layer, tests, and scenario table don't care which one you pick.

```bash
./run.sh demo                                          # Azure replay (default)
./run.sh demo --backend gemma                          # Gemma replay
./run.sh demo --backend gemma --mode live --scenario email
./run.sh demo --backend gemma --mode record            # record all 16 Gemma cassettes
```

The Gemma tool-call surface is emulated: tool JSON schemas are injected into the system prompt as text, and the adapter parses a JSON block out of the reply back into OpenAI-shape `tool_calls`. That keeps the agent loop identical across backends.

## Three execution modes

| Mode     | Live LLM? | Cassette | Use when                                |
|----------|-----------|----------|-----------------------------------------|
| `replay` | no        | read     | default — fast, deterministic, no cost  |
| `record` | yes       | write    | after changing scenarios, prompts, schemas, or mocks |
| `live`   | yes       | —        | ad-hoc probing of the live backend      |

Cassettes live under `cassettes/<backend>/<scenario>.json` — check them in so anyone can run the demo without credentials. Re-record when you change prompts/schemas; the cassette format is line-diffable JSON.

## Environment

`.env` keys (each block only needed for `record` / `live` on that backend):

**Azure**

| Var                          | Example                                       |
|------------------------------|-----------------------------------------------|
| `AZURE_OPENAI_API_KEY`       | your Azure key (don't commit)                 |
| `AZURE_OPENAI_ENDPOINT`      | `https://YOUR-RESOURCE.openai.azure.com`      |
| `AZURE_OPENAI_DEPLOYMENT`    | your deployment name (e.g. `gpt-5-mini`)      |
| `AZURE_OPENAI_API_VERSION`   | `2024-10-21` (default)                        |

**Gemma (LiteRT-LM)**

| Var                          | Example                                                  |
|------------------------------|----------------------------------------------------------|
| `HUGGINGFACE_TOKEN`          | `hf_…` — accept the Gemma license on the HF UI first     |
| `GEMMA_HF_REPO`              | `litert-community/gemma-4-E2B-it-litert-lm` (default)    |
| `GEMMA_MODEL_FILE`           | `gemma-4-E2B-it.litertlm` (default, 2.58 GB)             |
| `GEMMA_MODEL_PATH`           | optional — absolute path to a local `.litertlm` override |

First `--backend gemma` run downloads the bundle into `./models/` (gitignored) and takes a few minutes; subsequent runs reuse the cached file.

## Scenarios (all 16)

```
./run.sh demo --list

  ●  email           expect: clarify → send_message
  ●  web-research    expect: web_search → web_extract
  ●  file-explore    expect: search_files → read_file
  ●  code-exec       expect: execute_code
  ●  memory          expect: memory
  ●  todo            expect: todo
  ●  image-gen       expect: image_generate
  ●  multi-step      expect: write_file → terminal
  ●  skill-arxiv     expect: terminal (arxiv API)
  ●  vision          expect: vision_analyze
  ●  delegate        expect: delegate_task
  ●  web-extract     expect: web_extract
  ●  cronjob         expect: cronjob
  ●  session-search  expect: session_search
  ●  skills-list     expect: skills_list
  ●  ambiguous       expect: clarify
```

(● = cassette present, ○ = missing. 16/16 cassettes ship with the repo.)

## How the mock stack works

1. `core.shim.install_shim()` registers a `sys.meta_path` finder that returns `MagicMock` for every deleted internal package (`run_agent`, `hermes_cli`, `gateway`, `agent.*`, …) and every heavy external dep (`exa_py`, `fal_client`, `firecrawl`, …). Tool files import freely.
2. The same call monkey-patches `tools.registry.registry.register(...)` — every handler becomes `_make_mock_handler(name, schema)`, `is_async` is forced to `False`, `check_fn` is forced to `lambda: True`.
3. Mock handlers validate args against the tool's JSON schema, then return either a generic `{"ok": true, "tool": "X", "args": args, "mock": true}` envelope or a richer override payload (see `_TOOL_OVERRIDES` in `core/shim.py` — cronjob, vision, web_search, etc. return realistic-looking shapes so the model accepts the result and moves on).
4. `tools.registry.discover_builtin_tools()` AST-scans `tools/*.py` and imports every self-registering module. Result: 47 mock tools available to the model.

**No tool has a real side effect.** `terminal("rm -rf /")`, `write_file(...)`, `send_message(...)`, `cronjob(action="create", ...)`, `curl ...` — all return mock success in 0ms without touching the host.

## Extending

- **Routing experiment**: the `core.backends` factory already picks between Azure and Gemma; drop in a `core/router.py` with `(messages, tools, context_budget) → backend_name` that calls `get_backend(...)` at each turn. Because both backends are swapped in via the same `client` param to `run_conversation`, the agent loop doesn't change.
- **Measurement**: wrap `cli.chat.completions.create(...)` in `core/agent.py` to log token counts, latency, and cost per call.
- **Better mocks**: add entries to `_TOOL_OVERRIDES` in `core/shim.py` — the LLM works with whatever shape you return.
- **RAG**: hook upstream of `build_system_prompt()` in `core.prompt` — retrieve chunks, inject before the skills section.
- **New scenarios**: append a `Scenario(...)` to the list in `demo.py`, then re-record with `--mode record --scenario <name>` (per backend).

### Known trade-offs for the Gemma backend

- **Context budget**: Gemma-4 E2B was trained at 4K tokens; the adapter extends `Engine(max_num_tokens=16384)`, but quality still degrades past the trained length. The full 79-skill system prompt (~5K tokens) plus 47 tool schemas rendered by the chat template would overflow that budget, so for Gemma the skill listing is trimmed to 10 entries via `build_system_prompt(max_skills=10)` and tool descriptions are truncated to `GEMMA_TOOL_DESC_MAX=120` chars per field. Azure keeps the full 79.
- **Tool-call surface**: Gemma's trained `<|tool_call>` format is exercised via `litert_lm.Engine.create_conversation(tools=[...])`. The 47 agent-side JSON schemas are fed through a lightweight callable synthesizer (`core/gemma.py:_synthesize_callable`) so the engine's introspection still yields the correct OpenAI schema. A `ToolEventHandler` captures the model's tool choice and the synthesized callable raises `_ToolCallIntercepted`, so the outer agent loop handles dispatch through the mock registry — same semantics as Azure.
- **Model load time**: first `--backend gemma` run downloads ~2.6 GB into `./models/` and then takes ~60 s to load the bundle into RAM (~3 GB resident). Subsequent scenarios reuse the shared `Engine` singleton, so per-scenario inference is dominated by prefill/decode rather than load. On M-series CPU a full 16-scenario record takes ~50 min.
- **Metal/GPU backend (`GEMMA_BACKEND=GPU`)** works end-to-end on Apple Silicon and cuts per-scenario time by ~6×, but LiteRT-LM's current GPU sampler path occasionally emits a truncated `<|tool_call>` that the engine parser rejects (`Failed to parse tool calls from response`). Left as CPU-default for now; revisit when `litert-lm-api` ships a WebGPU sampler.
- **Known scenario quirks (Gemma cassettes as-recorded)**:
  - `file-explore` — model picks `execute_code` instead of `search_files`.
  - `multi-step` — model chains `terminal` twice instead of `write_file → terminal`.
  - `skill-arxiv` — model picks `execute_code` instead of `terminal`.
  - `todo` — the model sometimes emits mixed prose + a malformed `<|tool_call>` block, which the LiteRT-LM parser rejects mid-generation. The adapter catches that `RuntimeError`, returns empty text, and the agent loop continues; the recorded cassette captures a later successful `todo` call. Rerun `--mode record --scenario todo` if you want a cleaner take.
  The Azure cassettes pass all 16; Gemma cassettes replay at **13/16** for the right reasons. The three fails are about Gemma's tool-choice behavior, not the agent loop — they're prompt-engineering work, recorded as-is.

## Tests

```bash
./run.sh test                          # 16 tests, ~15s (integration tests skip w/o Azure key)
./run.sh test tests/test_shim.py -v
```

| Suite | Covers |
|-------|--------|
| `test_shim.py`            | MagicMock import resolution; mock handler validation + success envelope |
| `test_skill_loader.py`    | SKILL.md frontmatter parsing; real skills/ load |
| `test_agent_loop.py`      | Agent loop with fake Azure client (plain text / tool call / max-iterations) |
| `test_azure_integration.py` | Real Azure round-trip — skipped without `AZURE_OPENAI_API_KEY` |

## Attribution

Forked from [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT). Upstream is a full-featured agent with messaging gateway, IDE integrations, skills hub, cron scheduler, RL training, and more. This fork strips ~99% of that down to a minimal skeleton for optimization experiments.

Tool schemas, skill content, and the `tools/registry.py` layer are preserved verbatim from upstream.

MIT — see [LICENSE](LICENSE).
