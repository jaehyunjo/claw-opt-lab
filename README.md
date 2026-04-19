# claw-opt-lab

A stripped-down agent skeleton for **LLM optimization experiments** — routing (on-device vs cloud), small-context prompt assembly, RAG, cost measurement. Forked from [hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT) and reduced to the bare bones needed to observe agent-loop + tool-call behavior end-to-end.

- **47 tool schemas** preserved from upstream so the LLM sees a realistic tool surface
- **All tool handlers mocked** — no real filesystem / network / process side effects
- **79 skills** indexed into the system prompt for skill-aware behavior
- **Cassette record/replay** — hit Azure OpenAI once, replay instantly thereafter
- **16 Korean scenario demos** covering email, web research, code exec, file ops, cron, vision, and more

## Quick start

```bash
git clone <this repo>
cd claw-opt-lab
./run.sh setup                         # python -m venv + pip install -e ".[dev]"
cp .env.example .env                   # fill in Azure creds for live/record modes
./run.sh demo                          # replay 16 pre-recorded scenarios (~0s)
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
│   ├── agent.py             # Agent loop: LLM → tool dispatch → repeat
│   ├── cassette.py          # Record/replay Azure responses
│   ├── prompt.py            # System prompt + skills index
│   └── skill_loader.py      # Scan skills/**/SKILL.md
├── tools/                   # 21 self-registering tool modules (handlers mocked)
│   ├── registry.py          # Tool registry (unchanged from upstream)
│   └── ...
├── skills/                  # 79 bundled skills (SKILL.md + helpers)
├── cassettes/               # Recorded Azure responses — one JSON per scenario
├── hermes_constants.py      # Path helpers used by some tool files
└── tests/                   # pytest suite
```

## Three execution modes

| Mode     | Azure?  | Cassette | Use when                          |
|----------|---------|----------|-----------------------------------|
| `replay` | no      | read     | default — fast, deterministic, no API cost |
| `record` | yes     | write    | after changing scenarios, prompts, schemas, or mocks |
| `live`   | yes     | —        | ad-hoc probing of the live model  |

```bash
./run.sh demo                               # replay (default)
./run.sh demo --mode record                 # re-record every scenario
./run.sh demo --mode record --scenario vision,cronjob
./run.sh demo --mode live --prompt "자유 질문"
```

Cassettes live under `cassettes/<scenario>.json` — check them in so anyone can run the demo without Azure credentials. Re-record when you change prompts/schemas; the cassette format is line-diffable JSON.

## Environment

`.env` keys (only needed for `record` / `live`):

| Var                          | Example                                       |
|------------------------------|-----------------------------------------------|
| `AZURE_OPENAI_API_KEY`       | your Azure key (don't commit)                 |
| `AZURE_OPENAI_ENDPOINT`      | `https://YOUR-RESOURCE.openai.azure.com`      |
| `AZURE_OPENAI_DEPLOYMENT`    | your deployment name (e.g. `gpt-5-mini`)      |
| `AZURE_OPENAI_API_VERSION`   | `2024-10-21` (default)                        |

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

- **Routing experiment**: add `core/router.py` with `(messages, tools, context_budget) → (client, deployment)`; route between `core.azure.get_client()` and an on-device OpenAI-compatible endpoint (ollama / vLLM / llama.cpp).
- **Measurement**: wrap `cli.chat.completions.create(...)` in `core/agent.py` to log token counts, latency, and cost per call.
- **Better mocks**: add entries to `_TOOL_OVERRIDES` in `core/shim.py` — the LLM works with whatever shape you return.
- **RAG**: hook upstream of `build_system_prompt()` in `core.prompt` — retrieve chunks, inject before the skills section.
- **New scenarios**: append a `Scenario(...)` to the list in `demo.py`, then re-record with `--mode record --scenario <name>`.

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
