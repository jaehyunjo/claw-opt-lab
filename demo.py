"""Scripted demo — walks the agent through realistic multi-category scenarios.

Three execution modes:

    replay  (default)  read cassettes/<scenario>.json, no Azure calls — instant
    record             call real Azure, save cassettes — run this once after
                       changing prompts, schemas, or scenarios
    live               call real Azure, skip caching — handy for ad-hoc probing

A summary table always prints at the end: expected vs. actual tool calls,
iterations, wall-clock time, and a pass flag (every expected tool was called).

Usage:
    ./run.sh demo                                  # replay every scenario
    ./run.sh demo --list                           # list scenario names
    ./run.sh demo --scenario email,todo            # a subset
    ./run.sh demo --mode record                    # re-record every scenario
    ./run.sh demo --mode record --scenario email   # re-record just one
    ./run.sh demo --mode live                      # call Azure; don't cache
    ./run.sh demo --prompt "..."                   # ad-hoc prompt (live only)
    ./run.sh demo --summary-only                   # skip per-turn trace
    ./run.sh demo --debug                          # + openai/httpx debug logs
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from core import shim  # noqa: E402
shim.install_shim()

from tools.registry import registry, discover_builtin_tools  # noqa: E402
from core.agent import run_conversation  # noqa: E402
from core.azure import AzureConfigError  # noqa: E402
from core.backends import ALL_BACKENDS, BACKEND_AZURE, get_backend  # noqa: E402
from core.cassette import (  # noqa: E402
    RecordingClient, ReplayClient,
    CassetteMissingError, CassetteExhaustedError,
)
from core.gemma import GemmaConfigError  # noqa: E402
from core.prompt import build_system_prompt  # noqa: E402
from core.skill_loader import load_skill_index  # noqa: E402

CASSETTE_ROOT = ROOT / "cassettes"


def cassette_dir(backend: str) -> Path:
    return CASSETTE_ROOT / backend


# ── ANSI helpers (off when stdout isn't a tty) ─────────────────────────────

_USE_COLOR = sys.stdout.isatty()
def _c(code, text): return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text
def dim(t):     return _c("2", t)
def bold(t):    return _c("1", t)
def cyan(t):    return _c("36", t)
def yellow(t):  return _c("33", t)
def green(t):   return _c("32", t)
def magenta(t): return _c("35", t)
def red(t):     return _c("31", t)


# ── Scenarios ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Scenario:
    name: str
    expect: str
    expected_tools: Tuple[str, ...]
    prompt: str


SCENARIOS: List[Scenario] = [
    Scenario("email",           "clarify → send_message",          ("send_message",),
             "Gmail로 jaehyun.jo@gmail.com에 메일 한 통 보내줘. "
             "제목은 '내일 오후 3시 미팅 확인', "
             "본문은 '재현님, 내일 3시 미팅 일정 맞는지 확인 부탁드립니다. 감사합니다.'"),
    Scenario("web-research",    "web_search → web_extract",        ("web_search",),
             "'LLM routing' 관련 최근 블로그 포스트 3개를 찾아서 "
             "각 제목과 한 줄 요약으로 정리해줘."),
    Scenario("file-explore",    "search_files → read_file",        ("search_files",),
             "현재 프로젝트 디렉토리에서 파일명에 'registry'가 들어간 Python 파일을 찾고, "
             "그 파일이 어떤 역할을 하는지 한국어로 2-3문장으로 요약해줘."),
    Scenario("code-exec",       "execute_code",                    ("execute_code",),
             "피보나치 수열 1번째부터 30번째까지 Python으로 계산해서 리스트로 보여줘. 반드시 코드 실행 툴을 사용해."),
    Scenario("memory",          "memory",                          ("memory",),
             "내 이름은 재현이고 백엔드 개발을 주로 한다는 점을 memory 툴을 이용해 기억해 둬."),
    Scenario("todo",            "todo",                            ("todo",),
             "오늘 할 일 3개를 추가해줘: "
             "1) PR #142 리뷰 답변, 2) API 문서 업데이트, 3) 내일 스탠드업 자료 준비."),
    Scenario("image-gen",       "image_generate",                  ("image_generate",),
             "노트북 앞에 앉아있는 고양이 일러스트를 수채화 스타일로 하나 생성해줘."),
    Scenario("multi-step",      "write_file → terminal",           ("write_file", "terminal"),
             "hello.py라는 파일에 'Hello World'를 출력하는 파이썬 한 줄짜리 스크립트를 만들고, "
             "'add hello.py'라는 메시지로 git commit까지 해줘."),
    Scenario("skill-arxiv",     "terminal (arxiv API)",            ("terminal",),
             "arxiv에서 'speculative decoding' 관련 최근 논문 3편의 제목과 arXiv ID를 찾아줘. "
             "terminal 툴로 `curl 'https://export.arxiv.org/api/query?search_query=all:speculative+decoding&max_results=3'` "
             "을 실행해서 arXiv API를 호출해. 파싱 결과에서 title과 id 필드만 뽑으면 돼."),
    Scenario("vision",          "vision_analyze",                  ("vision_analyze",),
             "`vision_analyze` 툴을 (browser_vision 말고 꼭 vision_analyze) 사용해서 "
             "/tmp/screenshot.png 로컬 이미지 파일을 분석해줘. 파일은 이미 존재하니 확인 묻지 말고 바로 호출."),
    Scenario("delegate",        "delegate_task",                   ("delegate_task",),
             "다음 세 가지 작업을 subagent에게 병렬로 위임해서 결과를 모아줘: "
             "A) 'hello world'라는 단어가 포함된 파일 찾기, "
             "B) package.json에서 프로젝트 이름 추출, "
             "C) README.md 첫 100자 요약."),
    Scenario("web-extract",     "web_extract",                     ("web_extract",),
             "https://www.python.org/about/ 페이지의 핵심 내용을 한국어로 3줄 요약해줘."),
    Scenario("cronjob",         "cronjob",                         ("cronjob",),
             "확인 묻지 말고 바로 `cronjob` 툴을 호출해서 아래 작업을 등록해. "
             "action=create, schedule='0 9 * * *', "
             "message='스탠드업 시간입니다', name='daily-standup-reminder'. "
             "terminal이나 write_file 같은 다른 툴은 절대 사용하지 말고 cronjob 툴 한 번만 불러."),
    Scenario("session-search",  "session_search",                  ("session_search",),
             "지난 대화들 중에서 내가 'LLM routing'이나 'on-device LLM'에 대해 뭐라고 "
             "언급했는지 session_search로 찾아서 2-3줄로 정리해줘."),
    Scenario("skills-list",     "skills_list",                     ("skills_list",),
             "지금 사용 가능한 skill들을 5개 정도만 이름과 간단 설명으로 알려줘."),
    Scenario("ambiguous",       "clarify",                         ("clarify",),
             "정리해줘."),
]

SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}


# ── Running state ──────────────────────────────────────────────────────────

@dataclass
class RunResult:
    scenario: Scenario
    tool_calls: List[str] = field(default_factory=list)
    iterations: int = 0
    elapsed: float = 0.0
    final: str = ""
    error: str | None = None

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        required = set(self.scenario.expected_tools)
        return required.issubset(set(self.tool_calls))


# ── Config banner ──────────────────────────────────────────────────────────

def _redact(s: str | None, keep: int = 4) -> str:
    if not s:
        return "(unset)"
    if len(s) <= keep * 2:
        return "*" * len(s)
    return f"{s[:keep]}…{s[-keep:]}"


def print_config(mode: str, backend: str) -> None:
    print(bold("[config]"))
    print(f"  {'mode':28s} {mode}")
    print(f"  {'backend':28s} {backend}")
    if mode in ("record", "live"):
        if backend == "azure":
            for key in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION"):
                print(f"  {key:28s} {os.environ.get(key, '(unset)')}")
            print(f"  {'AZURE_OPENAI_API_KEY':28s} {_redact(os.environ.get('AZURE_OPENAI_API_KEY'))}")
        elif backend == "gemma":
            print(f"  {'GEMMA_HF_REPO':28s} {os.environ.get('GEMMA_HF_REPO', 'litert-community/gemma-4-E2B-it-litert-lm')}")
            print(f"  {'GEMMA_MODEL_FILE':28s} {os.environ.get('GEMMA_MODEL_FILE', 'gemma-4-E2B-it.litertlm')}")
            override = os.environ.get("GEMMA_MODEL_PATH")
            if override:
                print(f"  {'GEMMA_MODEL_PATH':28s} {override}")
            token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
            print(f"  {'HUGGINGFACE_TOKEN':28s} {_redact(token)}")
    else:
        print(f"  {'cassette dir':28s} {cassette_dir(backend).relative_to(ROOT)}")
    print()


# ── Per-scenario runner ────────────────────────────────────────────────────

def _run_one(scenario: Scenario, system_prompt: str, tool_names: List[str],
             idx: int, total: int, max_iters: int, mode: str, backend: str,
             summary_only: bool) -> RunResult:
    result = RunResult(scenario=scenario)

    cassette_path = cassette_dir(backend) / f"{scenario.name}.json"
    try:
        if mode == "replay":
            client = ReplayClient(cassette_path)
            deployment = "replay"
            recording = None
        elif mode == "record":
            real, deployment = get_backend(backend)
            recording = RecordingClient(real, cassette_path, model_hint=deployment)
            client = recording
        elif mode == "live":
            client, deployment = get_backend(backend)
            recording = None
        else:
            raise ValueError(f"unknown mode: {mode}")
    except CassetteMissingError as e:
        result.error = str(e)
        return result
    except (AzureConfigError, GemmaConfigError) as e:
        result.error = f"{type(e).__name__}: {e}"
        return result

    buf = io.StringIO() if not summary_only else None

    def emit(line: str = "") -> None:
        if buf is not None:
            print(line, file=buf)

    emit(cyan(bold(f"── [{idx}/{total}] {scenario.name} ──")))
    emit(f"{dim('expect: ')} {scenario.expect}")
    emit(f"{bold('[user]')}   {scenario.prompt}")

    def on_tool_call(name, args):
        result.tool_calls.append(name)
        s = json.dumps(args, ensure_ascii=False)
        if len(s) > 240:
            s = s[:237] + "…"
        emit(f"  {yellow('→ call  ')}{name}({s})")

    def on_tool_result(_name, r):
        try:
            parsed = json.loads(r)
            s = json.dumps(parsed, ensure_ascii=False)
        except Exception:
            s = str(r)
        if len(s) > 200:
            s = s[:197] + "…"
        emit(f"  {green('← result')} {s}")

    start = time.time()
    try:
        out = run_conversation(
            user_message=scenario.prompt,
            system_prompt=system_prompt,
            enabled_tools=tool_names,
            max_iterations=max_iters,
            client=client,
            deployment=deployment,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
        )
    except CassetteExhaustedError as e:
        result.error = str(e)
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"

    result.elapsed = time.time() - start

    if result.error:
        emit(f"  {magenta('ERROR')} {result.error}")
    else:
        result.iterations = out["iterations"]
        result.final = out["final"] or ""
        final_display = result.final or dim("(empty)")
        emit(f"{bold('[assistant]')}")
        for line in (final_display.splitlines() or [""]):
            emit(f"  {line}")
        emit(dim(
            f"[stats]  iterations={result.iterations}  time={result.elapsed:.2f}s  "
            f"tools={len(result.tool_calls)}  pass={'✓' if result.passed else '✗'}"
        ))
        emit()

    # Save the cassette only on successful record (scenario passed or at least didn't error).
    if recording is not None and result.error is None:
        recording.save()

    if buf is not None:
        sys.stdout.write(buf.getvalue())
        sys.stdout.flush()

    return result


# ── Summary table ──────────────────────────────────────────────────────────

def _truncate(s: str, width: int) -> str:
    return s if len(s) <= width else s[: width - 1] + "…"


def print_summary(results: List[RunResult], mode: str, backend: str) -> None:
    if not results:
        return

    name_w = max(4, min(14, max(len(r.scenario.name) for r in results)))
    expect_w = max(8, min(28, max(len(r.scenario.expect) for r in results)))
    actual_w = 30

    header = (
        f"{'#':>2}  "
        f"{'scenario':<{name_w}}  "
        f"{'expected':<{expect_w}}  "
        f"{'actual calls':<{actual_w}}  "
        f"{'iter':>4}  "
        f"{'time':>6}  "
        f"{'✓'}"
    )
    sep = "─" * len(header)

    print()
    print(bold(f"── Summary ({mode} / {backend}) ──"))
    print(sep)
    print(bold(header))
    print(sep)
    passed = 0
    total_tool_calls = 0
    total_time = 0.0
    for i, r in enumerate(results, start=1):
        actual_calls = ", ".join(r.tool_calls) if r.tool_calls else "(none)"
        pass_mark = green("✓") if r.passed else red("✗")
        if r.error:
            pass_mark = magenta("E")
        print(
            f"{i:>2}  "
            f"{_truncate(r.scenario.name, name_w):<{name_w}}  "
            f"{_truncate(r.scenario.expect, expect_w):<{expect_w}}  "
            f"{_truncate(actual_calls, actual_w):<{actual_w}}  "
            f"{r.iterations:>4}  "
            f"{r.elapsed:>5.1f}s  "
            f"{pass_mark}"
        )
        if r.passed:
            passed += 1
        total_tool_calls += len(r.tool_calls)
        total_time += r.elapsed
    print(sep)
    print(dim(
        f"pass: {passed}/{len(results)}   "
        f"tool calls: {total_tool_calls}   "
        f"total time: {total_time:.2f}s"
    ))

    freq: dict[str, int] = {}
    for r in results:
        for t in r.tool_calls:
            freq[t] = freq.get(t, 0) + 1
    if freq:
        print()
        print(bold("── Tool call frequency ──"))
        width = max(len(k) for k in freq)
        for name, count in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
            bar = "█" * count
            print(f"  {name:<{width}}  {count:>3}  {dim(bar)}")

    failures = [r for r in results if not r.passed]
    if failures:
        print()
        print(bold("── Failures / non-matches ──"))
        for r in failures:
            reason = r.error or (
                f"expected {list(r.scenario.expected_tools)}; "
                f"actual {r.tool_calls or '(none)'}"
            )
            print(f"  {red('✗')} {r.scenario.name}: {reason}")


# ── CLI ────────────────────────────────────────────────────────────────────

def _resolve(names_csv: str | None) -> List[Scenario]:
    if not names_csv:
        return list(SCENARIOS)
    out: List[Scenario] = []
    for n in names_csv.split(","):
        n = n.strip()
        if not n:
            continue
        if n not in SCENARIOS_BY_NAME:
            raise SystemExit(f"unknown scenario: {n} (known: {', '.join(SCENARIOS_BY_NAME)})")
        out.append(SCENARIOS_BY_NAME[n])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Hermes skeleton scenario demo")
    parser.add_argument("--list", action="store_true", help="Print scenario names and exit")
    parser.add_argument("--scenario", help="Comma-separated scenario name(s) to run")
    parser.add_argument("--prompt", action="append",
                        help="Custom prompt (live mode only); may be passed multiple times")
    parser.add_argument("--mode", choices=("replay", "record", "live"), default="replay",
                        help="replay cassettes (default), record fresh cassettes, or call the live backend without caching")
    parser.add_argument("--backend", choices=ALL_BACKENDS, default=BACKEND_AZURE,
                        help="LLM backend: azure (default) or gemma (HuggingFace Inference)")
    parser.add_argument("--max-iterations", type=int, default=6,
                        help="Max agent-loop iterations per turn (default: 6)")
    parser.add_argument("--summary-only", action="store_true",
                        help="Suppress per-turn traces; print only the final summary table")
    parser.add_argument("--debug", action="store_true",
                        help="Enable openai/httpx debug logs")
    args = parser.parse_args()

    if args.list:
        width = max(len(s.name) for s in SCENARIOS)
        cdir = cassette_dir(args.backend)
        for s in SCENARIOS:
            cassette = cdir / f"{s.name}.json"
            marker = green("●") if cassette.exists() else dim("○")
            print(f"  {marker}  {s.name:<{width}}  expect: {s.expect}")
        print()
        print(dim(
            f"Total: {len(SCENARIOS)} scenarios   backend={args.backend}   "
            f"(● = cassette present at {cdir.relative_to(ROOT)}, ○ = missing)"
        ))
        return 0

    if args.prompt and args.mode == "replay":
        raise SystemExit("--prompt requires --mode live or --mode record")

    if args.debug:
        os.environ["OPENAI_LOG"] = "debug"
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("httpx").setLevel(logging.DEBUG)
        logging.getLogger("openai").setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    discover_builtin_tools()
    tool_names = registry.get_all_tool_names()
    # Gemma-4 E2B is a small on-device model (4K trained context; we extend
    # the engine to 16K). The default prompt ships all 79 skills (~5K tokens)
    # which, once combined with 47 tool schemas rendered by the chat template,
    # overflows the window. For Gemma we keep a short skill listing (the top
    # GEMMA_MAX_SKILLS names, default 10) so the model still has some
    # skill-awareness without blowing the context. Azure keeps the full list.
    if args.backend == "gemma":
        max_skills = int(os.environ.get("GEMMA_MAX_SKILLS", "10"))
    else:
        max_skills = None
    system_prompt = build_system_prompt(max_skills=max_skills)
    skill_count = len(load_skill_index())

    print(bold("Hermes skeleton scenario demo"))
    print_config(args.mode, args.backend)
    effective_skills = skill_count if max_skills is None else min(max_skills, skill_count)
    print(dim(
        f"Tools: {len(tool_names)}   Skills: {effective_skills}/{skill_count}   "
        f"System prompt: {len(system_prompt)} chars   "
        f"max_iterations={args.max_iterations}"
    ))
    print()

    if args.prompt:
        scenarios = [
            Scenario(name=f"custom-{i+1}", expect="(free-form)",
                     expected_tools=(), prompt=p)
            for i, p in enumerate(args.prompt)
        ]
    else:
        scenarios = _resolve(args.scenario)

    results: List[RunResult] = []
    wall_start = time.time()
    for idx, sc in enumerate(scenarios, start=1):
        r = _run_one(
            sc, system_prompt, tool_names, idx, len(scenarios),
            args.max_iterations, args.mode, args.backend, args.summary_only,
        )
        results.append(r)
    wall = time.time() - wall_start

    print_summary(results, args.mode, args.backend)
    print()
    print(dim(f"wall-clock: {wall:.2f}s"))

    failed = any((not r.passed) and r.scenario.expected_tools for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
