#!/usr/bin/env bash
# Convenience runner for the Hermes skeleton.
#
# Usage:
#   ./run.sh setup           # create venv, install deps (one-time)
#   ./run.sh demo [args...]  # run demo.py (default scenarios unless args given)
#   ./run.sh test [args...]  # run pytest
#   ./run.sh repl            # interactive main.py
#
# Examples:
#   ./run.sh demo --debug
#   ./run.sh demo --prompt "Hello in Korean"
#   ./run.sh test -v
#   ./run.sh test tests/test_shim.py

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VENV="$HERE/venv"
PYTHON_BIN="${PYTHON:-python3.11}"

die() { echo "error: $*" >&2; exit 1; }

ensure_venv() {
  if [ ! -x "$VENV/bin/python" ]; then
    die "venv not found at $VENV — run: ./run.sh setup"
  fi
}

activate() {
  ensure_venv
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
}

cmd_setup() {
  if [ -x "$VENV/bin/python" ]; then
    echo "venv already exists at $VENV"
  else
    echo "creating venv with $PYTHON_BIN"
    "$PYTHON_BIN" -m venv "$VENV"
  fi
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -e ".[dev]"
  echo "setup complete — $VENV"
  if [ ! -f "$HERE/.env" ]; then
    if [ -f "$HERE/.env.example" ]; then
      echo "note: .env not found; copy .env.example and fill in Azure creds:"
      echo "  cp .env.example .env"
    fi
  fi
}

cmd_demo() {
  activate
  python demo.py "$@"
}

cmd_test() {
  activate
  if [ $# -eq 0 ]; then
    pytest tests/ -v
  else
    pytest "$@"
  fi
}

cmd_repl() {
  activate
  python main.py "$@"
}

main() {
  local sub="${1:-}"
  if [ -z "$sub" ]; then
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
  fi
  shift || true
  case "$sub" in
    setup) cmd_setup "$@" ;;
    demo)  cmd_demo "$@" ;;
    test)  cmd_test "$@" ;;
    repl)  cmd_repl "$@" ;;
    -h|--help|help)
      sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
      ;;
    *)
      die "unknown subcommand: $sub (try: setup | demo | test | repl)"
      ;;
  esac
}

main "$@"
