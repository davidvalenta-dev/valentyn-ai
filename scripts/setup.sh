#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install -e .
if ! command -v elan >/dev/null 2>&1 && [ ! -x "$HOME/.elan/bin/elan" ]; then
  installer=$(mktemp)
  trap 'rm -f "$installer"' EXIT
  curl -fsSL https://elan.lean-lang.org/elan-init.sh -o "$installer"
  sh "$installer" -y --no-modify-path --default-toolchain none
fi
PATH="$HOME/.elan/bin:$PATH"
export PATH
lake build
.venv/bin/python -m valentyn doctor
printf '\nReady. Activate with: . .venv/bin/activate\n'
