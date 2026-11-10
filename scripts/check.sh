#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
PATH="$HOME/.elan/bin:$PATH"
export PATH
lake build
python -m unittest discover -s tests -v
python scripts/audit_lean.py
