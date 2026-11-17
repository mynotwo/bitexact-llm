#!/usr/bin/env bash
# The CPU checks, as CI runs them (.github/workflows/cpu.yml): lint, the test suite (CUDA and model cases
# skip without their prerequisites), the weight-free arithmetic self-check, the wheel build, and the same
# self-check from the built wheel outside the checkout.
#
#   python -m pip install -e '.[test]' ruff      # once; CPU-only torch is enough
#   bash scripts/check.sh [OUT]
#
#   OUT  receives test-results.xml, arithmetic-<system>-py<X.Y>.json, arithmetic-wheel-py<X.Y>.json and
#        dist/*.whl (default: a new temporary directory). Reports are never overwritten, so OUT must not
#        already hold them.
#   PY   python with the package and ruff installed (default: python)
set -euo pipefail
PY=$("${PY:-python}" -c 'import sys; print(sys.executable)')
out=${1:-$(mktemp -d)}
mkdir -p "$out"
out=$(cd "$out" && pwd)
cd "$(dirname "$0")/.."
system=$("$PY" -c 'import platform; print(platform.system().lower())')
py=$("$PY" -c 'import sys; print("py%d.%d" % sys.version_info[:2])')

"$PY" -m ruff check
"$PY" -m pytest -q --junitxml="$out/test-results.xml"
"$PY" -m bitexact.repro self-check --out "$out/arithmetic-$system-$py.json"
"$PY" -m pip wheel --no-deps . --wheel-dir "$out/dist"

# Unpack the wheel away from the checkout and run it from there. PYTHONPATH precedes the installed
# (editable) package, which is left as it is; the first command fails unless the unpacked wheel is what loads.
site=$(mktemp -d)
"$PY" -m pip install --no-deps --target "$site" "$out"/dist/*.whl
cd "$site"
PYTHONPATH=$site "$PY" -c 'import os, sys, bitexact; sys.exit(not os.path.samefile(os.path.dirname(bitexact.__path__[0]), sys.argv[1]))' "$site"
PYTHONPATH=$site "$PY" -m bitexact.repro self-check --out "$out/arithmetic-wheel-$py.json"
echo "CPU checks passed; results in $out"
