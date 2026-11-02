#!/bin/bash
# usage: make_entry.sh OUT.sh [TAG]   -> entry script with this checkout's committed HEAD embedded (git archive)
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd); R=$(git -C "$D" rev-parse --show-toplevel); C=$(git -C "$R" rev-parse --short HEAD)
B=$(git -C "$R" archive --format=tar.gz HEAD src examples experiments/xhw_8b | base64 | tr -d '\n' | fold -w 100)
python3 - "$D/entry_8b.sh.in" "$1" "$C" "${2:-l20}" "$B" <<'PY'
import sys
src, out, commit, tag, b64 = sys.argv[1:]
s = open(src).read().replace("@CODE_B64@", b64).replace("@COMMIT@", commit).replace("@TAG@", tag)
open(out, "w").write(s)
PY
echo "wrote $1 (commit $C, $(wc -c < "$1") bytes)"
