#!/usr/bin/env bash
# agg.sh DIR : one line per prof_vllm.py run, base vs tree, from DIR/*/int.json
set -u
D=${1:?dir}
/usr/bin/env python3 - "$D" <<'EOF'
import glob, json, os, sys
rows = []
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*", "int.json"))):
    r = json.load(open(p))
    d = r.get("decode", {})
    t = r.get("timing", {})
    rows.append(dict(run=os.path.basename(os.path.dirname(p)), batch=r["batch"], temp=r["temp"],
                     wall=t.get("decode_step_ms_median"), busy=d.get("kernel_ms_per_step"),
                     tok=r.get("tok_per_s")))
print("%-34s %5s %4s %8s %8s %8s %8s" % ("run", "B", "T", "wall", "busy", "wall-busy", "tok/s"))
for x in rows:
    wb = round(x["wall"] - x["busy"], 3) if x["wall"] and x["busy"] else None
    print("%-34s %5d %4s %8s %8s %8s %8s" % (x["run"], x["batch"], x["temp"], x["wall"], x["busy"], wb, x["tok"]))
EOF