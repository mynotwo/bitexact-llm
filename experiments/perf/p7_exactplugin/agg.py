"""Aggregate bench_all.sh outputs: python agg.py BENCH_DIR  -> tok/s median [min-max] per (config, batch), n reps."""
import glob
import json
import os
import re
import statistics as st
import sys

O = sys.argv[1]
rows = {}
for f in glob.glob(f"{O}/*_b*_r*.log"):
    m = re.match(r"(\w+?)_b(\d+)_r(\d+)\.log", os.path.basename(f))
    for ln in open(f, errors="ignore"):
        if ln.startswith("RESULT"):
            rows.setdefault((m[1], int(m[2])), []).append(json.loads(ln[7:])["tok_per_s"])
bad = [ln.strip() for ln in open(f"{O}/progress.log") if "CONTAMINATED" in ln]
cfgs = ["exact", "int8", "bf16", "bi"]
print("| config | " + " | ".join(f"batch {b}" for b in sorted({k[1] for k in rows})) + " |")
print("|---|" + "---|" * len({k[1] for k in rows}))
for c in cfgs:
    cells = []
    for b in sorted({k[1] for k in rows}):
        v = rows.get((c, b))
        cells.append(f"{st.median(v):,.0f} [{min(v):,.0f}-{max(v):,.0f}] (n={len(v)})" if v else "-")
    print(f"| {c} | " + " | ".join(cells) + " |")
print("contaminated:", bad or "none")
