"""Print the gate and speed results of every tag in ~/som/out (or DIR): python summary.py [DIR]."""
import glob
import json
import os
import re
import statistics
import sys

d = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/som/out")
tags = sorted({re.split(r"[._]", os.path.basename(p))[0] for p in glob.glob(f"{d}/*") if not os.path.isdir(p)})
for t in tags:
    out = [t]
    p = f"{d}/{t}.scheck.jsonl"
    if os.path.exists(p):
        rs = [json.loads(x) for x in open(p)]
        out.append(f"scorer gate {sum(r['bit_exact'] for r in rs)}/{len(rs)}")
    p = f"{d}/{t}.stime.jsonl"
    if os.path.exists(p):
        rs = [json.loads(x) for x in open(p)]
        out.append("scorer tok/s " + " ".join(f"{r['tag'].split('-', 1)[1]}={r['tok_per_s']:.0f}" for r in rs))
    sp = []
    for b in (1, 64, 512):
        p = f"{d}/{t}_speed_B{b}.json"
        if os.path.exists(p):
            sp.append(f"B{b}={json.load(open(p))['ms_per_step_median']}")
    if sp:
        out.append("decode ms/step " + " ".join(sp))
    p = f"{d}/{t}_pytest.log"
    if os.path.exists(p):
        last = [x for x in open(p) if re.search(r"\d+ (passed|failed)", x)]
        out.append("pytest " + (last[-1].strip() if last else "?"))
    p = f"{d}/{t}_bench.log"
    if os.path.exists(p):
        cells = {}
        for x in open(p):
            r = json.loads(x.split("RESULT ", 1)[1])
            cells.setdefault((r["temp"], r["batch"]), []).append(r["tok_per_s"])
        out.append("bench " + " ".join(f"T{k[0]:g}B{k[1]}={statistics.median(v):.0f}(n{len(v)})"
                                       for k, v in sorted(cells.items())))
    p = f"{d}/{t}_suite.log"
    if os.path.exists(p):
        s = open(p).read()
        exits = " ".join(re.findall(r"compare(?:_long)?_exit \d+", s))
        out.append(f"suite {exits} tracebacks={s.count('Traceback')}")
    print(" | ".join(out))
