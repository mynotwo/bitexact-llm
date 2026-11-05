import json, sys, statistics as st
# usage: med.py OUTDIR  -> median of steps 2-19 per run
for rep in (1, 2):
    for c in ("A", "D", "Cp_base", "Cp_new"):
        rs = [json.loads(l) for l in open(f"{sys.argv[1]}/t{rep}_{c}/steps.jsonl") if '"time"' in l][2:]
        f = lambda k: st.median(r["time"].get(k, 0) for r in rs)
        print(rep, c, len(rs), "step %.1f" % st.median(r["step_sec"] for r in rs), "sample %.1f score %.1f trainer %.1f" % (f("sample"), f("score_int"), f("backward") + f("optim") + f("quantize")))
