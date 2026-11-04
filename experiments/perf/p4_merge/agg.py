"""Aggregate p4_merge outputs: python agg.py OUT_DIR"""
import json, glob, re, os, sys, statistics as st
O = sys.argv[1]
med = st.median
def res(f):
    for l in open(f, errors="ignore"):
        if l.startswith("RESULT"):
            return json.loads(l[7:])
def tps(j): return j.get("tok_per_s") or j.get("tps")
def fmt(v): return f"{med(v):,.0f} [{min(v):,.0f}-{max(v):,.0f}]"
for name in ("bench8", "bench"):
    print(f"## {name} tok/s median [min-max]")
    rows = {}
    for f in glob.glob(f"{O}/{name}/*.log"):
        m = re.match(r"(base|repo|stock|bi|vllm_stock|vllm_bi)_b(\d+)(?:_t(\d))?_r(\d+)", os.path.basename(f))
        j = res(f)
        if m and j: rows.setdefault((m[1], int(m[2]), m[3]), []).append(tps(j))
    for k in sorted(rows, key=lambda k: (k[1], str(k[2]), k[0])): print(k, len(rows[k]), fmt(rows[k]))
print("## prof8 B=128 T=0 decode ms/step (median of 3)")
for t in ("base", "repo"):
    ds = [json.load(open(f)) for f in sorted(glob.glob(f"{O}/prof8/{t}?/int_b128.json"))]
    if not ds: continue
    g = lambda f: med([f(d) for d in ds])
    bk = lambda k: g(lambda d: d["decode"]["buckets_ms_per_step"][k])
    wall = g(lambda d: d["timing"]["decode_step_ms_median"]); busy = g(lambda d: d["decode"]["busy_ms_per_step"])
    print(f"{t} n={len(ds)} wall={wall:.3f} kernel={g(lambda d: d['decode']['kernel_ms_per_step']):.3f} gemm={bk('gemm_int8'):.3f} elem={bk('elementwise'):.3f} attn={bk('attention'):.3f} samp={bk('sampling_logits'):.3f} other={bk('other'):.3f} wall-busy={wall-busy:.3f} idle={g(lambda d: d['decode']['gpu_idle_ms_per_step']):.3f} kern/step={g(lambda d: d['decode']['kernels_per_step']):.0f} tok/s={g(lambda d: d['tok_per_s']):.0f}")
print("## longctx8 decode_ms_per_step median [min-max]")
rows = {}
for f in glob.glob(f"{O}/longctx8/*.log"):
    m = re.match(r"(base|repo)_s(\d+)_b(\d+)_r", os.path.basename(f))
    for l in open(f, errors="ignore"):
        if l.startswith("{"):
            j = json.loads(l); rows.setdefault((m[2], m[3], m[1]), []).append(j["decode_ms_per_step"]); break
for k in sorted(rows): v = rows[k]; print(k, len(v), f"{med(v):.3f} [{min(v):.3f}-{max(v):.3f}]")
