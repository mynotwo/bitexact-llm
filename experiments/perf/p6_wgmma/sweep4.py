"""Two-stage config sweep for gemm_sm90: prune with plain event timing, then CUDA-graph timing on the best.

  python sweep4.py --rows 128,256 --shapes 3072x2048,... --grid small [--json out.json]
Stage 1: one event timing per cfg (fast, launch-biased); keep the best PRUNE. Stage 2: graph timing (serve
replays a graph, and its weights are HBM-cold) on those, rotating weight copies as bench_graph.py does.
Every cfg kept is checked exact with torch.equal against torch._int_mm (random, all -128, all 127).
"""
import argparse
import itertools
import json
import statistics

import torch

from bitexact.int8 import gemm_skinny, gemm_sm90
from gt import graph_time, timeit

SMALL = list(itertools.product((64, 128), (64, 128, 256), (64, 128, 256), (4, 8, 16), (4, 8), (3, 4, 5), (0, 1)))
LARGE = list(itertools.product((64, 128, 256), (128, 256), (64, 128, 256), (4, 8, 16), (8,), (3, 4, 5), (0, 1)))


def med(f, warm=3, rep=15):
    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(rep)]
    for e0, e1 in ev:
        e0.record()
        f()
        e1.record()
    torch.cuda.synchronize()
    return statistics.median(e0.elapsed_time(e1) for e0, e1 in ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="128,256")
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--grid", default="small")
    ap.add_argument("--prune", type=int, default=16)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    cfgs = SMALL if a.grid == "small" else LARGE
    g = torch.Generator(device="cuda").manual_seed(0)
    out = []
    for sh in a.shapes.split(","):
        N, K = map(int, sh.split("x"))
        ncp = max(2, min(16, int(3e9 // (N * K))))                 # weight copies > L2 footprint: cold reads
        ws = [torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda", generator=g) for _ in range(ncp)]
        w0 = ws[0]
        for M in map(int, a.rows.split(",")):
            x = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="cuda", generator=g)
            ref = torch._int_mm(x, w0.t())
            tc = timeit(lambda i: torch._int_mm(x, ws[i % ncp].t()), M, N, ncp)
            to = timeit(lambda i: gemm_skinny.int_mm(x, ws[i % ncp]), M, N, ncp)   # what the repo does today
            fl = 2 * M * N * K / 1e9
            wmb = N * K / 1e6
            stage1, bad = [], 0
            for cfg in cfgs:
                bm, bn, bk, gm, nw, ns, wsx = cfg
                if K % bk or bn * bk * ns * 2 > 250000:
                    continue
                try:
                    if not torch.equal(gemm_sm90.gemm(x, w0, cfg), ref):
                        bad += 1
                        continue
                    stage1.append((med(lambda: gemm_sm90.gemm(x, w0, cfg)), cfg))
                except Exception:                                  # noqa: BLE001 - cfg not supported
                    continue
            stage1.sort()
            res = []
            for _, cfg in stage1[:a.prune]:
                try:
                    res.append((timeit(lambda i, c=cfg: gemm_sm90.gemm(x, ws[i % ncp], c), M, N, ncp), cfg))
                except Exception:                                  # noqa: BLE001
                    continue
            res.sort()
            out.append({"shape": sh, "M": M, "wmb": round(wmb, 1), "cutlass_ms": round(tc, 4),
                        "old_ms": round(to, 4), "old_tops": round(fl / to),
                        "cutlass_tops": round(fl / tc), "cutlass_gbs": round(wmb / tc / 1000),
                        "best": [{"cfg": list(c), "ms": round(t, 4), "tops": round(fl / t),
                                  "gbs": round(wmb / t / 1000)} for t, c in res[:a.top]],
                        "n_nonequal": bad, "n_tried": len(stage1)})
            print(json.dumps(out[-1]), flush=True)
            del x
        del ws
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()