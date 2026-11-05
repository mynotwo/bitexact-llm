"""Microbench: torch._int_mm (cutlass sm80) vs the old gemm_skinny table vs the new gemm_sm90 (wgmma + TMA).

One ROW json line per (shape, rows): ms / TOPS / GB/s of each path and the cfg gemm_sm90's table picks.
Timing is CUDA-graph replay with the weight rotating over several HBM-cold copies, matching serve (graphs,
weights not resident in L2) and hiding the ~17 us launch overhead that dominates a plain event timing at these
sizes; see gt.py. Every candidate is checked with torch.equal against torch._int_mm on random inputs and on
all -128 / all 127 inputs.

`old` is the base-commit gemm_skinny, loaded from the base tree with --old (its int_mm is cutlass or the p2/p5 Triton
tables), since the repo copy now calls gemm_sm90 itself.

  python bench_gemm.py --old /data41/.../wgmma/base/src/bitexact/int8/gemm_skinny.py --json out.json
"""
import argparse
import importlib.util
import json

import torch

from bitexact.int8 import gemm_sm90, gemm_skinny
from gt import graph_time

SHAPES = "qkv1:3072x2048,o1:2048x2048,gu1:16384x2048,dn1:2048x8192,lm1:128256x2048," \
         "qkv8:6144x4096,o8:4096x4096,gu8:28672x4096,dn8:4096x14336,lm8:128256x4096"


def exact(f, x, w, ref):
    """The candidate is exact on random inputs and on the two extremes."""
    for xx in (x, torch.full_like(x, -128), torch.full_like(x, 127)):
        for ww in (w, torch.full_like(w, -128), torch.full_like(w, 127)):
            if not torch.equal(f(xx, ww), torch._int_mm(xx, ww.t())):
                return False
    return torch.equal(f(x, w), ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="128,256,384,512,2048,8192,32768")
    ap.add_argument("--shapes", default=SHAPES)
    ap.add_argument("--json", default="")
    ap.add_argument("--old", default="", help="base-tree gemm_skinny.py to time as the old path")
    a = ap.parse_args()
    if a.old:
        spec = importlib.util.spec_from_file_location("gemm_skinny_base", a.old)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
    else:
        old = gemm_skinny
    g = torch.Generator(device="cuda").manual_seed(0)
    out = []
    for sh in a.shapes.split(","):
        name, nk = sh.split(":")
        N, K = map(int, nk.split("x"))
        ncp = max(2, min(16, int(3e9 // (N * K))))         # copies > L2 footprint, so every replay reads HBM
        ws = [torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda", generator=g) for _ in range(ncp)]
        w = ws[0]
        n = 4 * ncp
        for M in map(int, a.rows.split(",")):
            x = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="cuda", generator=g)
            ref = torch._int_mm(x, w.t())
            fl = 2 * M * N * K / 1e9
            cfg = gemm_sm90.pick(N, K, M)
            def f_new(i):
                return gemm_sm90.gemm(x, ws[i % ncp], cfg) if cfg else torch._int_mm(x, ws[i % ncp].t())
            times = {"cutlass": graph_time(lambda i: torch._int_mm(x, ws[i % ncp].t()), n),
                     "old": graph_time(lambda i: old.int_mm(x, ws[i % ncp]), n),
                     "new": graph_time(f_new, n)}
            row = {"name": name, "shape": nk, "M": M, "gflop": round(fl, 2), "new_cfg": list(cfg or [])}
            for kx, t in times.items():
                row.update({f"{kx}_ms": round(t, 4), f"{kx}_tops": round(fl / t),
                            f"{kx}_gbs": round(N * K / 1e6 / t / 1000)})
            row.update({"old_equal": torch.equal(old.int_mm(x, w), ref),
                        "new_equal": exact(f_new, x, w, ref)})
            out.append(row)
            print("ROW " + json.dumps(row), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()