"""Which kernel runs for the int8 GEMMs, and at what TOPS (baseline diagnosis).

For every (N, K, rows) of the 1B / 8B GEMM shapes: time torch._int_mm (cutlass) and the gemm_skinny
Triton `_k` config the table would pick, print the profiler kernel names and whether the Triton PTX
contains wgmma / mma.sync. TOPS = 2*M*N*K / t. CUDA events, warmup 10, median of 50.

  python diag.py [--json out.json]
"""
import argparse
import collections
import json
import statistics

import torch

from bitexact.int8 import gemm_skinny as gs

SHAPES = "qkv1:3072x2048,o1:2048x2048,gu1:16384x2048,dn1:2048x8192,lm1:128256x2048," \
         "qkv8:6144x4096,o8:4096x4096,gu8:28672x4096,dn8:4096x14336,lm8:128256x4096"
ROWS = "128,256,2048,8192,32768"


def med_time(f, warm=10, rep=50):
    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    ev = [[torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)] for _ in range(rep)]
    for e0, e1 in ev:
        e0.record()
        f()
        e1.record()
    torch.cuda.synchronize()
    return statistics.median(e0.elapsed_time(e1) for e0, e1 in ev)


def kernels(f, iters=20):
    f()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            f()
        torch.cuda.synchronize()
    agg = collections.Counter()
    for e in p.key_averages():
        if e.self_device_time_total > 0:
            agg[e.key] += e.self_device_time_total / iters
    return agg


def cfg_for(cfg, w):
    """The gemm_skinny config int_mm() would pick for this (N, K) and the given rows, or None for cutlass."""
    for mx, c in gs.LARGE.get((w.shape[0], w.shape[1]), ()):
        if cfg <= mx:
            return ("LARGE", c)
    for mx, c in gs.TABLE.get((w.shape[0], w.shape[1]), ()):
        if cfg <= mx:
            return ("TABLE", c)
    return (None, None)


def ptx_flags(ck):
    """Which tensor-core / async-copy instructions the compiled kernel actually contains."""
    ptx = ck.asm["ptx"]
    return {"wgmma": "wgmma" in ptx, "mma.sync": "mma.sync" in ptx, "tcgen05": "tcgen05" in ptx,
            "cp.async.bulk": "cp.async.bulk" in ptx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="")
    ap.add_argument("--shapes", default=SHAPES)
    ap.add_argument("--rows", default=ROWS)
    a = ap.parse_args()
    g = torch.Generator(device="cuda").manual_seed(0)
    out = []
    for sh in a.shapes.split(","):
        name, nk = sh.split(":")
        N, K = map(int, nk.split("x"))
        w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda", generator=g)
        for M in map(int, a.rows.split(",")):
            x = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="cuda", generator=g)
            ref = torch._int_mm(x, w.t())
            t0 = med_time(lambda: torch._int_mm(x, w.t()))
            kn = kernels(lambda: torch._int_mm(x, w.t()))
            row = {"shape": nk, "name": name, "M": M, "tops": 2 * M * N * K / 1e9,
                   "cutlass_ms": round(t0, 4), "cutlass_tops": round(2 * M * N * K / 1e9 / t0),
                   "cutlass_kernels": {k: round(v, 4) for k, v in kn.most_common(4)}}
            where, c = cfg_for(M, w)
            row["table"] = where
            if c is not None:
                try:
                    ok = torch.equal(gs.gemm(x, w, c), ref)
                    t1 = med_time(lambda: gs.gemm(x, w, c))
                    kn = kernels(lambda: gs.gemm(x, w, c))
                    BM, BN, BK, S, nw, ns = c
                    import triton
                    ck = gs._k[(triton.cdiv(N, BN), triton.cdiv(M, BM), S)](
                        x, w, torch.empty((M, N), dtype=torch.int32, device="cuda"), M, N, K,
                        BM, BN, BK, S, K // S, num_warps=nw, num_stages=ns)
                    row.update({"tri_cfg": list(c), "tri_ms": round(t1, 4),
                                "tri_tops": round(2 * M * N * K / 1e9 / t1), "tri_equal": ok,
                                "tri_kernels": {k: round(v, 4) for k, v in kn.most_common(3)},
                                "tri_ptx": ptx_flags(ck)})
                except Exception as ex:                                  # noqa: BLE001 - report, do not stop
                    row["tri_err"] = repr(ex)[:200]
            out.append(row)
            print(json.dumps(row), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()