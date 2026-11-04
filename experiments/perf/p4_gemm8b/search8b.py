"""Offline tile search for Llama-3.1-8B decode linears (+LM head), CUDA-graph timing, HBM-cold rotating weights.
Usage: search8b.py out.json [name ...]. Selection is offline; integer sums are exact so config cannot change outputs."""
import sys, json, itertools, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/../p2_gemm")
from bench_graph import graph_us
from bitexact.int8.gemm_skinny import gemm
SH = {"wqkv": (6144, 4096), "wo": (4096, 4096), "wgu": (28672, 4096), "wd": (4096, 14336), "lm": (128256, 4096)}
import os
ROWS = [int(x) for x in os.environ.get("ROWS", "24,64,128,256").split(",")]
EXT = os.environ.get("EXT") == "1"   # wider grid: BM 256, BK 64, 2..6 stages, 16 warps
names = sys.argv[2:] or list(SH)
out = {}
for name in names:
    N, K = SH[name]
    ncp = max(2, min(12, int(3e9 // (N * K))))
    ws = [torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda") for _ in range(ncp)]
    for R in ROWS:
        a = torch.randint(-128, 128, (R, K), dtype=torch.int8, device="cuda")
        ref = torch._int_mm(a, ws[0].t())
        try: ti = graph_us(lambda i: torch._int_mm(a, ws[i % ncp].t()), 3 * ncp)
        except Exception: ti = float("nan")
        best = (1e9, None)
        BMs = [16, 32] if R <= 64 else [32, 64, 128] + ([256] if EXT else [])
        for cfg in itertools.product(BMs, [32, 64, 128, 256], [64, 128, 256] if EXT else [128, 256], [1, 2, 4, 8], [4, 8, 16] if EXT else [4, 8], [2, 3, 4, 5, 6] if EXT else [3, 4, 5]):
            BM, BN, BK, S, nw, ns = cfg
            if K % (BK * S) or (BM + BN) * BK * ns > 225000: continue
            try:
                if not torch.equal(gemm(a, ws[0], cfg), ref): print("MISMATCH", cfg); continue
                x = graph_us(lambda i: gemm(a, ws[i % ncp], cfg), 3 * ncp)
            except Exception: continue
            if x < best[0]: best = (x, cfg)
        roof = N * K / 3.35e12 * 1e6
        print(f"{name},{R} int_mm={ti:.1f} tri={best[0]:.1f} roof={roof:.1f} {best[1]}", flush=True)
        out[f"{name},{R}"] = dict(int_mm=ti, tri=best[0], roof=roof, cfg=best[1])
    del ws
    json.dump(out, open(sys.argv[1], "w"), indent=1)
