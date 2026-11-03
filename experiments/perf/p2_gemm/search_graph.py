"""Offline per-shape config search, timed under CUDA graph with HBM-cold rotating weights. Writes configs json.
Selection is offline; integers are exact so the chosen config cannot change any output."""
import sys, json, itertools, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bench_graph import graph_us, SH
from bitexact.int8.gemm_skinny import gemm
ROWS = [24, 64, 256]
out = {}
for name, (N, K) in SH.items():
    ncp = max(2, min(12, int(3e9 // (N * K))))
    ws = [torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda") for _ in range(ncp)]
    for R in ROWS:
        a = torch.randint(-128, 128, (R, K), dtype=torch.int8, device="cuda")
        ref = torch._int_mm(a, ws[0].t())
        ti = graph_us(lambda i: torch._int_mm(a, ws[i % ncp].t()), 3 * ncp)
        best = (1e9, None)
        BMs = [16, 32] if R <= 64 else [64, 128]
        for cfg in itertools.product(BMs, [32, 64, 128, 256], [128, 256], [1, 2, 4, 8], [4, 8], [3, 4, 5]):
            BM, BN, BK, S, nw, ns = cfg
            if K % (BK * S) or (BM + BN) * BK * ns > 190000: continue
            try:
                if not torch.equal(gemm(a, ws[0], cfg), ref): print("MISMATCH", cfg); continue
                x = graph_us(lambda i: gemm(a, ws[i % ncp], cfg), 3 * ncp)
            except Exception as e:
                continue
            if x < best[0]: best = (x, cfg)
        roof = N * K / 3.35e12 * 1e6
        print(f"{name},{R} int_mm={ti:.1f} tri={best[0]:.1f} roof={roof:.1f} {best[1]}", flush=True)
        out[f"{name},{R}"] = dict(int_mm=ti, tri=best[0], roof=roof, cfg=best[1])
    del ws
    json.dump(out, open(sys.argv[1], "w"), indent=1)
