"""Time bitexact.bmm.int_mm (BITEXACT_GEMM path) against torch._int_mm on Llama-3.2-1B linear shapes; check bits."""
import json
import sys

import torch
import triton

from bitexact.bmm import int_mm

SHAPES = {"qkv": (3072, 2048), "o": (2048, 2048), "gate_up": (16384, 2048), "down": (2048, 8192),
          "lm_head": (128256, 2048)}
rows = []
for name, (N, K) in SHAPES.items():
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda")
    for M in (24, 128, 1024, 4096, 16384):              # int_mm rows = 2 limbs x tokens: B=1 decode .. 8192 tokens
        if name == "lm_head" and M > 4096:
            continue
        a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="cuda")
        eq = bool(torch.equal(int_mm(a, w, "lt"), torch._int_mm(a, w.t())))
        t0 = triton.testing.do_bench(lambda: torch._int_mm(a, w.t()), warmup=25, rep=200)
        t1 = triton.testing.do_bench(lambda: int_mm(a, w, "lt"), warmup=25, rep=200)
        ops = 2 * M * N * K
        r = dict(name=name, M=M, N=N, K=K, intmm_us=t0 * 1e3, lt_us=t1 * 1e3, intmm_tops=ops / t0 / 1e9,
                 lt_tops=ops / t1 / 1e9, speedup=t0 / t1, equal=eq)
        rows.append(r)
        print(f"{name:8s} M={M:6d} int_mm {r['intmm_us']:8.1f}us {r['intmm_tops']:5.0f}T  lt {r['lt_us']:8.1f}us "
              f"{r['lt_tops']:5.0f}T  x{r['speedup']:.2f} eq={eq}", flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
