"""LM-head GEMM (N = 128256, K = 2048) at rows = 2 x scored tokens: cutlass _int_mm vs gemm_skinny._k configs."""
import itertools
import torch
from bitexact.int8 import gemm_skinny


def bench(f, n=10):
    f(); f()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        f()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


g = torch.Generator(device="cuda").manual_seed(0)
N, K = 128256, 2048
w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda", generator=g)
for rows in (1024, 2048, 4096):
    x = torch.randint(-128, 128, (rows, K), dtype=torch.int8, device="cuda", generator=g)
    ref = torch._int_mm(x, w.t())
    t0 = bench(lambda: torch._int_mm(x, w.t()))
    tops = 2 * rows * N * K / 1e9
    print(f"rows {rows}: _int_mm {t0:.3f} ms ({tops / t0:.0f} TOPS)", flush=True)
    res = []
    for bm, bn, bk, nw, ns in itertools.product((64, 128, 256), (64, 128, 256), (64, 128, 256), (4, 8), (3, 4, 5)):
        if bm * bn > 128 * 256 or K % bk or bk * (bm + bn) * ns > 200000:
            continue
        cfg = (bm, bn, bk, 1, nw, ns)
        try:
            ok = torch.equal(gemm_skinny.gemm(x, w, cfg), ref)
            t = bench(lambda: gemm_skinny.gemm(x, w, cfg))
        except Exception:
            continue
        res.append((t, cfg, ok))
    for t, cfg, ok in sorted(res)[:3]:
        print(f"   {cfg} {t:.3f} ms ({tops / t:.0f} TOPS) equal={ok}", flush=True)
