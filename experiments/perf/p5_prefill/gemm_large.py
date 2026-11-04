"""Large-M int8 GEMM sweep: torch._int_mm (cutlass) vs gemm_skinny._k configs, rows = 2 * tokens (two limbs).
python gemm_large.py [--rows 32768] ; prints ms per (N, K, cfg) and whether the integers match."""
import argparse
import itertools

import torch
import triton

from bitexact.int8 import gemm_skinny


def bench(f, n=20):
    f(); f()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        f()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="512,1024,2048,4096,8192,16384,32768,65536")
    ap.add_argument("--shapes", default="3072x2048,2048x2048,16384x2048,2048x8192")
    a = ap.parse_args()
    g = torch.Generator(device="cuda").manual_seed(0)
    for sh, rows in itertools.product(a.shapes.split(","), map(int, a.rows.split(","))):
        a.rows_ = rows
        N, K = map(int, sh.split("x"))
        x = torch.randint(-128, 128, (a.rows_, K), dtype=torch.int8, device="cuda", generator=g)
        w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda", generator=g)
        ref = torch._int_mm(x, w.t())
        t0 = bench(lambda: torch._int_mm(x, w.t()))
        tops = 2 * a.rows_ * N * K / 1e9
        print(f"{N}x{K} rows {a.rows_}: _int_mm {t0:.3f} ms ({tops / t0:.0f} TOPS)", flush=True)
        res = []
        for bm, bn, bk, nw, ns in itertools.product((64, 128, 256), (64, 128, 256), (64, 128, 256), (4, 8), (3, 4, 5)):
            if bm * bn > 128 * 256 or K % bk or bk * (bm + bn) * ns > 200000:
                continue
            cfg = (bm, bn, bk, 1, nw, ns)
            try:
                c = gemm_skinny.gemm(x, w, cfg)
                ok = torch.equal(c, ref)
                t = bench(lambda: gemm_skinny.gemm(x, w, cfg), 10)
            except Exception as ex:
                continue
            res.append((t, cfg, ok))
        for t, cfg, ok in sorted(res)[:2]:
            print(f"   {cfg} {t:.3f} ms ({tops / t:.0f} TOPS) equal={ok}", flush=True)


if __name__ == "__main__":
    main()
