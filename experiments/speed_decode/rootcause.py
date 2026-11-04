"""Root cause of the single-CTA lm_head epilogue: the split-reduction decision of the shared dynamic-shape compile.

Segs.post is one torch.compile(dynamic=None) function for every GEMM epilogue (q/k/v, o, gate/up, down, lm_head). Its
overflow flag is amax over the whole [M, N] output, a reduction to one element. Inductor decides how many CTAs share
a reduction (RSPLIT) from the *size hint* of the shape the graph was traced with, and bakes it into the kernel. After a
size change the dynamic-shape recompile traces with the new shape as its hint: when that shape is small
(reduction_numel_hint <= 8192, Choices.reduction_split_factor) the kernel is compiled without a split, and the same
kernel then runs the lm_head epilogue (M x 128,256) in one CTA.

  python rootcause.py --const C.pt --out OUT.json [--first-rows 1] [--first-width 2048]
"""
import argparse
import json
import statistics
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--const", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from bitexact.int8.model import IntLlama
    c = torch.load(a.const, map_location="cpu", weights_only=False)
    m = IntLlama(c, "cuda", nl=2, compiled=True)
    x = torch.randint(-(1 << 20), 1 << 20, (64, 2048), device="cuda")
    ly = m.layers[0]

    def tm_logits(rows, n=20):
        xs = x[:rows]
        m.logits(xs)
        torch.cuda.synchronize()
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            m.logits(xs)
            torch.cuda.synchronize()
            out.append(time.perf_counter() - t0)
        return round(statistics.median(out) * 1e3, 3)

    res = {}
    m.norm_linear(x, ly["ga"], ly["wqkv"])                     # first compile: static [64, 3072]
    # a second shape. Large hint (64 x 2048 = 131072 > 8192): the dynamic kernel keeps a split
    m.norm_linear(x[:61], ly["gm"], ly["wgu"])
    res["after_large_hint"] = tm_logits(64)
    # lm_head after a recompile whose hint is tiny
    m2 = IntLlama(c, "cuda", nl=2, compiled=True)
    m2.norm_linear(x, ly["ga"], ly["wqkv"])
    m2.norm_linear(x[:1], ly["ga"], ly["wqkv"])                # M = 1, N = 3072 -> hint 3072 <= 8192
    xs = x[:64]
    m2.logits(xs)
    torch.cuda.synchronize()
    ts = []
    for _ in range(20):
        t0 = time.perf_counter()
        m2.logits(xs)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    res["after_small_hint"] = round(statistics.median(ts) * 1e3, 3)
    print(json.dumps(res), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
