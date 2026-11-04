"""Root cause of the single-CTA lm_head epilogue: the dynamic-shape recompile of the shared Segs.post.

Segs.post is one torch.compile(dynamic=None) function for every GEMM epilogue (q/k/v, o, gate/up, down, lm_head). Its
overflow flag is amax over the whole [M, N] output, a reduction to one element. The first (static) compile splits it
over many CTAs. The first call with another row count or width recompiles with symbolic sizes, and that kernel has
grid (1, 1, 1): one CTA loops over all M x N elements. Below, lm_head logits for 64 rows after several recompile
orders (second shape from 2 to 512 rows) always take about 4 ms (8 ms inside vLLM, where the whole step runs) against
0.09-0.3 ms for the static kernel; vllm_plugin.logits.lm_logits, with static bucketed compiles, takes 0.3 ms after the
same sequences.

  python rootcause.py --const C.pt --out OUT.json
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
    x = torch.randint(-(1 << 20), 1 << 20, (64, 2048), device="cuda")
    ly = IntLlama(c, "cuda", nl=2, compiled=False).layers[0]

    def fresh():
        return IntLlama(c, "cuda", nl=2, compiled=True)

    def tm(mm, rows, n=20):
        xs = x[:rows]
        mm.logits(xs)
        torch.cuda.synchronize()
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            mm.logits(xs)
            torch.cuda.synchronize()
            out.append(time.perf_counter() - t0)
        return round(statistics.median(out) * 1e3, 3)

    from bitexact.vllm_plugin.logits import lm_logits

    def tm_plugin(mm, rows, n=20):
        xs = x[:rows]
        lm_logits(mm, xs)
        torch.cuda.synchronize()
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            lm_logits(mm, xs)
            torch.cuda.synchronize()
            out.append(time.perf_counter() - t0)
        return round(statistics.median(out) * 1e3, 3)

    big = torch.randint(-(1 << 20), 1 << 20, (512, 2048), device="cuda")
    res = {}
    # Segs.post is one function for every GEMM, so its shapes at the moment of each recompile decide the size hint
    # baked into the kernel. Sequences of (rows, weight) calls, then logits for 64 rows:
    seqs = {
        "wo512_wqkv3": [(512, "wo"), (3, "wqkv")],
        "wo512_wo2_wqkv3": [(512, "wo"), (2, "wo"), (3, "wqkv")],
        "wqkv512_wo2_wgu4": [(512, "wqkv"), (2, "wo"), (4, "wgu")],
        "wo512_wqkv2_wo3": [(512, "wo"), (2, "wqkv"), (3, "wo")],
        "wo3_wo512": [(3, "wo"), (512, "wo")],
        "wo3_wo512_wo4": [(3, "wo"), (512, "wo"), (4, "wo")],
    }
    for name, seq in seqs.items():
        m = fresh()
        for rows, w in seq:
            m.norm_linear(big[:rows], ly["ga"], ly[w])
        res[name + "_then_logits64_ms"] = tm(m, 64)
        res[name + "_then_plugin_lm_logits64_ms"] = tm_plugin(m, 64)
    print(json.dumps(res), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
