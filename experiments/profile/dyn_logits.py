"""Time compute_logits (m.logits, compiled segments) for 64 rows: static, after a call with another row count, and
after the shared post segment has also seen another output width (as in vLLM, where Segs.post serves the q/k/v,
gate/up and lm_head GEMMs and prefill steps bring other row counts). torch.compile(dynamic=None) recompiles with
dynamic sizes for every dimension that changed.

  python dyn_logits.py --const C.pt --out OUT.json
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

    def tm(rows, n=20):
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
    res = {"ms_64_static": tm(64), "ms_7_second_shape": tm(7), "ms_64_after_dynamic_rows": tm(64)}
    ly = m.layers[0]
    for rows in (64, 7):                                   # q/k/v and gate/up GEMMs through the same post segment
        xs = x[:rows]
        m.norm_linear(xs, ly["ga"], ly["wqkv"])
        m.norm_linear(xs, ly["gm"], ly["wgu"])
    torch.cuda.synchronize()
    res["ms_64_after_dynamic_rows_and_width"] = tm(64)
    res["ms_1_after"] = tm(1)
    big = torch.randint(-(1 << 20), 1 << 20, (2048, 2048), device="cuda")
    m.logits(big)                                          # a large row count, as in vLLM's profiling / prefill runs
    torch.cuda.synchronize()
    res["ms_64_after_2048_rows"] = tm(64)
    print(json.dumps(res), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
