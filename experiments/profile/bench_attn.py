"""Scorer attention, existing kernels only: the engine's unfused path (bmm q.k -> int64 scores -> compiled softmax
segment -> bmm p.v) vs the plugin's fused paged kernels (tri_paged.attend: two sweeps, no score tensor), on the
same integer q / K / V for one prefill chunk. Checks num and den are bit-identical, reports times.

  python bench_attn.py --B 4 --T 256 --start 3840 --out OUT.json
"""
import argparse
import json
import statistics
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--start", type=int, nargs="+", default=[0, 1024, 3840])
    ap.add_argument("--qk-bits", type=int, default=15, help="|q|, |k| < 2^bits (FQ = 14 fractional bits)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from bitexact.bmm import bmm
    from bitexact.int8 import consts
    from bitexact.int8.segs import softmax
    from bitexact.llama import H, HD, HKV
    from bitexact.vllm_plugin import tri_paged
    torch.manual_seed(0)
    dev = "cuda"
    g = H // HKV
    expt = consts.exp_table().to(dev)
    ptab = tri_paged.weight_table(expt)
    sm = torch.compile(softmax, dynamic=None)
    rows = []
    for st in a.start:
        B, T, S = a.B, a.T, st + a.T
        lim = 1 << a.qk_bits
        qg = torch.randint(-lim, lim, (B * HKV, g * T, HD), device=dev)
        K = torch.randint(-lim, lim, (B, HKV, S, HD), device=dev, dtype=torch.int32)
        V = torch.randint(-(1 << 15), 1 << 15, (B, HKV, S, HD), device=dev, dtype=torch.int32)
        pos = torch.arange(st, S, device=dev).expand(B, T)
        qpos = pos[:, None, None, :].expand(B, 1, g, T).reshape(B, 1, g * T, 1)

        def unfused():
            raw = bmm(qg, K.transpose(-1, -2).reshape(B * HKV, HD, S)).view(B, HKV, g * T, S)
            p, den = sm(raw, qpos, expt)
            num = bmm(p.view(B * HKV, g * T, S), V.reshape(B * HKV, S, HD)).view(B, HKV, g * T, HD)
            return num, den
        bs = 16
        Sp = (S + bs - 1) // bs * bs
        kv = torch.zeros(B, HKV, Sp, 2 * HD, dtype=torch.int32, device=dev)
        kv[:, :, :S, :HD], kv[:, :, :S, HD:] = K, V
        nb = Sp // bs
        cache = kv.view(B, HKV, nb, bs, 2 * HD).permute(0, 2, 1, 3, 4).reshape(B * nb, HKV, bs, 2 * HD).contiguous()
        bt = torch.arange(B * nb, device=dev).view(B, nb)
        q4 = qg.view(B, HKV, g * T, HD)
        qp = qpos.view(B, g * T)

        def fused():
            return tri_paged.attend(q4, qp, cache, bt, ptab)

        def tm(f, n=10):
            f()
            torch.cuda.synchronize()
            out = []
            for _ in range(n):
                t0 = time.perf_counter()
                f()
                torch.cuda.synchronize()
                out.append(time.perf_counter() - t0)
            return statistics.median(out) * 1e3
        n1, d1 = unfused()
        n2, d2 = fused()
        r = {"B": B, "T": T, "start": st, "S": S, "unfused_ms": round(tm(unfused), 3), "fused_ms": round(tm(fused), 3),
             "num_equal": bool(torch.equal(n1, n2)), "den_equal": bool(torch.equal(d1, d2)),
             "nonzero_den_rows": int((d1 > 0).sum())}
        r["speedup"] = round(r["unfused_ms"] / r["fused_ms"], 2)
        print(json.dumps(r), flush=True)
        rows.append(r)
    json.dump({"gpu": torch.cuda.get_device_name(0), "args": vars(a), "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
