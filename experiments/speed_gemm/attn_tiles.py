"""Tile sweep for tri_paged.attend (bm, bn, warps) on scorer-like and decode-like shapes; outputs must stay equal."""
import itertools
import json
import sys

import torch
import triton

from bitexact.int8.consts import exp_table
from bitexact.vllm_plugin import tri_paged

g = torch.Generator(device="cuda").manual_seed(0)
ptab = tri_paged.weight_table(exp_table().cuda())
bs, HKV, HD = 16, 8, 64
# (name, R, M, keys): scorer 4k b4 last chunk (4 q heads x 256 positions), scorer GSM8K b8, decode B=64 at 512 / 2304
CASES = [("scorer4k_b4", 4, 1024, 4096), ("scorer_gsm_b8", 8, 1024, 512), ("decode_b64_512", 64, 4, 512),
         ("decode_b64_2304", 64, 4, 2304)]
rows = []
for name, R, M, S in CASES:
    nb = R * S // bs + 1
    c = torch.randint(-(1 << 17), 1 << 17, (nb, HKV, bs, 2 * HD), device="cuda", generator=g, dtype=torch.int32)
    bt = (1 + torch.arange(R * S // bs, device="cuda")).view(R, -1).to(torch.int32)
    q = torch.randint(-(1 << 19), 1 << 19, (R, HKV, M, HD), device="cuda", generator=g)
    qpos = (S - M // 4 + torch.arange(M, device="cuda") % (M // 4))[None].expand(R, -1).contiguous()
    ref = tri_paged.attend(q, qpos, c, bt, ptab)
    t0 = triton.testing.do_bench(lambda: tri_paged.attend(q, qpos, c, bt, ptab), warmup=10, rep=100)
    for bm, bn, w in itertools.product((16, 32, 64, 128), (32, 64, 128), (4, 8)):
        if M <= 16 and bm > 16:
            continue
        try:
            out = tri_paged.attend(q, qpos, c, bt, ptab, bm=bm, bn=bn, warps=w)
            eq = all(torch.equal(x, y) for x, y in zip(out, ref))
            t = triton.testing.do_bench(lambda: tri_paged.attend(q, qpos, c, bt, ptab, bm=bm, bn=bn, warps=w),
                                        warmup=10, rep=100)
        except Exception as e:                              # out of shared memory / registers
            print(name, bm, bn, w, "fail", type(e).__name__, flush=True)
            continue
        r = dict(case=name, bm=bm, bn=bn, warps=w, us=t * 1e3, default_us=t0 * 1e3, speedup=t0 / t, equal=eq)
        rows.append(r)
        print(f"{name:16s} bm={bm:3d} bn={bn:3d} w={w} {r['us']:9.1f}us default {r['default_us']:9.1f}us "
              f"x{r['speedup']:.2f} eq={eq}", flush=True)
json.dump(rows, open(sys.argv[1], "w"), indent=1)
