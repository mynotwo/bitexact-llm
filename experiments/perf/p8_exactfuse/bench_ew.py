"""Microbench of the fused element-wise kernels (exact_fuse.py), CUDA-graph timing, us per call.
  python bench_ew.py CONST [ROWS=1,8,32,128] """
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gtime import graph_time  # noqa: E402

from bitexact.exact.model import ExactLlama  # noqa: E402
from bitexact.vllm_plugin import exact_fuse as ef  # noqa: E402

c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = ExactLlama(c, "cuda", n_layers=2)
g = torch.Generator(device="cuda").manual_seed(0)
fl = torch.zeros(1, dtype=torch.int32, device="cuda")


def rnd(shape, lo, hi):
    b = torch.randint(lo, hi, (shape[0],) + (1,) * (len(shape) - 1), device="cuda", generator=g)
    return ((torch.rand(shape, device="cuda", generator=g, dtype=torch.float64) * 2 - 1) * torch.pow(2.0, b.double())).round().long()


import os
for k in ('NORM','ATTN','SILU'):
    if os.environ.get('NW_'+k): setattr(ef, 'NW_'+k, int(os.environ['NW_'+k]))
H, HKV, HD, D = m.H, m.HKV, m.HD, m.D
FF = c["0.wd.limbs"].shape[2]
for M in map(int, (sys.argv[2] if len(sys.argv) > 2 else "1,8,32,128").split(",")):
    x, y = rnd((M, D), 22, 26), rnd((M, D), 20, 24)
    yq = rnd((M, (H + 2 * HKV) * HD), 22, 26)
    pos = torch.randint(0, 200, (M,), device="cuda", generator=g)
    slot = torch.arange(M, device="cuda")
    cache = torch.zeros(M // 16 + 2, HKV, 16, 8 * HD, dtype=torch.int8, device="cuda")
    num = rnd((M, HKV, H // HKV, HD), 36, 40)
    den = torch.randint(1 << 22, 1 << 33, (M, HKV, H // HKV, 1), device="cuda", generator=g)
    yg = rnd((M, 2 * FF), 24, 27)
    t = lambda f: round(graph_time(f), 1)  # noqa: E731
    print(f"M={M:4d} add_norm {t(lambda: ef.add_norm(m, y, x, m.layers[0]['gm'], fl))}  "
          f"qkv {t(lambda: ef.qkv(m, yq, pos, slot, cache, fl))}  attn_out {t(lambda: ef.attn_out(m, num, den))}  "
          f"silu {t(lambda: ef.silu_quant(m, yg, m.exp2, fl))}", flush=True)
