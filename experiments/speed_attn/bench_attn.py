"""Kernel micro-benchmark of tri_paged.attend at decode shape (M = 4 query rows per KV head, R requests of ~ctx keys).
  python bench_attn.py [R] [ctx] [layout]    layout int32 (K | V as int32) or limb (3 + 3 int8 planes)."""
import sys

import torch

from bitexact.int8.consts import exp_table
from bitexact.vllm_plugin import tri_paged

R = int(sys.argv[1]) if len(sys.argv) > 1 else 430
ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 400
limb = len(sys.argv) > 3 and sys.argv[3] == "limb"
HKV, HD, bs = 8, 64, 16
g = torch.Generator(device="cuda").manual_seed(0)
nbr = (ctx + bs - 1) // bs + 1
nb = R * nbr + 1
lim = 1 << 20
if limb:
    c = torch.randint(-128, 128, (nb, HKV, bs, 6 * HD), device="cuda", generator=g, dtype=torch.int8)
else:
    c = torch.randint(-lim, lim, (nb, HKV, bs, 2 * HD), device="cuda", generator=g).to(torch.int32)
bt = (1 + torch.arange(R * nbr, device="cuda").view(R, nbr)).to(torch.int32)
qpos = (ctx - 1 - torch.randint(0, 50, (R, 4), device="cuda", generator=g))
q = torch.randint(-lim, lim, (R, HKV, 4, HD), device="cuda", generator=g)
ptab = tri_paged.weight_table(exp_table().cuda())
for bn, w in [(b, w) for b in (32, 64) for w in (2, 4)]:
    f = lambda: tri_paged.attend(q, qpos, c, bt, ptab, nsplit=1, bn=bn, warps=w)
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(50):
        f()
    e1.record()
    torch.cuda.synchronize()
    print(f"R={R} ctx={ctx} {'limb' if limb else 'int32'} bn={bn} w={w}: {e0.elapsed_time(e1) / 50 * 1e3:.1f} us/call", flush=True)
