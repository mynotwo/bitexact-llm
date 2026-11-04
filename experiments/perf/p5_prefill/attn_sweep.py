"""tri_prefill.attend timing sweep (bm, bn, warps) at prefill shapes; checks each config equals the first.
python attn_sweep.py [--B 64 --L 256]"""
import argparse
import itertools

import torch

from bitexact.int8.consts import exp_table
from bitexact.vllm_plugin import tri_paged, tri_prefill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--cfg", default="", help="bm,bn,warps; default all")
    a = ap.parse_args()
    HD, HKV, G, bs = 128, 8, 4, 16
    nb = a.B * (a.L // bs + 1) + 8
    g = torch.Generator(device="cuda").manual_seed(0)
    lim = 1 << 12
    ptab = tri_paged.weight_table(exp_table().cuda())
    c = tri_paged.to_limbs(torch.randint(-lim, lim + 1, (nb, HKV, bs, 2 * HD), device="cuda", generator=g).to(torch.int32))
    bt = (torch.arange(a.B * (a.L // bs + 1), device="cuda").view(a.B, -1) + 1).long()
    N = a.B * a.L
    q = torch.randint(-lim, lim + 1, (N, HKV, G, HD), device="cuda", generator=g)
    pos = torch.arange(a.L, device="cuda").repeat(a.B)
    pre = [(r, r * a.L, a.L, a.L) for r in range(a.B)]
    ref = None
    for bm, bn, w in ([tuple(map(int, a.cfg.split(",")))] if a.cfg else itertools.product((32, 64, 128), (32, 64, 128), (4, 8))):
        tiles = tri_prefill.tile_table(pre, G, bm).cuda()
        num = torch.zeros(N, HKV, G, HD, dtype=torch.long, device="cuda")
        den = torch.ones(N, HKV, G, 1, dtype=torch.long, device="cuda")
        f = lambda: tri_prefill.attend(q, pos, c, bt, ptab, tiles, num, den, bm=bm, bn=bn, warps=w)
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(20):
            f()
        e1.record(); torch.cuda.synchronize()
        if ref is None:
            ref = (num.clone(), den.clone())
        print(f"bm {bm} bn {bn} warps {w}: {e0.elapsed_time(e1) / 20:.3f} ms equal={torch.equal(num, ref[0]) and torch.equal(den, ref[1])}", flush=True)


if __name__ == "__main__":
    main()
