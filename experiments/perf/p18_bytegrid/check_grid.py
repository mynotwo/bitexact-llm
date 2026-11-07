"""p18 self-check: block_arrays() aligned-grid counts vs a plain-Python brute force on a small random matrix.

Also checks the closed form used in RESULTS.md: zero residual on the m-aligned grid at capacity C holds iff
top - m*floor(bot/m) <= C. Run on the host with the shipped src on PYTHONPATH.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as an  # noqa: E402
from bitexact.exact.consts import bf16_parts  # noqa: E402

torch.manual_seed(0)
N, K = 24, 96
W = (torch.randn(N, K) * torch.exp2(torch.randint(-12, 4, (N, K)).float())).to(torch.bfloat16)
W[0, :16] = 0
sm, ex = bf16_parts(W)
sm = sm.numpy()
ex = ex.numpy()


def bitlen(x):
    return int(x).bit_length()


def lowbit(x):
    return (int(x) & -int(x)).bit_length() - 1


fails = 0
for B in (0, 32, 16, 8):
    bs = K if B == 0 else B
    A, bsz, bmh = an.block_arrays(W, B)
    nb = K // bs
    brute = {k: [] for k in A}
    bm = np.zeros(8, dtype=np.int64)
    for r in range(N):
        for b in range(nb):
            idx = [(r, c) for c in range(b * bs, (b + 1) * bs) if sm[r, c] != 0]
            if not idx:
                for k in A:
                    if k == "nnz":
                        brute[k].append(0)
                    elif k == "span":
                        brute[k].append(-1)
                    else:
                        brute[k].append(0)
                continue
            tops = [ex[i, j] + bitlen(abs(sm[i, j])) for i, j in idx]
            lows = [ex[i, j] + lowbit(abs(sm[i, j])) for i, j in idx]
            top, bot = max(tops), min(lows)
            bm[bot % 8] += 1
            brute["nnz"].append(len(idx))
            brute["span"].append(top - bot)
            for c, C in an.CAPS.items():
                brute[f"r{c}"].append(sum(1 for l in lows if l < top - C))
                for m in an.ALIGNS:
                    g = m * (-(-(top - C) // m))          # smallest multiple of m >= top - C
                    brute[f"g{m}_r{c}"].append(sum(1 for l in lows if l < g))
            # closed form for zero residual on the 8-aligned grid
            for c, C in an.CAPS.items():
                zero8 = brute[f"g8_r{c}"][-1] == 0
                if zero8 != (top - 8 * (bot // 8) <= C):
                    fails += 1
                    print("closed-form mismatch", B, r, b, c)
    for k in A:
        if not np.array_equal(np.asarray(brute[k], dtype=np.int64), A[k]):
            fails += 1
            print("MISMATCH", B, k)
    if not np.array_equal(bm, bmh):
        fails += 1
        print("MISMATCH botmod8", B, bm.tolist(), bmh.tolist())
    print("block", B, "checked", len(brute["nnz"]), "blocks")
print("FAILS", fails)
