"""Config sweep of exact_gemm.FusedLin per (layer kind, M): prints best (BM, BN, BK, warps, stages) in us (graph timing).
  python sweep_gemm.py CONST"""
import itertools
import sys

import torch

from bitexact.exact.model import Lin
from bitexact.vllm_plugin.exact_gemm import FusedLin
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gtime import graph_time  # noqa: E402

c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
for name in ("0.wqkv", "0.wo", "0.wgu", "0.wd", "lm"):
    lin = Lin(c, name, "cuda", "triton", False)
    K = lin.limbs.shape[2]
    for M in (1, 8, 32, 64, 128):
        q = torch.randint(-(1 << 22), 1 << 22, (M, K), device="cuda").to(torch.int32)
        sh = torch.full((M,), -26, device="cuda")
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        res = []
        for BM, BN, BK, nw, ns in itertools.product((16, 32, 64), (16, 32, 64), (64, 128, 256), (4, 8), (2, 3, 4)):
            if BM > max(16, 2 * M) and BM > 16: continue
            try:
                fl = FusedLin(lin, (BM, BN, BK, nw, ns))
                res.append((graph_time(lambda: fl(q, sh, flag), 20), (BM, BN, BK, nw, ns)))
            except Exception:
                pass
        res.sort()
        print(name, M, [(round(t, 1), cfg) for t, cfg in res[:2]], flush=True)
