"""Microbench: exact-mode linear, old path (wmm_triton + segs.lin_y, compiled) vs FusedLin, CUDA-graph timing.
  python bench_gemm.py CONST [--layer 0]"""
import argparse
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gtime import graph_time  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("const")
ap.add_argument("--rows", default="1,8,32,64,128")
a = ap.parse_args()
from bitexact.exact.model import Lin
from bitexact.exact.segs import lin_y
from bitexact.vllm_plugin.exact_gemm import FusedLin

c = torch.load(a.const, map_location="cpu", weights_only=False)
yc = torch.compile(lin_y, dynamic=None)


for name in ("0.wqkv", "0.wo", "0.wgu", "0.wd", "lm"):
    R = 0
    lin_e = Lin(c, name, "cuda", "triton", True, 0 if name == "lm" else max(
        int(torch.bincount(c[f"{i}.{name[2:]}.rr"]).max()) if c[f"{i}.{name[2:]}.rr"].numel() else 0 for i in range(16)))
    lin_p = Lin(c, name, "cuda", "triton", False)
    fl = FusedLin(lin_p)
    K = lin_p.limbs.shape[2]
    for M in map(int, a.rows.split(",")):
        q = torch.randint(-(1 << 22), 1 << 22, (M, K), device="cuda").to(torch.int32)
        sh = torch.full((M,), -26, device="cuda")
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        old = graph_time(lambda: yc(*lin_e(q), sh, True))
        new = graph_time(lambda: fl(q, sh, flag))
        print(f"{name:7s} M={M:4d} old {old:8.1f} us  fused {new:8.1f} us  x{old / new:.2f}", flush=True)
