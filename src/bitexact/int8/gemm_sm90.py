"""Hopper (sm90) int8 GEMM: C[M, N] int32 = A[M, K] int8 @ W[N, K]^T int8, persistent wgmma + TMA.

torch._int_mm dispatches to `cutlass_80_tensorop_i16832gemm_s8_*` kernels (Ampere mma.sync), which leave ~35%
of the H800 int8 peak unused; this kernel uses Hopper wgmma with TMA loads and reaches 1.1-1.7x its TOPS on the
shapes in TABLE. Same integers as torch._int_mm either way: integer sums are exact and order independent
while no output overflows int32 (K * 127 * 128 < 2^31), so only the speed changes.

TMA descriptors give the bounds handling: A / W tiles past M / N read as zeros and the C tile store is
clipped, so the loops carry no masks. The grid is min(tiles, NSMS) programs; each walks its tiles in
groups of GROUP_M row tiles sharing an N tile (L2 reuse of the weight tile).

cfg = (BM, BN, BK, GROUP_M, num_warps, num_stages). Configs from experiments/perf/p6_wgmma/sweep4.py
(CUDA-graph timing, HBM-cold rotating weights, H800); every config is `torch.equal` to torch._int_mm on
random int8 and on all -128 / all 127 inputs (tests/test_gemm_sm90.py).
"""
import functools

import torch
import triton
import triton.language as tl

NSMS = 132                                          # H800: one persistent program per SM
_Min = 1 << 30


@triton.jit
def _wgmma(a_ptr, w_ptr, c_ptr, M, N, K,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, NSM: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BM, BK])
    w_desc = tl.make_tensor_descriptor(w_ptr, [N, K], [K, 1], [BN, BK])
    c_desc = tl.make_tensor_descriptor(c_ptr, [M, N], [N, 1], [BM, BN])
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    total = nm * nn
    pid = tl.program_id(0)
    for t in tl.range(pid, total, NSM, flatten=True):
        gsz = min(nm - (t // (GROUP_M * nn)) * GROUP_M, GROUP_M)
        pm = (t // (GROUP_M * nn)) * GROUP_M + (t % (GROUP_M * nn)) % gsz
        pn = (t % (GROUP_M * nn)) // gsz
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        for k in tl.range(0, K, BK):
            acc = tl.dot(a_desc.load([pm * BM, k]), tl.trans(w_desc.load([pn * BN, k])), acc, out_dtype=tl.int32)
        c_desc.store([pm * BM, pn * BN], acc)


def _alloc(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


def _setup():
    if not getattr(tl, "_bx_alloc", False):         # TMA descriptors built in the kernel need a scratch allocator
        triton.set_allocator(_alloc)
        tl._bx_alloc = True


def gemm(a, w, cfg):
    """Exact int32 a @ w.t() with cfg = (BM, BN, BK, GROUP_M, num_warps, num_stages). Ragged shapes are fine."""
    BM, BN, BK, G, nw, ns = cfg
    _setup()
    M, K = a.shape
    N = w.shape[0]
    c = torch.empty((M, N), dtype=torch.int32, device=a.device)
    _wgmma[(min(triton.cdiv(M, BM) * triton.cdiv(N, BN), NSMS),)](
        a, w, c, M, N, K, BM, BN, BK, G, NSMS, num_warps=nw, num_stages=ns)
    return c


# (N, K) -> [(max_M, cfg)] with None = "stay on the gemm_skinny path". Buckets come from the sweep in
# experiments/perf/p6_wgmma/sweep4.py (logs /data41/audreymao/perf6_20261105/wgmma/sw4_*.log, summarised by
# mk_table.py): each bucket's cfg is >= 8% faster than the old path at every measured row count in the bucket,
# with CUDA-graph timing and HBM-cold weights. Row counts are 2 x batch for decode (plugin limb buffers) and
# tokens for prefill, which is why the buckets are cut where they are.
# Shapes with no entry here lose at every row count (2048x2048, 3072x2048) and stay entirely on the old path.
TABLE = {
    # 1B: 3072 qkv / 2048 o,down / 16384 gate_up / 128256 LM head
    (16384, 2048): [(256, None), (512, (128, 128, 128, 8, 8, 5)), (_Min, (128, 128, 128, 16, 8, 5))],
    (2048, 8192): [(256, None), (384, (64, 128, 256, 4, 8, 3)), (512, (64, 128, 128, 4, 8, 5)),
                    (1024, (128, 128, 128, 4, 8, 5)), (1536, (64, 128, 256, 8, 8, 3)), (_Min, None)],
    (128256, 2048): [(256, (128, 128, 128, 16, 8, 5)), (1536, (128, 128, 128, 8, 8, 5)),
                      (_Min, (128, 128, 128, 16, 8, 5))],
    # Llama-3.1-8B: 6144 qkv / 4096 o,down / 28672 gate_up / 128256 LM head
    (6144, 4096): [(512, None), (1536, (128, 128, 128, 8, 8, 5)), (_Min, (128, 128, 128, 16, 8, 5))],
    (4096, 4096): [(256, None), (384, (128, 128, 128, 4, 8, 5)), (512, (128, 128, 128, 8, 8, 4)), (768, None),
                    (1024, (128, 128, 128, 16, 8, 5)), (2048, (128, 128, 128, 4, 8, 5)),
                    (_Min, (128, 128, 128, 16, 8, 5))],
    (28672, 4096): [(128, None), (512, (128, 128, 128, 4, 8, 5)), (1024, (128, 128, 128, 8, 8, 4)),
                     (1536, (128, 128, 128, 16, 8, 4)), (_Min, (128, 128, 128, 16, 8, 5))],
    (4096, 14336): [(256, None), (512, (128, 128, 128, 4, 8, 5)), (1024, (128, 128, 128, 16, 8, 5)),
                     (2048, (128, 128, 128, 8, 8, 5)), (8192, (128, 256, 64, 16, 8, 4)),
                     (_Min, (128, 128, 128, 4, 8, 4))],
    (128256, 4096): [(128, None), (512, (128, 128, 128, 16, 8, 5)), (_Min, (256, 128, 64, 4, 8, 4))],
}


def pick(N, K, M):
    """cfg for this (N, K, M), or None when the old path is at least as fast."""
    for mx, cfg in TABLE.get((N, K), ()):
        if M <= mx:
            return cfg
    return None


@functools.lru_cache(maxsize=None)
def sm90():
    """Hopper or newer: wgmma + TMA. Queried on first dispatch, not at import (importing must not initialise CUDA)."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def gemm_dispatch(a, w):
    """int8 GEMM with this module's own dispatch (benchmarks call this); serve goes through gemm_skinny.int_mm."""
    cfg = pick(w.shape[0], w.shape[1], a.shape[0]) if sm90() else None
    return gemm(a, w, cfg) if cfg is not None else torch._int_mm(a, w.t())