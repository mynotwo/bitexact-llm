"""Skinny-M int8 GEMM for decode: C[R, N] int32 = A[R, K] int8 @ W[N, K]^T int8, int32 accumulation.

Integer sums are exact and order-independent while no output overflows int32 (K * 127 * 128 < 2^31 for
K <= 132,000), so any tiling / split-K gives the same integers as torch._int_mm. Configs are fixed per
(R, N, K) by a static table (no run-time timing), so serve-time behaviour is deterministic.
"""
import torch
import triton
import triton.language as tl

from bitexact.int8.gemm_lt import int_mm as lt_int_mm


@triton.jit
def _k(a_ptr, w_ptr, c_ptr, R, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
       SPLIT: tl.constexpr, KPER: tl.constexpr):
    pn = tl.program_id(0)
    pm = tl.program_id(1)
    pk = tl.program_id(2)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    k0 = pk * KPER
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k in range(k0, k0 + KPER, BK):
        a = tl.load(a_ptr + rm[:, None] * K + (k + rk)[None, :], mask=rm[:, None] < R, other=0)
        w = tl.load(w_ptr + rn[None, :] * K + (k + rk)[:, None], mask=rn[None, :] < N, other=0)
        acc = tl.dot(a, w, acc, out_dtype=tl.int32)
    cp = c_ptr + rm[:, None] * N + rn[None, :]
    m = (rm[:, None] < R) & (rn[None, :] < N)
    if SPLIT == 1:
        tl.store(cp, acc, mask=m)
    else:
        tl.atomic_add(cp, acc, mask=m, sem="relaxed")


def gemm(a, w, cfg):
    """cfg = (BM, BN, BK, SPLIT, warps, stages). K must divide by BK * SPLIT."""
    BM, BN, BK, S, nw, ns = cfg
    R, K = a.shape
    N = w.shape[0]
    assert K % (BK * S) == 0 and K * 127 * 128 < 1 << 31
    c = torch.zeros((R, N), dtype=torch.int32, device=a.device) if S > 1 else \
        torch.empty((R, N), dtype=torch.int32, device=a.device)
    grid = (triton.cdiv(N, BN), triton.cdiv(R, BM), S)
    _k[grid](a, w, c, R, N, K, BM, BN, BK, S, K // S, num_warps=nw, num_stages=ns)
    return c


# (N, K) -> [(max_rows, cfg)] chosen offline (experiments/perf/p2_gemm/search_graph.py, H800, CUDA-graph timing, cold weights).
# Shapes / row buckets not listed: rows > 256 go to gemm_lt (cuBLASLt), the rest (decode LM head) to torch._int_mm.
TABLE = {
    (3072, 2048): [(24, (32, 32, 256, 1, 4, 4)), (64, (32, 64, 256, 1, 8, 4)), (256, (64, 64, 128, 1, 4, 5))],
    (2048, 2048): [(24, (16, 32, 256, 1, 4, 5)), (64, (32, 32, 256, 1, 4, 5)), (256, (64, 64, 256, 1, 4, 4))],
    (16384, 2048): [(256, (128, 128, 128, 1, 4, 3))],
    (2048, 8192): [(24, (32, 32, 256, 4, 4, 4)), (64, (32, 32, 256, 1, 4, 5)), (256, (64, 64, 256, 1, 4, 5))],
}


def int_mm(a, w, lt=True):
    """Exact int32 a @ w.t(); skinny Triton kernel where the table has a faster config; else, above 256 rows and with
    lt, cuBLASLt with a workspace (gemm_lt; BITEXACT_GEMM=intmm: torch._int_mm); else torch._int_mm. All give the
    same integers."""
    rows = TABLE.get((w.shape[0], w.shape[1]))
    if rows is not None and a.shape[0] <= 256 and a.is_contiguous() and w.is_contiguous():
        for mx, cfg in rows:
            if a.shape[0] <= mx:
                return gemm(a, w, cfg)
    if lt and a.shape[0] > 256:     # scorer / prefill rows: cuBLASLt's sm90 kernels (decode rows: measured slower,
        return lt_int_mm(a, w)      # experiments/speed_on_main/RESULTS.md)
    return torch._int_mm(a, w.t())
