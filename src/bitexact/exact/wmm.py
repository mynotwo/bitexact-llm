"""Exact products of activations with the original bf16 weights.

T = q @ W^T, q int [M, K] with |q| <= 2^22 (3 signed int8 limbs), W the dense 3-limb part of a weight
(int8 [3, N, K]). |T| <= 2^22 * 8192 * 8355711 < 2^59, so T is one exact int64. Three implementations:
  triton  one kernel: splits q into limbs in registers, 9 int8 tensor-core dots per K step into 5 int32
          accumulators (one per limb weight i + j; |acc| <= 3 * K * 2^14 < 2^31), int64 at the end
  cublas  torch._int_mm on stacked limbs [3M, K] x [K, 3N] (int32 exact), joined in int64
  cpu     float64 GEMM of q's limbs against W (every partial sum < 2^43 < 2^53: exact in any order)
The sparse residual (weight bits below the row grid) adds B = q[:, cols] * vals into the columns, and the
epilogue rounds (T * 2^d + B) * 2^p once (ops.combine, ops.shift_round_hl).
"""

import torch

from bitexact.exact.ops import combine, shift_round_hl
from bitexact.intops import split_limbs, stack_limbs

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if HAVE_TRITON:
    @triton.jit
    def _split(r):
        lo = ((r + 128) & 255) - 128
        return lo.to(tl.int8), (r - lo) >> 8

    @triton.jit
    def _wmm_kernel(a_ptr, w_ptr, c_ptr, M, N, K, KS, sam, sak, swl, swn, scz, scm,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        pz = tl.program_id(2)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        c0 = tl.zeros((BM, BN), dtype=tl.int32)
        c1 = tl.zeros((BM, BN), dtype=tl.int32)
        c2 = tl.zeros((BM, BN), dtype=tl.int32)
        c3 = tl.zeros((BM, BN), dtype=tl.int32)
        c4 = tl.zeros((BM, BN), dtype=tl.int32)
        k_end = tl.minimum(pz * KS + KS, K)
        for k0 in range(pz * KS, k_end, BK):
            ka = k0 + rk
            ra = tl.load(a_ptr + rm[:, None] * sam + ka[None, :] * sak,
                         mask=(rm[:, None] < M) & (ka[None, :] < k_end), other=0)
            bm = (ka[:, None] < k_end) & (rn[None, :] < N)
            bp = w_ptr + rn[None, :] * swn + ka[:, None]
            b0 = tl.load(bp, mask=bm, other=0)
            b1 = tl.load(bp + swl, mask=bm, other=0)
            b2 = tl.load(bp + 2 * swl, mask=bm, other=0)
            a0, ra = _split(ra)
            a1, ra = _split(ra)
            a2, ra = _split(ra)
            c0 = tl.dot(a0, b0, c0, out_dtype=tl.int32)
            c1 = tl.dot(a0, b1, c1, out_dtype=tl.int32)
            c1 = tl.dot(a1, b0, c1, out_dtype=tl.int32)
            c2 = tl.dot(a0, b2, c2, out_dtype=tl.int32)
            c2 = tl.dot(a1, b1, c2, out_dtype=tl.int32)
            c2 = tl.dot(a2, b0, c2, out_dtype=tl.int32)
            c3 = tl.dot(a1, b2, c3, out_dtype=tl.int32)
            c3 = tl.dot(a2, b1, c3, out_dtype=tl.int32)
            c4 = tl.dot(a2, b2, c4, out_dtype=tl.int32)
        tot = c0.to(tl.int64) + (c1.to(tl.int64) << 8) + (c2.to(tl.int64) << 16) \
            + (c3.to(tl.int64) << 24) + (c4.to(tl.int64) << 32)
        tl.store(c_ptr + pz * scz + rm[:, None] * scm + rn[None, :], tot,
                 mask=(rm[:, None] < M) & (rn[None, :] < N))


CFG_SMALL = (16, 64, 128, 4, 3)      # BM, BN, BK, warps, stages: M <= 16 (decode), tuned on H800
CFG_LARGE = (32, 128, 64, 4, 3)      # M > 16 (BM 64 x BN 128 with 8 warps hits a Triton 3.0 layout assertion)
CFG_MID = CFG_LARGE
TARGET_PROGRAMS = 264


def wmm_triton(q, limbs, cfg=None, target=None):
    """q int32 [M, K] (|q| <= 2^22), limbs int8 [3, N, K] -> exact int64 partial sums [split, M, N]
    (split-K slices; their int64 sum is the exact product)."""
    M, K = q.shape
    N = limbs.shape[1]
    if cfg is None:
        cfg = CFG_SMALL if M <= 16 else CFG_LARGE
    BM, BN, BK, nw, ns = cfg
    tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    nk = triton.cdiv(K, BK)
    split = max(1, min(nk, (target or TARGET_PROGRAMS) // tiles))
    ks = triton.cdiv(nk, split) * BK
    split = triton.cdiv(K, ks)
    c = torch.empty((split, M, N), dtype=torch.long, device=q.device)
    _wmm_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN), split)](
        q, limbs, c, M, N, K, ks, q.stride(0), q.stride(1), limbs.stride(0), limbs.stride(1), M * N, N,
        BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
    return c


def wmm_cublas(q, wcat, M_chunk=2048):
    """wcat int8 [3N, K] (limb j rows j*N..), q [M, K] -> exact int64 [M, N] via torch._int_mm."""
    M = q.shape[0]
    N = wcat.shape[0] // 3
    outs = []
    for m0 in range(0, M, M_chunk):
        qq = q[m0:m0 + M_chunk]
        m = qq.shape[0]
        C = torch._int_mm(stack_limbs(qq, 3), wcat.t())[:3 * m].view(3, m, 3, N)
        cs = [None] * 5
        for i in range(3):
            for j in range(3):
                t = C[i, :, j].long()
                cs[i + j] = t if cs[i + j] is None else cs[i + j] + t
        outs.append(cs[0] + (cs[1] << 8) + (cs[2] << 16) + (cs[3] << 24) + (cs[4] << 32))
    return torch.cat(outs) if len(outs) > 1 else outs[0]


def wmm_cpu(q, w64):
    """w64 float64 [K, N] = W (|W| <= 2^23); q split into 3 limbs so every partial sum < 2^43."""
    M = q.shape[0]
    C = (torch.cat(split_limbs(q, 3)[0]).double() @ w64).long()
    return C[:M] + (C[M:2 * M] << 8) + (C[2 * M:] << 16)


def residual(q, rr, rc, rv, N):
    """B [M, N] = sum over residual entries (row n, col k, value v) of q[:, k] * v (exact int64 sums)."""
    B = torch.zeros(q.shape[0], N, dtype=torch.long, device=q.device)
    if rr.numel():
        B.index_add_(1, rr, q[:, rc].long() * rv)
    return B


def epilogue(T, B, d, g, sh, FR):
    """round((T * 2^d + B) * 2^(sh + g - d + FR)) per element: the linear output at FR bits."""
    H, L = combine(T, d, B)
    return shift_round_hl(H, L, sh[:, None] + (g - d)[None, :] + FR)
