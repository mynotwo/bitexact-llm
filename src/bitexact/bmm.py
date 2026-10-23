"""Exact batched integer matrix product for attention (q.k and p.v), shared by both engines.

bmm(a, b, na, nb): a [Z, M, K], b [Z, K, N] integer tensors (any strides, int32 or int64 values), every
|a| and |b| within na / nb signed base-256 limbs (3 limbs: |x| <= 8355711). Returns the exact int64
product. On CUDA one Triton kernel splits each tile into int8 limbs and multiplies them on the int8
tensor cores with int32 accumulators, one per limb weight s = i + j (|sum| <= 3 * K * 2^14 < 2^31 for
K < 43690); the int64 result is formed once at the end. On CPU: float64 GEMM, exact while every partial
sum stays below 2^53 (the caller checks). Both give the same integers.
"""

import torch

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
    def _bmm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, saz, sam, sak, sbz, sbk, sbn, scz, scm, scn,
                    NA: tl.constexpr, NB: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        pz = tl.program_id(2).to(tl.int64)
        a_ptr += pz * saz
        b_ptr += pz * sbz
        c_ptr += pz * scz
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        c0 = tl.zeros((BM, BN), dtype=tl.int32)
        c1 = tl.zeros((BM, BN), dtype=tl.int32)
        c2 = tl.zeros((BM, BN), dtype=tl.int32)
        c3 = tl.zeros((BM, BN), dtype=tl.int32)
        c4 = tl.zeros((BM, BN), dtype=tl.int32)
        for k0 in range(0, K, BK):
            ka = k0 + rk
            ra = tl.load(a_ptr + rm[:, None] * sam + ka[None, :] * sak,
                         mask=(rm[:, None] < M) & (ka[None, :] < K), other=0).to(tl.int32)
            rb = tl.load(b_ptr + ka[:, None] * sbk + rn[None, :] * sbn,
                         mask=(ka[:, None] < K) & (rn[None, :] < N), other=0).to(tl.int32)
            a0, ra = _split(ra)
            b0, rb = _split(rb)
            c0 += tl.dot(a0, b0, out_dtype=tl.int32)
            if NA > 1:
                a1, ra = _split(ra)
                c1 += tl.dot(a1, b0, out_dtype=tl.int32)
            if NB > 1:
                b1, rb = _split(rb)
                c1 += tl.dot(a0, b1, out_dtype=tl.int32)
            if NA > 1 and NB > 1:
                c2 += tl.dot(a1, b1, out_dtype=tl.int32)
            if NA > 2:
                a2, ra = _split(ra)
                c2 += tl.dot(a2, b0, out_dtype=tl.int32)
                if NB > 1:
                    c3 += tl.dot(a2, b1, out_dtype=tl.int32)
            if NB > 2:
                b2, rb = _split(rb)
                c2 += tl.dot(a0, b2, out_dtype=tl.int32)
                if NA > 1:
                    c3 += tl.dot(a1, b2, out_dtype=tl.int32)
                if NA > 2:
                    c4 += tl.dot(a2, b2, out_dtype=tl.int32)
        tot = c0.to(tl.int64) + (c1.to(tl.int64) << 8) + (c2.to(tl.int64) << 16) \
            + (c3.to(tl.int64) << 24) + (c4.to(tl.int64) << 32)
        tl.store(c_ptr + rm[:, None] * scm + rn[None, :] * scn, tot, mask=(rm[:, None] < M) & (rn[None, :] < N))


def bmm(a, b, na=3, nb=3):
    """Exact a @ b for [Z, M, K] x [Z, K, N] integer tensors -> int64 [Z, M, N]."""
    Z, M, K = a.shape
    N = b.shape[2]
    if not (a.is_cuda and HAVE_TRITON):
        return (a.double() @ b.double()).long()
    assert K < 43690
    c = torch.empty((Z, M, N), dtype=torch.long, device=a.device)
    BM = 16 if M <= 16 else 64
    BN = 64 if N <= 64 else 128
    BK = 64
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN), Z)
    with torch.cuda.device(a.device):
        _bmm_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), a.stride(2), b.stride(0), b.stride(1),
                          b.stride(2), c.stride(0), c.stride(1), c.stride(2), NA=na, NB=nb, BM=BM, BN=BN,
                          BK=BK, num_warps=4)
    return c
