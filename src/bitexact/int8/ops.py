"""Integer primitives of the int8 engine.

All tensors are int64 fixed point unless noted. Every function is a fixed integer rule, so its result does
not depend on device, batch size, row order or summation order:
  - additions, products, shifts and floor divisions of int64 are exact (callers keep values < 2^62);
  - reductions (sum, max) of int64 are exact in any order;
  - matrix products are exact integer sums: int8 limbs on CUDA tensor cores with int32 accumulation
    (|partial sums| < 2^31), or float64 GEMM with every partial sum < 2^53 (then any order is exact).
"""

import torch

from bitexact.int8.consts import EXP_MIN, FA, FR, SILU_LIM
from bitexact.intops import isqrt, rdiv, rsh

EXACT = 1 << 53
EPS40 = 10995116         # round(1e-5 * 2^40): RMSNorm eps


def qmax(nl):
    return 127 << (8 * (nl - 1))


def quant_rows(z, nl):
    """Per-row quantization of int64 z [M, K] to |q| <= qmax(nl).

    amax is first brought to 30 bits by a per-row shift sh (as_ = rsh(amax, sh) in [2^29, 2^30]), then
    q = round(rsh(z, sh) * qmax / as_). Row value = q / qmax * as_ * 2^sh (in z's units).
    """
    amax = z.abs().amax(-1).clamp(min=1)
    sh = bitlen32(amax) - 30
    as_ = rsh(amax, sh)
    zs = rsh(z, sh[:, None])
    div = rdiv32 if qmax(nl) < 1 << 15 else rdiv          # |q| <= qmax + 1/2 < 2^16 (see rdiv32)
    return div(zs * qmax(nl), as_[:, None]), as_, sh


def bitlen32(x):
    """intops.bitlen without float64 (slow on H800), for int64 x >= 0. Conversion to float32 is monotone and
    2^(k-1), 2^k are exact float32 values, so x in [2^(k-1), 2^k) becomes a float32 in [2^(k-1), 2^k]: frexp
    gives k or k + 1, and the integer comparison removes the + 1. Same integers as bitlen."""
    b = torch.frexp(x.float()).exponent.long()
    up = (torch.ones_like(b) << (b - 1).clamp(min=0)) > x
    return torch.where(x > 0, b - up.long(), torch.zeros_like(b))


def rdiv32(a, b):
    """rdiv(a, b) = floor(n / d), n = 2a + b, d = 2b, without int64 division, when |n / d| < 2^16 and
    |n|, |d| < 2^46. The float32 quotient of the float32-rounded n and d is within a relative 2^-20 of n / d
    (three roundings of 2^-24, or a division of a few ulps), so its floor is off by at most one (absolute
    error < 2^-4); the two exact int64 comparisons (|q * d| < 2^63) correct it. Same integers as rdiv."""
    n, d = 2 * a + b, 2 * b
    q = torch.floor(n.float() / d.float()).long()
    q = torch.where(q * d > n, q - 1, q)
    return torch.where((q + 1) * d <= n, q + 1, q)


def post_bound(w, m, fmsh, nl):
    """Largest |t| = |rsh(acc * m_c, fmsh)| over every input of a linear layer (Python int).

    The GEMM input q comes from quant_rows, so |q| <= qmax(nl) and |acc_c| <= qmax(nl) * sum_k |W_ck|
    (exact integer product). rsh rounds half up and is monotone, and |rsh(x, s)| <= rsh(|x|, s), so
    |t_c| <= rsh(qmax(nl) * l1_c * |m_c|, fmsh). Taken over all output channels c."""
    l1 = w.long().abs().sum(1)
    assert int(l1.max()) < 1 << 24 and int(m.abs().max()) < 1 << 36       # l1 * |m| < 2^60 in int64
    a = qmax(nl) * int((l1 * m.abs()).max())
    return (a + (1 << (fmsh - 1))) >> fmsh if fmsh > 0 else a << -fmsh


def norm24(mant, e):
    """(mant, e) with value mant * 2^e -> same value with mant in [2^23, 2^24] (rounded)."""
    b = bitlen32(mant) - 24
    return rsh(mant, b), e + b


def rms(x):
    """RMS of each row of x (FR bits) as (r, shx): rms value = r * 2^(shx - FR), r has >= 17 bits."""
    d = x.shape[-1]
    amax = x.abs().amax(-1).clamp(min=1)
    shx = bitlen32(amax) - 24
    xs = rsh(x, shx[:, None])                       # |xs| <= 2^24
    ms = (xs * xs).sum(-1) // d                     # < 2^48
    eps = rsh(torch.full_like(shx, EPS40), 40 - 2 * (FR - shx))
    return isqrt(ms + eps), shx


def imm(q, w64):
    """Exact q @ W^T on CPU: q int64 [M, K] with |q| <= qmax(3), w64 = W^T as float64 [K, N] with |W| <= 127
    (|partial sums| <= 2^23 * 127 * 8192 < 2^53, so the float64 GEMM is exact in any order)."""
    return (q.double() @ w64).long()


def exp_of(z, lut):
    """e^z at FE bits for z <= 0 at FA bits; below EXP_MIN gives 0."""
    idx = z - (EXP_MIN << FA)
    val = lut[idx.clamp(min=0)]
    return torch.where(idx < 0, torch.zeros_like(val), val)


def silu(g, lut):
    """SiLU at FR bits from the table at step 2^-FA, linear interpolation between entries."""
    lo, hi = -SILU_LIM << FA, SILU_LIM << FA
    gi = g >> (FR - FA)                              # floor
    r = g - (gi << (FR - FA))
    i = gi.clamp(lo, hi - 1) - lo
    a = lut[i]
    v = a + rsh((lut[i + 1] - a) * r, FR - FA)
    return torch.where(gi >= hi, g, torch.where(gi < lo, torch.zeros_like(g), v))
