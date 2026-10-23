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
from bitexact.intops import bitlen, isqrt, rdiv, rsh

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
    sh = bitlen(amax) - 30
    as_ = rsh(amax, sh)
    zs = rsh(z, sh[:, None])
    return rdiv(zs * qmax(nl), as_[:, None]), as_, sh


def norm24(mant, e):
    """(mant, e) with value mant * 2^e -> same value with mant in [2^23, 2^24] (rounded)."""
    b = bitlen(mant) - 24
    return rsh(mant, b), e + b


def rms(x):
    """RMS of each row of x (FR bits) as (r, shx): rms value = r * 2^(shx - FR), r has >= 17 bits."""
    d = x.shape[-1]
    amax = x.abs().amax(-1).clamp(min=1)
    shx = bitlen(amax) - 24
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
