"""Integer rules shared by both engines.

Every function is a fixed integer rule on int64 tensors, so its result does not depend on device, batch
size, row order or summation order: additions, products, shifts and floor divisions of int64 are exact
(callers keep values < 2^62), and reductions (sum, max) of int64 are exact in any order. Rounding is
round-half-up throughout: round(v) = floor(v + 1/2).
"""

import torch

LIM = {1: 127, 2: 32639, 3: 8355711}          # n signed base-256 digits reach 127 * (256^n - 1) / 255


def bitlen(x):
    """Number of bits of int64 x >= 0 (0 for 0), elementwise, exact.

    frexp of the float64 value gives it up to rounding; float64 rounding can only lift x to the next
    power of two, which the integer comparison undoes."""
    b = torch.frexp(x.double()).exponent.long()
    up = (torch.ones_like(b) << (b - 1).clamp(min=0)) > x
    return torch.where(x > 0, b - up.long(), torch.zeros_like(b))


def rsh(x, s):
    """round(x / 2^s), ties up, for s > 0; x * 2^-s for s <= 0. s is an int or an int64 tensor."""
    if not torch.is_tensor(s):
        if s > 0:
            return (x + (1 << (s - 1))) >> s
        return x << (-s)
    pos = s.clamp(min=0)
    neg = (-s).clamp(min=0)
    half = torch.where(pos > 0, torch.ones_like(pos) << (pos - 1).clamp(min=0), torch.zeros_like(pos))
    return ((x + half) >> pos) << neg


def fdiv(a, b):
    """floor(a / b) for int64 a, b > 0."""
    return torch.div(a, b, rounding_mode="floor")


def rdiv(a, b):
    """round(a / b), ties up, for int64 a and b > 0 (b may broadcast)."""
    return fdiv(2 * a + b, 2 * b)


def isqrt(x):
    """floor(sqrt(x)) for int64 x >= 0, exact (float64 estimate, then integer corrections)."""
    s = torch.sqrt(x.double()).floor().long()
    for _ in range(2):
        s = torch.where(s * s > x, s - 1, s)
        s = torch.where((s + 1) * (s + 1) <= x, s + 1, s)
    return s


def first_argmax(v):
    """Index of the first maximum of each row (explicit, so ties never depend on the backend)."""
    m = v.amax(-1, keepdim=True)
    col = torch.arange(v.shape[-1], device=v.device).expand_as(v)
    return torch.where(v == m, col, torch.full_like(col, v.shape[-1])).amin(-1)


def split_limbs(x, n):
    """Signed base-256 digits of integer x, least significant first: (n int64 tensors in [-128, 127],
    remainder). The remainder is 0 iff |x| fits n digits."""
    out, r = [], x.long()
    for _ in range(n):
        lo = ((r + 128) & 255) - 128
        out.append(lo)
        r = (r - lo) >> 8
    return out, r


def stack_limbs(q, n):
    """q [M, K] within n limbs -> int8 [n M padded, K], limbs stacked along rows and padded to >= 24 rows,
    a multiple of 8 (torch._int_mm's shape rule)."""
    a = torch.cat([lo.to(torch.int8) for lo in split_limbs(q, n)[0]])
    rows = a.shape[0]
    pad = max(24, (rows + 7) // 8 * 8) - rows
    return torch.cat([a, a.new_zeros(pad, a.shape[1])]) if pad else a


def join_limbs(c, M, n):
    """Rows [i M, (i + 1) M) of c hold the product of limb i: sum_i c_i 2^(8 i) as int64 [M, N]."""
    acc = c[:M].long()
    for i in range(1, n):
        acc = acc + (c[i * M:(i + 1) * M].long() << (8 * i))
    return acc
