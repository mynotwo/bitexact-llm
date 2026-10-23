"""Integer sampling: greedy first-argmax, or Gumbel-max with integer Gumbel noise.

The noise for generated token t of a request with seed s is a pure function of (SEED_BASE, s, t): the raw
64-bit words of numpy's PCG64 stream np.random.default_rng([SEED_BASE, s, t]) (one word per vocabulary
entry), turned into -ln(-ln u) with integer arithmetic only (ln from a table plus linear interpolation,
absolute error ~2^-35 before the final rounding). Same key, same noise, on any device and in any batch.
Temperature is a rational T = tn / td.
"""

from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np
import torch

from bitexact.intops import bitlen, first_argmax, rdiv, rsh
from bitexact.llama import V

SEED_BASE = 20261022
FL = 40                      # fractional bits of the ln table
LN2Q = 762123384786          # round(ln 2 * 2^40)
UMIN = 9008                  # u is clipped to [1e-12, 1 - 1e-12]: 1e-12 * 2^53 ~ 9007.2

_POOL = None


def _raw(key):
    return np.random.default_rng([SEED_BASE, int(key[0]), int(key[1])]).bit_generator.random_raw(V)


def raw_rows(keys):
    """[(seed, t)] -> int64 tensor [n, V] of PCG64 raw words (CPU)."""
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(16)
    rows = list(_POOL.map(_raw, keys))
    return torch.from_numpy(np.stack(rows).view(np.int64)) if rows else torch.zeros(0, V, dtype=torch.long)


def ln_q(n, lnt):
    """ln(n) at FL bits for int64 n >= 1 (table + linear interpolation)."""
    k = bitlen(n) - 1
    m = torch.where(k <= 52, n << (52 - k).clamp(min=0), n >> (k - 52).clamp(min=0))   # [2^52, 2^53)
    idx = (m >> 36) - (1 << 16)
    r = (m >> 16) & ((1 << 20) - 1)
    a = lnt[idx]
    return k * LN2Q + a + (((lnt[idx + 1] - a) * r) >> 20)


def gumbel(raw, lnt, bits):
    """-ln(-ln u) at `bits` fractional bits, u = (raw >> 11) / 2^53 clipped to [UMIN, 2^53 - UMIN] / 2^53."""
    U = ((raw >> 11) & ((1 << 53) - 1)).clamp(UMIN, (1 << 53) - UMIN)
    Dd = (1 << 53) - U                                      # 1 - u = Dd / 2^53
    A = 53 * LN2Q - ln_q(U, lnt)                            # a = -ln u at FL
    lna_main = ln_q(A.clamp(min=1), lnt) - FL * LN2Q
    dq = Dd >> 13                                           # d = 1 - u at FL (< 2^32 when used)
    lna_ser = ln_q(Dd, lnt) - 53 * LN2Q + (dq >> 1) + (5 * (dq >> 8) ** 2) // (24 << 24)
    lna = torch.where(Dd < (1 << 45), lna_ser, lna_main)    # a = d (1 + d/2 + d^2/3 ...) for small d
    return rsh(-lna, FL - bits)


def sample(logits, tn, td, g=None):
    """Greedy (tn = 0) or Gumbel-max at T = tn / td: first argmax of round(l * td / tn) + g, with logits l
    and noise g at the same fractional bits."""
    if tn == 0:
        return first_argmax(logits)
    return first_argmax(rdiv(logits * td, torch.full_like(logits[:, :1], tn)) + g)


def temp_ratio(t):
    """Float temperature -> (tn, td) with td <= 100; T < 1e-5 is greedy (0, 1)."""
    if t is None or t < 1e-5:
        return 0, 1
    f = Fraction(t).limit_denominator(100)
    return f.numerator, f.denominator
