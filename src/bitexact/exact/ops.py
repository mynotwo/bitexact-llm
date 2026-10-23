"""Integer primitives of the exact-bf16 engine.

All tensors are int64 unless noted. Every function is a fixed integer rule (shifts, exact int64 sums and
products kept below 2^63, floor division), so its value does not depend on device, batch or order.
Rounding is always round-half-up: round(v) = floor(v + 1/2).
"""

import torch

from bitexact.exact.consts import EXP2_BITS, EXP2_ONE, FR
from bitexact.intops import bitlen, fdiv, isqrt, rdiv

QB = 22                       # activation rows: |q| <= 2^22 (3 int8 limbs)
FN = 40                       # normalized activations before quantization
FT2 = 26                      # exponent argument t of 2^-t
FZ = 26                       # softmax z = score - max
L2E28 = 387270501             # round(log2(e) * 2^28)
L2E30 = 1549082005            # round(log2(e) * 2^30)
EPS60 = 11529215046068        # round(1e-5 * 2^60)
NEG_BIG = -(1 << 20)


def rshc(x, s):
    """round(x / 2^s) for tensor or int s (s <= 0: x * 2^-s, caller keeps it < 2^62). Shifts are clamped to
    [-62, 62]; for |x| < 2^61 a right shift of 62 already gives the exact result 0."""
    if not torch.is_tensor(s):                                  # python int: no device tensor (graph-safe)
        s = max(-62, min(62, s))
        return ((x + (1 << (s - 1))) >> s) if s > 0 else x << (-s)
    s = s.clamp(-62, 62)
    pos = s.clamp(min=0)
    half = torch.where(pos > 0, torch.ones_like(pos) << (pos - 1).clamp(min=0), torch.zeros_like(pos))
    return ((x + half) >> pos) << (-s).clamp(min=0)


def div_round(a, p, r, chunk=28):
    """round(a * 2^p / r), exact, for |a| < 2^62, 0 < r < 2^34, p tensor/int (-inf < p <= 3 * chunk).

    Long division: a / r = Q + R / r; p >= 0 shifts R in chunks of 28 bits (R * 2^28 < 2^62). Quotients use
    fdiv, so r >= 2^11 is required (callers: rms r >= 2^30, softmax sums >= 2^22)."""
    if not torch.is_tensor(p):
        p = torch.full_like(a, p)
    Q = fdiv(a, r)
    R = a - Q * r
    left = p.clamp(min=0)
    for _ in range(3):
        c = left.clamp(max=chunk)
        X = R << c
        q1 = fdiv(X, r)
        Q = (Q << c) + q1
        R = X - q1 * r
        left = left - c
    up = Q + (2 * R >= r).long()
    k = (-p).clamp(min=1, max=62)
    down = (Q + (torch.ones_like(k) << (k - 1))) >> k
    return torch.where(p >= 0, up, down)


def combine(T, d, B):
    """(H, L) with H * 2^32 + L = T * 2^d + B, 0 <= L < 2^32 (T, B int64, 0 <= d <= 30 per column)."""
    s = 32 - d
    H = T >> s
    L = (T & ((torch.ones_like(s) << s) - 1)) << d
    L = L + B
    return H + (L >> 32), L & 0xFFFFFFFF


def shift_round_hl(H, L, p):
    """round((H * 2^32 + L) * 2^p), 0 <= L < 2^32; caller keeps the result below 2^62."""
    k = (-p).clamp(min=0)
    pp = p.clamp(min=0, max=30)
    up = (H << (32 + pp)) + (L << pp)
    k1 = k.clamp(min=1, max=31)
    small = (H << (32 - k1)) + ((L + (torch.ones_like(k1) << (k1 - 1))) >> k1)       # 1 <= k < 32
    k2 = k.clamp(min=32, max=94)
    carry = torch.where(k2 == 32, (L + (1 << 31)) >> 32, torch.ones_like(L) << (k2 - 33).clamp(min=0))
    big = (H + carry) >> (k2 - 32).clamp(max=62)                                        # k >= 32
    return torch.where(p >= 0, up, torch.where(k < 32, small, big))


def quant(P, eps):
    """Per-row power-of-two quantization of the values P * 2^eps (P int64 [M, K], eps int or tensor
    broadcastable to P). Returns q with |q| <= 2^22 and sh [M]: value ~ q * 2^sh (one rounding)."""
    if not torch.is_tensor(eps):
        eps = torch.full_like(P[:1], eps)
    e = torch.where(P != 0, bitlen(P.abs()) + eps, torch.full_like(P, NEG_BIG))
    sh = (e.amax(-1) - QB).clamp(min=NEG_BIG)
    return rshc(P, sh[..., None] - eps), sh


def rms(x):
    """sqrt(mean(x^2) + 1e-5) of each row of x (int64 [M, D] at FR) as (r, er): r * 2^er, 2^30 <= r < 2^31."""
    D = x.shape[-1]
    amax = x.abs().amax(-1).clamp(min=1)
    shx = (bitlen(amax) - 25).clamp(min=-6)
    xs = rshc(x, shx[:, None])                                   # |xs| <= 2^25
    S = (xs * xs).sum(-1)                                        # < 2^61 for D = 2048
    T = S + rshc(torch.full_like(S, EPS60 * D), 60 - 2 * (FR - shx))
    m = (73 - bitlen(T)) // 2                                    # T * 2^2m / D in [2^60, 2^62)
    s = 2 * m - 11
    V = torch.where(s >= 0, T << s.clamp(min=0), T >> (-s).clamp(min=0))
    return isqrt(V), shx - FR - m


def exp2neg_lut(t, lut, out_bits):
    """2^-t at out_bits for t >= 0 at FT2 bits: table 2^-f (f step 2^-14, at 2^60) plus linear interpolation."""
    n = t >> FT2
    f = t & ((1 << FT2) - 1)
    idx = f >> (FT2 - EXP2_BITS)
    rr = f & ((1 << (FT2 - EXP2_BITS)) - 1)
    a = lut[idx]
    v = a + rshc((lut[idx + 1] - a) * rr, FT2 - EXP2_BITS)
    return rshc(v, EXP2_ONE - out_bits + n)


def silu_mul(g, u, lut):
    """silu(g) * u for g, u at FR -> (P, eps, ok): value = P * 2^eps, P = silu(g at FR) * u exactly when ok
    (every |silu| * |u| < 2^62, checked per element by bit lengths; g is clamped at +-2^38 for the table)."""
    a = g.abs().clamp(max=64 << FR)
    t = rshc(a * L2E30, FR + 30 - FT2)
    E = exp2neg_lut(t, lut, 30)                                  # e^-|g| at 2^30
    den = (1 << 30) + E
    sig = torch.where(g >= 0, rdiv(torch.full_like(E, 1 << 60), den), rdiv(E << 30, den))
    gc = g.clamp(-(1 << 32), 1 << 32)
    s = rshc(gc * sig, 30)
    ok = ((bitlen(s.abs()) + bitlen(u.abs())) <= 62).all() & (g.abs().amax() <= 1 << 32)
    return s * u, -2 * FR, ok


def softmax_w(raw, ok, sh, kbits, lut, fp):
    """Attention weights at fp bits from exact scores raw [.., S] (value raw * 2^(sh - kbits - 3), sh per
    row [.., 1]); ok marks visible keys. Returns p (0 where masked) and the row sums."""
    NEG = -(1 << 62)
    raw = torch.where(ok, raw, torch.full_like(raw, NEG))
    z = raw.amax(-1, keepdim=True) - raw                         # >= 0
    zf = rshc(z, kbits + 3 - sh - FZ).clamp(max=48 << FZ)        # -score diff at FZ bits, <= 48
    t = rshc(zf * L2E28, FZ + 28 - FT2)
    p = torch.where(ok, exp2neg_lut(t, lut, fp), torch.zeros_like(t))
    return p, p.sum(-1, keepdim=True)
