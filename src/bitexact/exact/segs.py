"""Element-wise segments of the exact-bf16 engine between its matrix products.

Each stage is a pure function of integer tensors (shifts, exact int64 sums and products, floor division,
table lookups, and float64 frexp / sqrt results corrected to exact integers). Segs can wrap the stages in
torch.compile; fused and eager versions compute the same integers (checked by the determinism runs).
Stages are cut so that heavy element-wise work (int64 division, exp, the linear epilogue) is a parallel
kernel of its own and is not fused into a row reduction (one program per row is slow for a decode row).

A linear output is produced from C (the exact dense products, [split, M, N], summed here) and the sparse
residual: rs = (B,) with B precomputed (index_add, eager / CPU), or rs = (q, ell_c, ell_v) in padded
per-row (ELL) form, gathered inside the stage so a compiled stage never materializes it.
"""

import torch

from bitexact.exact import ops
from bitexact.exact.consts import FR, FT, LIM3
from bitexact.exact.wmm import epilogue
from bitexact.llama import FF, HD, HKV, D, H

FK, FV, FP = 17, 19, 22                                     # k, v, attention weights (fractional bits)


# ---- stages ----
def lin_y(C, rs, d, g, sh):
    T = C.sum(0) if C.dim() == 3 else C
    if len(rs) == 1:
        B = rs[0]
    else:
        q, ec, ev = rs
        B = (q[:, ec].long() * ev).sum(-1) if ec.shape[1] else torch.zeros_like(T)
    return epilogue(T, B, d, g, sh, FR)


def lin_add(C, rs, d, g, sh, x):
    return x + lin_y(C, rs, d, g, sh).view(x.shape)


def rms(x, err):
    r, er = ops.rms(x)
    return r, er, err | (x.abs().amax() >= 1 << 40)


def ndiv(x, gm, ge, r, er):
    """x * gamma / rms at FN (one rounding per element)."""
    return ops.div_round(x * gm, ge - FR - er[:, None] + ops.FN, r[:, None])


def quant(P, eps):
    q, sh = ops.quant(P, eps)
    return q.to(torch.int32), sh


def _rope_full(x, c, s):
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)


def lin_rope(C, rs, d, g, sh, pos, cos, sin, B, T):
    """qkv linear + RoPE -> q at FR + FT [B, T, H, HD], k at FK, v at FV [B, T, HKV, HD], y (for the check)."""
    y = lin_y(C, rs, d, g, sh).view(B, T, -1)
    c, s = cos[pos][:, :, None], sin[pos][:, :, None]
    Pq = _rope_full(y[..., :D].reshape(B, T, H, HD), c, s)
    k = ops.rshc(_rope_full(y[..., D:D + HKV * HD].reshape(B, T, HKV, HD), c, s), FR + FT - FK)
    v = ops.rshc(y[..., D + HKV * HD:].reshape(B, T, HKV, HD), FR - FV)
    return Pq, k, v, y


def q_group(Pq, k, v, y, err):
    """Quantize q per (token, head) and group by KV head: [B*HKV, 4T, HD] int32 and exponents [.., 1]."""
    B, T = Pq.shape[0], Pq.shape[1]
    qi, shq = ops.quant(Pq, -(FR + FT))
    g4 = H // HKV
    qg = qi.view(B, T, HKV, g4, HD).permute(0, 2, 3, 1, 4).reshape(B * HKV, g4 * T, HD).to(torch.int32)
    shg = shq.view(B, T, HKV, g4).permute(0, 2, 3, 1).reshape(B * HKV, g4 * T, 1)
    err = err | (y.abs().amax() >= 1 << 31) | (k.abs().amax() > LIM3) | (v.abs().amax() > LIM3) \
        | (shq.amax() > FK + 3 - ops.FZ)                       # rope products < 2^62, cache limbs
    return qg, shg, err


def softmax(raw, qpos, shg, lut):
    ok = torch.arange(raw.shape[-1], device=raw.device) <= qpos
    p, den = ops.softmax_w(raw, ok, shg, FK, lut, FP)
    return p.to(torch.int32), den


def attn_div(num, den, T):
    """num [B*HKV, 4T, HD] at FV + FP, den [B*HKV, 4T, 1] -> o at FR [B*T, D] (one rounding)."""
    B, g = num.shape[0] // HKV, H // HKV
    o = ops.div_round(num, FR - FV, den)
    return o.view(B, HKV, g, T, HD).permute(0, 3, 1, 2, 4).reshape(B * T, D)


def lin_silu(C, rs, d, g, sh, lut):
    gu = lin_y(C, rs, d, g, sh)
    P, _, ok = ops.silu_mul(gu[:, :FF], gu[:, FF:], lut)
    return P, ok


def quant_silu(P, ok, err):
    q, sh = ops.quant(P, -2 * FR)
    return q.to(torch.int32), sh, err | ~ok


STAGES = ("lin_y", "lin_add", "rms", "ndiv", "quant", "lin_rope", "q_group", "softmax", "attn_div",
          "lin_silu", "quant_silu")


class Segs:
    def __init__(self, compiled=False):
        if compiled:
            import torch._dynamo
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        wrap = (lambda f: torch.compile(f, dynamic=None)) if compiled else (lambda f: f)
        self.s = type("S", (), {n: staticmethod(wrap(globals()[n])) for n in STAGES})

    def norm_q(self, x, gm, ge, err):
        r, er, err = self.s.rms(x, err)
        q, sh = self.s.quant(self.s.ndiv(x, gm, ge, r, er), -ops.FN)
        return q, sh, err

    def pre_norm(self, x, gm, ge):
        return self.norm_q(x, gm, ge, torch.zeros((), dtype=torch.bool, device=x.device))[:2]

    def lin_out(self, C, rs, d, g, sh):
        return self.s.lin_y(C, rs, d, g, sh)

    def add_norm(self, C, rs, d, g, sh, x, gm, ge, err):
        x = self.s.lin_add(C, rs, d, g, sh, x)
        q, s, err = self.norm_q(x.view(-1, D), gm, ge, err)
        return x, q, s, err

    def qkv(self, C, rs, d, g, sh, pos, cos, sin, B, T, err):
        Pq, k, v, y = self.s.lin_rope(C, rs, d, g, sh, pos, cos, sin, B, T)
        qg, shg, err = self.s.q_group(Pq, k, v, y, err)
        return qg, shg, k, v, err

    def softmax(self, raw, qpos, shg, lut):
        return self.s.softmax(raw, qpos, shg, lut)

    def attn_out(self, num, den, T):
        return self.s.quant(self.s.attn_div(num, den, T), -FR)

    def gu_silu(self, C, rs, d, g, sh, lut, err):
        P, gu = self.s.lin_silu(C, rs, d, g, sh, lut)
        return self.s.quant_silu(P, gu, err)
