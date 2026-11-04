"""Element-wise segments of the int8 engine between its matrix products.

Each segment is a pure function of integer tensors. Segs binds them for one device and limb count, and
can wrap them in torch.compile (fused Triton kernels). Compiled and eager versions compute the same
integers: every operation is an integer rule (shifts, floor division, exact int64 sums, table lookups,
and a float64 frexp/sqrt whose result is corrected to the exact integer), so fusion cannot change it.
Tests compare the two bit for bit.

On CUDA the pre-segments return the int8 limb matrix (nl limbs stacked along rows, padded to >= 24
rows for torch._int_mm) and the post-segments join the limb products; on CPU they pass q / acc through.
"""

import torch

from bitexact.int8.consts import FA, FE, FG, FR, FT
from bitexact.int8.ops import exp_of, norm24, quant_rows, rms, silu
from bitexact.intops import LIM, join_limbs, rdiv, rsh, stack_limbs
from bitexact.llama import HD, HKV, H

FQ, FV, FP = 14, 14, 20                                     # q / k, v, attention weights (fractional bits)
NEG = -(1 << 62)

# Attention score scale 1 / sqrt(head_dim) applied to d = raw - max (raw at 2 FQ bits) to get z at FA bits.
# head_dim 64: 2^-3, one shift. head_dim 128: 2^-3 * 2^-1/2, the 2^-1/2 as the integer SC128 = round(2^24 / sqrt 2)
# (11863283.203...). d is first clamped to >= ZCLAMP (anything below it is under the exp table anyway, z < -32), so
# |d * SC128| < 2^38 * 2^24 = 2^62. z = rsh(d * SC128, 24 + 2 FQ + 3 - FA).
SC128 = 11863283
ZCLAMP = -(1 << 38)


def score_rule(hd):
    """(mul, zsh): z = rsh(d, zsh) when mul is None, else rsh(max(d, ZCLAMP) * mul, zsh)."""
    base = 2 * FQ + 3 - FA
    if hd == 64:
        return None, base
    if hd == 128:
        return SC128, base + 24
    raise ValueError(f"unsupported head_dim {hd}")


def _amax(z):
    """max |z| over the whole tensor, reduced per leading row first: the same integer as z.abs().amax(),
    but Inductor then runs it across all rows in parallel (a direct global amax became a split reduction
    on a handful of blocks that also computed the segment's output)."""
    return z.abs().flatten(1).amax(1).amax()


def pre_norm(x, gamma, nl):
    """RMSNorm + per-row quantization: x [M, D] -> q, and the row scale amax(x * gamma) / rms(x)."""
    r, shx = rms(x)
    q, as_, sh = quant_rows(x * gamma, nl)
    mant, e = norm24(rdiv(as_ << 30, r), sh - FG - shx - 30)
    return q, mant, e


def pre_plain(x, nl):
    q, as_, sh = quant_rows(x, nl)
    mant, e = norm24(as_, sh - FR)
    return q, mant, e


def post(acc, m, fmsh, mant, e, check=True):
    """acc * s_c / qmax * row scale at FR (fmsh = fm + 8 (nl - 1) - 24, a 0-d tensor). check=False when
    |t| < 2^38 holds for every input (ops.post_bound, checked once by IntLlama): flag is then None."""
    t = rsh(acc * m, fmsh)
    flag = _amax(t) >= 1 << 38 if check else None    # t * mant < 2^62
    return rsh(t * mant[:, None], (24 - FR - e)[:, None]), flag


def _rope(x, c, s):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    sh = FT + FR - FQ
    return torch.cat([rsh(x1 * c - x2 * s, sh), rsh(x2 * c + x1 * s, sh)], -1)


def qkv(y, c, s, dm=(H, HKV, HD)):
    """y [B, T, (H + 2 HKV) HD] at FR, c / s [B, T, 1, HD / 2] -> grouped q [B*HKV, 4T, HD], k, v [B, T, HKV, HD], flag."""
    H, HKV, HD = dm
    B, T = y.shape[0], y.shape[1]
    D = H * HD
    q = _rope(y[..., :D].reshape(B, T, H, HD), c, s)
    k = _rope(y[..., D:D + HKV * HD].reshape(B, T, HKV, HD), c, s)
    v = rsh(y[..., D + HKV * HD:].reshape(B, T, HKV, HD), FR - FV)
    g = H // HKV
    qg = q.view(B, T, HKV, g, HD).permute(0, 2, 3, 1, 4).reshape(B * HKV, g * T, HD)
    flag = (_amax(q) > LIM[3]) | (_amax(k) > LIM[3]) | (_amax(v) > LIM[3])
    return qg, k, v, flag


def qkv_rows(y, c, s, dm=(H, HKV, HD)):
    """Like qkv for flat rows y [N, (H + 2 HKV) HD] (T = 1): q as rows [N * H, HD] (row n * H + h, i.e.
    [n, HKV, G, HD] flattened) and K, V int64 [N, HKV, HD] for tri_paged.store."""
    qg, k, v, flag = qkv(y.view(y.shape[0], 1, -1), c, s, dm)
    return qg.reshape(-1, dm[2]), k[:, 0], v[:, 0], flag


def softmax(raw, qpos, expt, hd=HD):
    """raw [B, HKV, gT, S] q.k at 2 FQ, qpos [B, 1, gT, 1] -> weights p at FP (masked: 0), sums."""
    ok = torch.arange(raw.shape[-1], device=raw.device) <= qpos
    raw = torch.where(ok, raw, torch.full_like(raw, NEG))
    d = raw - raw.amax(-1, keepdim=True)
    mul, zsh = score_rule(hd)                                          # / sqrt(hd)
    z = rsh(d, zsh) if mul is None else rsh(d.clamp(min=ZCLAMP) * mul, zsh)
    p = torch.where(ok, rsh(exp_of(z.clamp(min=-(1 << 40)), expt), FE - FP), torch.zeros_like(z))
    return p, p.sum(-1, keepdim=True)


def attn_out(num, den, T, nl, dm=(H, HKV, HD)):
    """num [B, HKV, gT, HD], den [B, HKV, gT, 1] -> quantized o [B*T, H*HD] for o_proj."""
    H, HKV, HD = dm
    B = num.shape[0]
    g = H // HKV
    o = rdiv(num << (FR - FV), den)
    o = o.view(B, HKV, g, T, HD).permute(0, 3, 1, 2, 4).reshape(B * T, H * HD)
    return pre_plain(o, nl)


def attn_rows(num, den):
    """num [n, HKV, M, HD], den [n, HKV, M, 1] -> o = round(num 2^(FR - FV) / den) at FR, same shape."""
    return rdiv(num << (FR - FV), den)


def silu_mul(gu, lut, nl):
    ff = gu.shape[1] // 2
    g, u = gu[:, :ff], gu[:, ff:]
    sg = silu(g, lut)
    flag = _amax(sg) * _amax(u) >= 1 << 62
    q, mant, e = pre_plain(rsh(sg * u, FR), nl)
    return q, mant, e, flag


class Segs:
    def __init__(self, nl, cuda, compiled, post_check=True, dm=(H, HKV, HD), dynamic=None):
        """dynamic=None: one compile per shape until a shape changes, then shape-generic kernels (nl is a Python int
        closed over and stays static). dynamic=False: one compile per distinct shape, kernels specialised to it
        (faster than the shape-generic ones; callers bucket the row counts, e.g. the packed scorer)."""
        self.nl, self.cuda, self.post_check, self.dm = nl, cuda, post_check, dm
        self.compiled, self.dynamic = compiled, dynamic
        if compiled and dynamic is False:
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        wrap = (lambda f: torch.compile(f, dynamic=dynamic)) if compiled else (lambda f: f)
        lim = (lambda q: stack_limbs(q, nl)) if cuda else (lambda q: q)
        join = (lambda c, M: join_limbs(c, M, nl)) if cuda else (lambda c, M: c)

        def with_limbs(out):
            return (lim(out[0]),) + tuple(out[1:])

        self.pre_norm = wrap(lambda x, gamma: with_limbs(pre_norm(x, gamma, nl)))
        self.pre_plain = wrap(lambda x: with_limbs(pre_plain(x, nl)))
        self.post = wrap(lambda c, m, fmsh, mant, e: post(join(c, mant.shape[0]), m, fmsh, mant, e, post_check))
        self.post_add = wrap(lambda c, m, fmsh, mant, e, x: _add(post(join(c, mant.shape[0]), m, fmsh, mant, e,
                                                                         post_check), x))
        self.qkv = wrap(lambda y, c, s: qkv(y, c, s, dm))
        self.qkv_rows = wrap(lambda y, c, s: qkv_rows(y, c, s, dm))
        self.attn_rows = torch.compile(attn_rows, dynamic=None) if compiled else attn_rows    # tile count varies
        self.softmax = wrap(lambda raw, qpos, expt: softmax(raw, qpos, expt, dm[2]))
        self.attn_out = wrap(lambda num, den, T: with_limbs(attn_out(num, den, T, nl, dm)))
        self.silu_mul = wrap(lambda gu, lut: with_limbs(silu_mul(gu, lut, nl)))


def _add(out, x):
    y, flag = out
    return x + y.view(x.shape), flag
