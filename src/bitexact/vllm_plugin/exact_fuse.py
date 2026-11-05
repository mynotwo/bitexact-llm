"""Fused element-wise kernels of exact mode (vLLM plugin, CUDA): one Triton kernel for each of the stages that
exact/segs.py runs as ~10 Inductor kernels between two linears.

  _norm_kernel      x = emb[ids] | x + y | x ;  pre_norm(x, gamma): ops.rms, ndiv (div_round of x * gamma),
                    ops.quant(., -FN) -> int32 q [M, D], sh [M]        (segs.add_y + norm_q)
  _qkv_kernel       RoPE of q / k, v >> (FR - FV), q quantized per (token, head), K / V stored as int8 limb
                    planes in the paged cache                          (segs.rope_y + q_group + tri_paged.store)
  _attn_out_kernel  o = div_round(num, FR - FV, den), ops.quant(o, -FR) (segs.attn_div + quant)
  _silu_kernel      silu(g) * u (table + interpolation), ops.quant(., -2 FR): two passes over a row, P is
                    recomputed in the second                           (segs.silu_y + quant_silu)

Every element goes through the same int64 operations as exact/ops.py (shifts follow torch's rules, floor
divisions of possibly negative numerators are corrected, bitlen is 64 - clz), so the integers are the segs
stages'. Rows are independent; maxima and sums are exact integer reductions. Bound flags are written to
`fl` (int32 [1], set to 1, never cleared) under the same conditions as the err terms of the segs stages.
tests/test_vllm_exact_fuse.py compares every kernel with the eager segs stages.
"""

import torch
import triton
import triton.language as tl

from bitexact.exact import ops
from bitexact.exact.consts import EXP2_BITS, EXP2_ONE, FR, FT, LIM3
from bitexact.exact.segs import FK, FV
from bitexact.vllm_plugin.tri_fuse import _flagged, _isqrt
from bitexact.vllm_plugin.tri_mlp import _bitlen, _shl, _shr

EXPBC = tl.constexpr(EXP2_BITS)
FRC, FNC, QBC, NEGC, FTC, EPS60C, FRVC, FKC, FZC, LIM3C, L2E30C, FT2C, EXP2_ONEC = (tl.constexpr(v) for v in (
    FR, ops.FN, ops.QB, ops.NEG_BIG, FT, ops.EPS60, FR - FV, FK, ops.FZ, LIM3, ops.L2E30, ops.FT2, EXP2_ONE))


@triton.jit
def _rshc(x, s):
    """ops.rshc with an int64 shift s (clamped to [-62, 62])."""
    s = tl.minimum(tl.maximum(s, -62), 62)
    pos = tl.maximum(s, 0)
    half = tl.where(pos > 0, (x * 0 + 1) << tl.maximum(pos - 1, 0), 0)
    return ((x + half) >> pos) << tl.maximum(-s, 0)


@triton.jit
def _rshs(x, S: tl.constexpr):
    return (x + (1 << (S - 1))) >> S


@triton.jit
def _fdiv(a, b):
    """floor(a / b) for int64 b > 0 (a of either sign)."""
    q = a // b
    return tl.where(a - q * b < 0, q - 1, q)


@triton.jit
def _div_round(a, p, r, NIT: tl.constexpr):
    """ops.div_round(a, p, r) with NIT of its three 28-bit long-division rounds (NIT 3 is the general case;
    1 is exact for 0 <= p <= 28)."""
    Q = _fdiv(a, r)
    R = a - Q * r
    left = tl.maximum(p, 0)
    for _ in tl.static_range(NIT):
        c = tl.minimum(left, 28)
        X = R << c
        q1 = X // r
        Q = (Q << c) + q1
        R = X - q1 * r
        left = left - c
    up = Q + tl.where(2 * R >= r, 1, 0).to(tl.int64)
    k = tl.minimum(tl.maximum(-p, 1), 62)
    down = (Q + ((Q * 0 + 1) << (k - 1))) >> k
    return tl.where(p >= 0, up, down)


@triton.jit
def _quant_sh(amax, eps: tl.constexpr):
    """ops.quant's row exponent from the row's max |P| (>= 0): max(e - QB, NEG_BIG), e = bitlen + eps or NEG_BIG."""
    e = tl.where(amax != 0, _bitlen(amax) + eps, NEGC)
    return tl.maximum(e - QBC, NEGC)


@triton.jit
def _norm_kernel(x_ptr, y_ptr, ids_ptr, gm_ptr, ge_ptr, xo_ptr, q_ptr, sh_ptr, err_ptr,
                 K2: tl.constexpr, LG: tl.constexpr, MODE: tl.constexpr, NORM: tl.constexpr,
                 STORE: tl.constexpr):
    """One program per row. MODE 0: x = emb[ids] (x_ptr int32 table); 1: x = x_ptr + y_ptr; 2: x = x_ptr (int64)."""
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, K2)
    if MODE == 0:
        x = tl.load(x_ptr + tl.load(ids_ptr + row).to(tl.int64) * K2 + col).to(tl.int64)
    elif MODE == 1:
        x = tl.load(x_ptr + row * K2 + col) + tl.load(y_ptr + row * K2 + col)
    else:
        x = tl.load(x_ptr + row * K2 + col)
    if STORE:
        tl.store(xo_ptr + row * K2 + col, x)
    ax = tl.max(tl.abs(x), 0)
    _flagged(err_ptr, ax >= (1 << 40))
    if NORM:
        # ops.rms
        amax = tl.maximum(ax, 1)
        shx = tl.maximum(_bitlen(amax) - 25, -6)
        xs = _rshc(x, shx)
        S = tl.sum(xs * xs, 0)
        T = S + _rshc(tl.full((), EPS60C * K2, tl.int64), 60 - 2 * (FRC - shx))
        m = (62 + LG - _bitlen(T)) >> 1
        s = 2 * m - LG
        V = tl.where(s >= 0, _shl(T, tl.maximum(s, 0)), _shr(T, tl.maximum(-s, 0)))
        r = _isqrt(V)
        er = shx - FRC - m
        # ndiv + quant(., -FN)
        gm = tl.load(gm_ptr + col)
        ge = tl.load(ge_ptr + col)
        P = _div_round(x * gm, ge - FRC - er + FNC, r, 3)
        sh = _quant_sh(tl.max(tl.abs(P), 0), -FNC)
        q = _rshc(P, sh + FNC)
        tl.store(q_ptr + row * K2 + col, q.to(tl.int32))
        tl.store(sh_ptr + row, sh)


@triton.jit
def _qkv_kernel(y_ptr, cos_ptr, sin_ptr, pos_ptr, slot_ptr, kv_ptr, q_ptr, sh_ptr, err_ptr,
                sc_nb, sc_h, sc_bs, BS: tl.constexpr, HC: tl.constexpr, HKVC: tl.constexpr,
                HDC: tl.constexpr, GC: tl.constexpr, STORE: tl.constexpr):
    """One program per (token, KV head): that head's GC query heads, its K and its V (rope_y + q_group + store)."""
    row = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    HH: tl.constexpr = HDC // 2
    NT: tl.constexpr = (HC + 2 * HKVC) * HDC
    d = tl.arange(0, HH)
    g = tl.arange(0, GC)
    pos = tl.load(pos_ptr + row)
    cs = tl.load(cos_ptr + pos * HH + d)
    sn = tl.load(sin_ptr + pos * HH + d)
    yr = y_ptr + row * NT
    qc = h * (GC * HDC) + g[:, None] * HDC + d[None, :]
    x1 = tl.load(yr + qc)
    x2 = tl.load(yr + qc + HH)
    p1 = x1 * cs[None, :] - x2 * sn[None, :]
    p2 = x2 * cs[None, :] + x1 * sn[None, :]
    amax = tl.maximum(tl.max(tl.abs(p1), 1), tl.max(tl.abs(p2), 1))                # [GC]
    sh = _quant_sh(amax, -(FRC + FTC))
    q1 = _rshc(p1, sh[:, None] + (FRC + FTC))
    q2 = _rshc(p2, sh[:, None] + (FRC + FTC))
    qo = q_ptr + row * (HC * HDC) + qc
    tl.store(qo, q1.to(tl.int32))
    tl.store(qo + HH, q2.to(tl.int32))
    tl.store(sh_ptr + row * HC + h * GC + g, sh)
    kc = HC * HDC + h * HDC + d
    k1 = tl.load(yr + kc)
    k2 = tl.load(yr + kc + HH)
    vc = (HC + HKVC) * HDC + h * HDC + d
    v1r = tl.load(yr + vc)
    v2r = tl.load(yr + vc + HH)
    k1o = _rshs(k1 * cs - k2 * sn, FRC + FTC - FKC)
    k2o = _rshs(k2 * cs + k1 * sn, FRC + FTC - FKC)
    v1 = _rshs(v1r, FRVC)
    v2 = _rshs(v2r, FRVC)
    big = tl.maximum(tl.maximum(tl.max(tl.max(tl.abs(x1), 1), 0), tl.max(tl.max(tl.abs(x2), 1), 0)),
                     tl.maximum(tl.maximum(tl.max(tl.abs(k1), 0), tl.max(tl.abs(k2), 0)),
                                tl.maximum(tl.max(tl.abs(v1r), 0), tl.max(tl.abs(v2r), 0))))
    _flagged(err_ptr, big >= (1 << 31))
    kvbig = tl.maximum(tl.maximum(tl.max(tl.abs(k1o), 0), tl.max(tl.abs(k2o), 0)),
                       tl.maximum(tl.max(tl.abs(v1), 0), tl.max(tl.abs(v2), 0)))
    _flagged(err_ptr, kvbig > LIM3C)
    _flagged(err_ptr, tl.max(sh, 0) > FKC + 3 - FZC)
    if STORE:
        slot = tl.load(slot_ptr + row).to(tl.int64)
        base = kv_ptr + (slot // BS) * sc_nb + h * sc_h + (slot % BS) * sc_bs
        ok = slot >= 0
        for half in tl.static_range(2):
            kk = (k1o if half == 0 else k2o).to(tl.int32)
            vv = (v1 if half == 0 else v2).to(tl.int32)
            for i in tl.static_range(3):
                k0 = ((kk + 128) & 255) - 128
                v0 = ((vv + 128) & 255) - 128
                tl.store(base + i * HDC + half * HH + d, k0.to(tl.int8), mask=ok)
                tl.store(base + (3 + i) * HDC + half * HH + d, v0.to(tl.int8), mask=ok)
                kk, vv = (kk - k0) >> 8, (vv - v0) >> 8


@triton.jit
def _attn_out_kernel(num_ptr, den_ptr, q_ptr, sh_ptr, K2: tl.constexpr, HDC: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, K2)
    a = tl.load(num_ptr + row * K2 + col)
    b = tl.load(den_ptr + row * (K2 // HDC) + col // HDC)
    o = _div_round(a, a * 0 + (FRVC), b, 1)
    sh = _quant_sh(tl.max(tl.abs(o), 0), -FRC)
    tl.store(q_ptr + row * K2 + col, _rshc(o, sh + FRC).to(tl.int32))
    tl.store(sh_ptr + row, sh)


@triton.jit
def _silu_p(g, u, lut_ptr):
    """ops.silu_mul: P = silu(g) * u (int64) and the per-element bound violation (ok is False)."""
    a = tl.minimum(tl.abs(g), 64 << FRC)
    t = _rshs(a * L2E30C, FRC + 30 - FT2C)
    n = t >> FT2C
    f = t & ((1 << FT2C) - 1)
    idx = f >> (FT2C - EXPBC)
    rr = f & ((1 << (FT2C - EXPBC)) - 1)
    a0 = tl.load(lut_ptr + idx)
    a1 = tl.load(lut_ptr + idx + 1)
    v = a0 + _rshs((a1 - a0) * rr, FT2C - EXPBC)
    E = _rshc(v, EXP2_ONEC - 30 + n)
    den = (1 << 30) + E
    sig = tl.where(g >= 0, (2 * (1 << 60) + den) // (2 * den), (2 * (E << 30) + den) // (2 * den))
    gc = tl.minimum(tl.maximum(g, -(1 << 32)), 1 << 32)
    s = _rshs(gc * sig, 30)
    bad = (_bitlen(tl.abs(s)) + _bitlen(tl.abs(u)) > 62) | (tl.abs(g) > (1 << 32))
    return s * u, bad


@triton.jit
def _silu_kernel(y_ptr, lut_ptr, q_ptr, sh_ptr, err_ptr, FF, CH: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, CH)
    yr = y_ptr + row * (2 * FF)
    mx = tl.zeros([CH], tl.int64)
    nbad = tl.zeros([CH], tl.int32)
    for c0 in range(0, FF, CH):
        ok = c0 + lane < FF
        g = tl.load(yr + c0 + lane, mask=ok, other=0)
        u = tl.load(yr + FF + c0 + lane, mask=ok, other=0)
        P, bad = _silu_p(g, u, lut_ptr)
        mx = tl.maximum(mx, tl.abs(P))
        nbad += (bad & ok).to(tl.int32)
    _flagged(err_ptr, tl.sum(nbad, 0) > 0)
    sh = _quant_sh(tl.max(mx, 0), -2 * FRC)
    tl.store(sh_ptr + row, sh)
    for c0 in range(0, FF, CH):
        ok = c0 + lane < FF
        g = tl.load(yr + c0 + lane, mask=ok, other=0)
        u = tl.load(yr + FF + c0 + lane, mask=ok, other=0)
        P, bad = _silu_p(g, u, lut_ptr)
        tl.store(q_ptr + row * FF + c0 + lane, _rshc(P, sh + 2 * FRC).to(tl.int32), mask=ok)


# ---- host wrappers (all device-only, CUDA-graph safe) ----

def _norm(m, x, y, ids, gamma, fl, mode, norm, store):
    M, D = (ids.shape[0], m.D) if mode == 0 else x.shape
    assert D & (D - 1) == 0 and D <= 4096
    xo = torch.empty(M, D, dtype=torch.long, device=m.dev) if store else x
    q = torch.empty(M, D, dtype=torch.int32, device=m.dev) if norm else xo
    sh = torch.empty(M, dtype=torch.long, device=m.dev) if norm else xo
    gm, ge = gamma if gamma is not None else (xo, xo)
    if M:
        _norm_kernel[(M,)](m.emb if mode == 0 else x, y, ids, gm, ge, xo, q, sh, fl, K2=D, LG=D.bit_length() - 1,
                           MODE=mode, NORM=norm, STORE=store, num_warps=8)
    return xo, q, sh


def embed_norm(m, ids, gamma, fl):
    """x = emb[ids] (int64) and pre_norm(x, gamma) -> (x, q, sh)."""
    return _norm(m, None, None, ids, gamma, fl, 0, True, True)


def add_norm(m, y, x, gamma, fl):
    """x + y and, when gamma is given, pre_norm of it -> (x, q, sh); else (x, None, None) with the bound flag."""
    x, q, sh = _norm(m, x, y, None, gamma, fl, 1, gamma is not None, True)
    return (x, q, sh) if gamma is not None else (x, None, None)


def pre_norm(m, h, gamma, fl):
    """pre_norm of h -> (q, sh)."""
    return _norm(m, h, None, None, gamma, fl, 2, True, False)[1:]


def qkv(m, y, pos, slot, cache, fl):
    """qkv linear output y [N, (H + 2 HKV) HD] -> (q int32 [N, H HD], sh int64 [N, H]); K / V stored at slot in
    cache (int8 limb planes; None: no store)."""
    N = y.shape[0]
    H, HKV, HD = m.H, m.HKV, m.HD
    G = H // HKV
    assert G & (G - 1) == 0
    q = torch.empty(N, H * HD, dtype=torch.int32, device=m.dev)
    sh = torch.empty(N, H, dtype=torch.long, device=m.dev)
    c = cache if cache is not None else q
    st = c.stride()[:3] if cache is not None else (0, 0, 0)
    if N:
        _qkv_kernel[(N, HKV)](y, m.cos, m.sin, pos, slot, c, q, sh, fl, *st,
                              BS=c.shape[2] if cache is not None else 1, HC=H, HKVC=HKV, HDC=HD, GC=G,
                              STORE=cache is not None, num_warps=2)
    return q, sh


def attn_out(m, num, den):
    """num [N, HKV, G, HD] (or any shape with N rows of H HD), den likewise [.., 1] -> (q int32 [N, H HD], sh [N])."""
    N = num.shape[0]
    K2 = m.H * m.HD
    q = torch.empty(N, K2, dtype=torch.int32, device=m.dev)
    sh = torch.empty(N, dtype=torch.long, device=m.dev)
    if N:
        _attn_out_kernel[(N,)](num, den, q, sh, K2=K2, HDC=m.HD, num_warps=8)
    return q, sh


def silu_quant(m, y, lut, fl):
    """gate_up output y [N, 2 FF] -> (q int32 [N, FF], sh [N]) of quant(silu(g) u, -2 FR)."""
    N, FF = y.shape[0], y.shape[1] // 2
    q = torch.empty(N, FF, dtype=torch.int32, device=m.dev)
    sh = torch.empty(N, dtype=torch.long, device=m.dev)
    if N:
        _silu_kernel[(N,)](y, lut, q, sh, fl, FF, CH=min(2048, triton.next_power_of_2(FF)), num_warps=8)
    return q, sh
