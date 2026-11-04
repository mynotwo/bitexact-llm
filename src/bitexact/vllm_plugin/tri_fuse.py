"""Decode-step kernels that fuse the integer ops between the GEMMs of one layer (vLLM plugin, CUDA, nl = 2).

IntLlama (int8/segs.py) runs, per layer and in this order (each bullet was 1 to 7 Inductor kernels):
  _add_norm_kernel   y = post(c_prev) (t = rsh(acc * m, fmsh); y = rsh(t * mant, 24 - FR - e)), x += y, then
                     pre_norm(x, gamma): rms, quant_rows(x * gamma, 2), norm24, stack_limbs
  _qkv_kernel        post of the qkv GEMM, RoPE of q and k, v >> 2, bound flags, K / V limb planes stored in
                     the paged cache (tri_paged._store_kernel's layout)
  _attn_out_kernel   o = rdiv(num << 2, den), pre_plain(o): quant_rows, norm24, stack_limbs
  MLP                tri_mlp._h_kernel / _quant_kernel with the flag maxima as atomic maxima (scr)
  _flag_kernel       the MLP product flags of all layers, then resets scr
Same integers as the unfused ops: every element goes through the same int64 operations in the same order;
rows are independent; maxima are exact in any order; floor divisions of positive quotients are int64
divisions or rdiv32's float32 estimate with its two exact corrections (as tri_mlp). Bound flags set m.err (a
bool, only ever set to 1) from the kernels; the unfused code ORs the same conditions into it.
tests/test_tri_fuse.py compares every output with the eager segments.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from bitexact.int8.consts import FG, FR, FT
from bitexact.int8.ops import EPS40, qmax
from bitexact.int8.segs import FQ, FV
from bitexact.intops import LIM
from bitexact.vllm_plugin import tri_mlp
from bitexact.vllm_plugin.tri_mlp import _bitlen, _rsh

FRC, FGC, FVC = tl.constexpr(FR), tl.constexpr(FG), tl.constexpr(FV)
SHR = tl.constexpr(FT + FR - FQ)                              # RoPE shift
EPSC = tl.constexpr(EPS40)
QMAX = tl.constexpr(qmax(2))
LIM3 = tl.constexpr(LIM[3])


@triton.jit
def _flagged(err_ptr, bad):
    """err = 1 when the scalar bad holds: every writer stores the same value."""
    tl.store(err_ptr, 1, mask=bad)


@triton.jit
def _quant2(z, amax):
    """quant_rows(z, 2) for a row with max |z| = amax >= 1: (q, as_, sh)."""
    sh = _bitlen(amax) - 30
    as_ = _rsh(amax, sh)
    n = 2 * (_rsh(z, sh) * QMAX) + as_                       # rdiv32(zs * qmax, as_), |q| <= qmax
    d = 2 * as_
    q = tl.floor(n.to(tl.float32) / d.to(tl.float32)).to(tl.int64)
    q = tl.where(q * d > n, q - 1, q)
    q = tl.where((q + 1) * d <= n, q + 1, q)
    return q, as_, sh


@triton.jit
def _store_limbs(q, q_ptr, row, M, col, K: tl.constexpr):
    """stack_limbs(q, 2): int8 limb 0 in row `row`, limb 1 in row M + row."""
    lo = ((q + 128) & 255) - 128
    hi = ((((q - lo) >> 8) + 128) & 255) - 128
    tl.store(q_ptr + row.to(tl.int64) * K + col, lo.to(tl.int8))
    tl.store(q_ptr + (M + row).to(tl.int64) * K + col, hi.to(tl.int8))


@triton.jit
def _isqrt(x):
    s = libdevice.floor(libdevice.sqrt(x.to(tl.float64))).to(tl.int64)
    for _ in tl.static_range(2):
        s = tl.where(s * s > x, s - 1, s)
        s = tl.where((s + 1) * (s + 1) <= x, s + 1, s)
    return s


@triton.jit
def _y(c_ptr, m_ptr, col, row, M, N2, fmsh, mant, e):
    """segs.post for the columns col of row (c: limb 0 rows [0, M), limb 1 rows [M, 2M))."""
    off = row.to(tl.int64) * N2 + col
    acc = tl.load(c_ptr + off).to(tl.int64) + (tl.load(c_ptr + M * N2 + off).to(tl.int64) << 8)
    return _rsh(_rsh(acc * tl.load(m_ptr + col), fmsh) * mant, 24 - FRC - e)


@triton.jit
def _add_norm_kernel(c_ptr, m_ptr, fmsh_ptr, pmant_ptr, pe_ptr, x_ptr, ids_ptr, g_ptr, xo_ptr, q_ptr, mant_ptr,
                     e_ptr, err_ptr, M, PAD, K2: tl.constexpr, MODE: tl.constexpr, NORM: tl.constexpr):
    """MODE 0: x = emb[ids] (int32 table x_ptr; stored to xo); 1: x += post(c) (x_ptr; stored to xo).
    NORM: pre_norm(x, gamma) -> limbs, mant, e (pad rows zeroed); else only the bound flag |x| < 2^42."""
    row = tl.program_id(0)
    col = tl.arange(0, K2)
    if row < M:
        if MODE == 0:
            x = tl.load(x_ptr + tl.load(ids_ptr + row).to(tl.int64) * K2 + col).to(tl.int64)
        else:
            x = tl.load(x_ptr + row.to(tl.int64) * K2 + col) + _y(
                c_ptr, m_ptr, col, row, M, K2, tl.load(fmsh_ptr), tl.load(pmant_ptr + row), tl.load(pe_ptr + row))
        tl.store(xo_ptr + row.to(tl.int64) * K2 + col, x)
        ax = tl.max(tl.abs(x), 0)
        if not NORM:
            _flagged(err_ptr, ax >= (1 << 42))
        else:
            amax = tl.maximum(ax, 1)                         # rms
            shx = _bitlen(amax) - 24
            xs = _rsh(x, shx)
            ms = tl.sum(xs * xs, 0) // K2
            eps = _rsh(tl.full((), EPSC, tl.int64), 40 - 2 * (FRC - shx))
            r = _isqrt(ms + eps)
            z = x * tl.load(g_ptr + col)
            q, as_, sh = _quant2(z, tl.maximum(tl.max(tl.abs(z), 0), 1))
            _store_limbs(q, q_ptr, row, M, col, K2)
            mq = (2 * (as_ << 30) + r) // (2 * r)            # norm24(rdiv(as_ << 30, r), sh - FG - shx - 30)
            b = _bitlen(mq) - 24
            tl.store(mant_ptr + row, _rsh(mq, b))
            tl.store(e_ptr + row, sh - FGC - shx - 30 + b)
    if NORM:
        if row < PAD:
            tl.store(q_ptr + (2 * M + row).to(tl.int64) * K2 + col, tl.zeros([K2], tl.int8))


def _limbs(M, K, dev):
    rows = 2 * M
    pad = max(24, (rows + 7) // 8 * 8) - rows
    return torch.empty(rows + pad, K, dtype=torch.int8, device=dev), pad


def scratch(m):
    """Per-layer maxima |silu(g)|, |u| of the MLP (reset by flag_end after each forward)."""
    if getattr(m, "_scr", None) is None:
        m._scr = torch.zeros(m.L, 2, dtype=torch.long, device=m.dev)
        m._err8 = m.err.view(torch.uint8)
    return m._scr


def embed_norm(m, ids, gamma):
    """x = emb[ids] and pre_norm(x, gamma) -> (x, limbs, mant, e)."""
    return _run(m, None, None, None, None, None, ids, gamma, 0, True)


def add_norm(m, c, lin, mant, e, x, gamma):
    """x + post(c) and, when gamma is given, pre_norm of it -> (x, limbs, mant, e); else (x, None, None, None)
    with the 2^42 bound flag of x."""
    return _run(m, c, lin, mant, e, x, None, gamma, 1, gamma is not None)


def _run(m, c, lin, mant, e, x, ids, gamma, mode, norm):
    scratch(m)
    M = ids.shape[0] if mode == 0 else x.shape[0]
    D = m.D
    xo = torch.empty(M, D, dtype=torch.long, device=m.dev)
    if norm:
        q, pad = _limbs(M, D, m.dev)
        nm, ne = torch.empty(M, dtype=torch.long, device=m.dev), torch.empty(M, dtype=torch.long, device=m.dev)
    else:
        q, pad, nm, ne = xo, 0, xo, xo
    w, mw, fmsh = lin if lin is not None else (None, m.emb, None)
    _add_norm_kernel[(max(M, pad),)](c, mw, fmsh, mant, e, m.emb if mode == 0 else x, ids, gamma, xo, q, nm, ne,
                                     m._err8, M, pad, K2=D, MODE=mode, NORM=norm, num_warps=8)
    return (xo, q, nm, ne) if norm else (xo, None, None, None)


@triton.jit
def _qkv_kernel(c_ptr, m_ptr, fmsh_ptr, pmant_ptr, pe_ptr, cos_ptr, sin_ptr, pos_ptr, slot_ptr, kv_ptr, q_ptr,
                err_ptr, M, sc_nb, sc_h, sc_bs, BS: tl.constexpr, HC: tl.constexpr, HKVC: tl.constexpr,
                HDC: tl.constexpr, GC: tl.constexpr):
    row = tl.program_id(0)
    h = tl.program_id(1)
    fmsh, mant, e = tl.load(fmsh_ptr), tl.load(pmant_ptr + row), tl.load(pe_ptr + row)
    d = tl.arange(0, HDC // 2)
    pos = tl.load(pos_ptr + row)
    cs = tl.load(cos_ptr + pos * (HDC // 2) + d)
    sn = tl.load(sin_ptr + pos * (HDC // 2) + d)
    N2: tl.constexpr = (HC + 2 * HKVC) * HDC
    g = tl.arange(0, GC)
    qc = h * (GC * HDC) + g[:, None] * HDC + d[None, :]                  # q heads 4h .. 4h + 3, first halves
    x1 = _y(c_ptr, m_ptr, qc, row, M, N2, fmsh, mant, e)
    x2 = _y(c_ptr, m_ptr, qc + HDC // 2, row, M, N2, fmsh, mant, e)
    q1 = _rsh(x1 * cs[None, :] - x2 * sn[None, :], SHR)
    q2 = _rsh(x2 * cs[None, :] + x1 * sn[None, :], SHR)
    qo = q_ptr + row.to(tl.int64) * (HC * HDC) + qc
    tl.store(qo, q1)
    tl.store(qo + HDC // 2, q2)
    kc = HC * HDC + h * HDC + d
    k1 = _y(c_ptr, m_ptr, kc, row, M, N2, fmsh, mant, e)
    k2 = _y(c_ptr, m_ptr, kc + HDC // 2, row, M, N2, fmsh, mant, e)
    k1r = _rsh(k1 * cs - k2 * sn, SHR)
    k2r = _rsh(k2 * cs + k1 * sn, SHR)
    k1, k2 = k1r, k2r
    vc = (HC + HKVC) * HDC + h * HDC + d
    v1 = _rsh(_y(c_ptr, m_ptr, vc, row, M, N2, fmsh, mant, e), 2)
    v2 = _rsh(_y(c_ptr, m_ptr, vc + HDC // 2, row, M, N2, fmsh, mant, e), 2)
    big = tl.maximum(tl.maximum(tl.max(tl.max(tl.abs(q1), 1), 0), tl.max(tl.max(tl.abs(q2), 1), 0)),
                     tl.maximum(tl.maximum(tl.max(tl.abs(k1), 0), tl.max(tl.abs(k2), 0)),
                                tl.maximum(tl.max(tl.abs(v1), 0), tl.max(tl.abs(v2), 0))))
    _flagged(err_ptr, big > LIM3)
    slot = tl.load(slot_ptr + row).to(tl.int64)                       # tri_paged._store_kernel
    base = kv_ptr + (slot // BS) * sc_nb + h * sc_h + (slot % BS) * sc_bs
    ok = slot >= 0
    for half in tl.static_range(2):
        kk = (k1 if half == 0 else k2).to(tl.int32)
        vv = (v1 if half == 0 else v2).to(tl.int32)
        for i in tl.static_range(3):
            k0 = ((kk + 128) & 255) - 128
            v0 = ((vv + 128) & 255) - 128
            tl.store(base + i * HDC + half * (HDC // 2) + d, k0.to(tl.int8), mask=ok)
            tl.store(base + (3 + i) * HDC + half * (HDC // 2) + d, v0.to(tl.int8), mask=ok)
            kk, vv = (kk - k0) >> 8, (vv - v0) >> 8


def qkv(m, c, lin, mant, e, pos, slot, cache):
    """qkv GEMM output c -> q int64 [N, HKV, G, HD]; K / V of the tokens are stored at slot in cache (int8 planes)."""
    M = pos.shape[0]
    H, HKV, HD = m.H, m.HKV, m.HD
    G = H // HKV
    q = torch.empty(M, HKV, G, HD, dtype=torch.long, device=m.dev)
    _qkv_kernel[(M, HKV)](c, lin[1], lin[2], mant, e, m.cos, m.sin, pos, slot, cache, q, m._err8, M,
                          cache.stride(0), cache.stride(1), cache.stride(2), BS=cache.shape[2], HC=H, HKVC=HKV,
                          HDC=HD, GC=G, num_warps=2)
    return q


@triton.jit
def _attn_out_kernel(num_ptr, den_ptr, q_ptr, mant_ptr, e_ptr, M, PAD, K2: tl.constexpr, HDC: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, K2)
    if row < M:
        a = tl.load(num_ptr + row.to(tl.int64) * K2 + col) << (FRC - FVC)
        b = tl.load(den_ptr + row.to(tl.int64) * (K2 // HDC) + col // HDC)
        n, d = 2 * a + b, 2 * b                                          # rdiv: floor(n / d), d > 0
        o = n // d
        o = tl.where(n - o * d < 0, o - 1, o)
        q, as_, sh = _quant2(o, tl.maximum(tl.max(tl.abs(o), 0), 1))
        _store_limbs(q, q_ptr, row, M, col, K2)
        bl = _bitlen(as_) - 24                                           # norm24(as_, sh - FR)
        tl.store(mant_ptr + row, _rsh(as_, bl))
        tl.store(e_ptr + row, sh - FRC + bl)
    if row < PAD:
        tl.store(q_ptr + (2 * M + row).to(tl.int64) * K2 + col, tl.zeros([K2], tl.int8))


def attn_out(m, num, den):
    """num [N, HKV, G, HD], den [N, HKV, G, 1] -> (limbs, mant, e) of pre_plain(rdiv(num << 2, den))."""
    M = num.shape[0]
    K2 = num.shape[1] * num.shape[2] * num.shape[3]
    q, pad = _limbs(M, K2, m.dev)
    mant, e = torch.empty(M, dtype=torch.long, device=m.dev), torch.empty(M, dtype=torch.long, device=m.dev)
    _attn_out_kernel[(max(M, pad),)](num, den, q, mant, e, M, pad, K2=K2, HDC=num.shape[3], num_warps=8)
    return q, mant, e


def mlp_mid(m, ly, c, mant, e, layer):
    """tri_mlp.silu_quant on the gate_up GEMM output c, flag maxima into scr[layer] -> (limbs, mant, e)."""
    return tri_mlp.silu_quant(c, ly["wgu"][1], ly["wgu"][2], mant, e, m.silu_t, False, scr=scratch(m)[layer])[:3]


@triton.jit
def _flag_kernel(scr_ptr, err_ptr, L, LP: tl.constexpr):
    i = tl.arange(0, LP)
    ok = i < L
    s = tl.load(scr_ptr + 2 * i, mask=ok, other=0)
    u = tl.load(scr_ptr + 2 * i + 1, mask=ok, other=0)
    _flagged(err_ptr, tl.max(((s * u >= (1 << 62)) & ok).to(tl.int32), 0) > 0)
    tl.store(scr_ptr + 2 * i, tl.zeros_like(s), mask=ok)
    tl.store(scr_ptr + 2 * i + 1, tl.zeros_like(u), mask=ok)


def flag_end(m):
    _flag_kernel[(1,)](m._scr, m._err8, m.L, LP=triton.next_power_of_2(m.L), num_warps=1)
