"""The integer ops between the MLP's gate_up GEMM and its down GEMM in two Triton kernels (vLLM plugin, CUDA).

The engine runs (IntLlama.mlp, nl = 2 limbs): c = torch._int_mm(limbs, Wgu^T), then int8.segs
  post:      acc = c_0 + (c_1 << 8);  t = rsh(acc * m_c, fmsh);  y = rsh(t * mant_r, 24 - FR - e_r)
  silu_mul:  g, u = y[:, :FF], y[:, FF:];  sg = silu(g);  h = rsh(sg * u, FR)
  pre_plain: quant_rows(h, 2) -> q, as_, sh;  (mant', e') = norm24(as_, sh - FR);  stack_limbs(q, 2)
  flags:     _amax(t) >= 2^38 (only when post_check), _amax(sg) * _amax(u) >= 2^62
Inductor splits this into four kernels that pass int64 [M, 2 FF] / [M, FF] tensors through memory. Here:
  _h_kernel     (row, column chunk): h of the chunk (int64 scratch) and the chunk's max |t|, |sg|, |u|, |h|;
  _quant_kernel (row, column chunk): the row's amax from the chunk maxima, q and its two int8 limbs; chunk 0
                writes mant', e', and the row maxima of |t|, |sg|, |u| for the flags.

Why the integers are the same:
  - each element goes through the same int64 operations in the same order (+ and * wrap mod 2^64 in both,
    shifts follow torch's rules for every amount: _shr / _shl);
  - maxima are exact and order free, so chunked row maxima, then maxima over rows, equal _amax;
  - bitlen32(x) (frexp of float32, corrected) is exact for x >= 1, so 64 - clz(x) is the same integer;
  - rdiv32 is exact for any float32 quotient within one unit of n / d (its two integer corrections); the
    Triton quotient is a few ulps from n / d < 2^16, so its floor is off by at most one, same as torch's;
  - the flag products amax(sg) * amax(u) are the same int64 multiply (wrapping alike).
tests/test_vllm_kernels.py compares every output with the eager segments.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from bitexact.int8.consts import FA, FR, SILU_LIM
from bitexact.int8.ops import qmax

FRC, FAC, SLC = tl.constexpr(FR), tl.constexpr(FA), tl.constexpr(SILU_LIM)
QMAX = tl.constexpr(qmax(2))


@triton.jit
def _shr(x, s):
    """torch's x >> s for int64 x and s >= 0 (amounts >= 63 shift by 63)."""
    return x >> tl.minimum(s, 63)


@triton.jit
def _shl(x, s):
    """torch's x << s for int64 x and s >= 0 (amounts >= 64 give 0)."""
    return tl.where(s >= 64, 0, x << tl.minimum(s, 63))


@triton.jit
def _rsh(x, s):
    """intops.rsh with an int64 shift s (tensor or scalar)."""
    pos = tl.maximum(s, 0)
    neg = tl.maximum(-s, 0)
    half = tl.where(pos > 0, _shl(x * 0 + 1, tl.maximum(pos - 1, 0)), 0)
    return _shl(_shr(x + half, pos), neg)


@triton.jit
def _bitlen(x):
    """intops.bitlen for int64 x >= 1."""
    return 64 - libdevice.clz(x).to(tl.int64)


@triton.jit
def _silu(g, lut_ptr):
    """int8.ops.silu: table at step 2^-FA, linear interpolation."""
    lo: tl.constexpr = -SLC << FAC
    hi: tl.constexpr = SLC << FAC
    gi = g >> (FRC - FAC)
    r = g - (gi << (FRC - FAC))
    i = tl.minimum(tl.maximum(gi, lo), hi - 1) - lo
    a = tl.load(lut_ptr + i)
    v = a + (((tl.load(lut_ptr + i + 1) - a) * r + (1 << (FRC - FAC - 1))) >> (FRC - FAC))
    return tl.where(gi >= hi, g, tl.where(gi < lo, 0, v))


@triton.jit
def _y(c_ptr, m_ptr, col, ok, row, M, N2, fmsh, mant, e):
    """segs.post for columns col of row: (t, y)."""
    off = row.to(tl.int64) * N2 + col
    acc = tl.load(c_ptr + off, mask=ok, other=0).to(tl.int64) + \
        (tl.load(c_ptr + M * N2 + off, mask=ok, other=0).to(tl.int64) << 8)
    t = _rsh(acc * tl.load(m_ptr + col, mask=ok, other=0), fmsh)
    return t, _rsh(t * mant, 24 - FRC - e)


@triton.jit
def _h_kernel(c_ptr, m_ptr, fmsh_ptr, mant_ptr, e_ptr, lut_ptr, h_ptr, part_ptr, M, NCH,
              FF: tl.constexpr, CH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    ch = tl.program_id(1)
    fmsh = tl.load(fmsh_ptr)
    mant = tl.load(mant_ptr + row)
    e = tl.load(e_ptr + row)
    mt = tl.zeros([BLOCK], tl.int64)
    ms = tl.zeros([BLOCK], tl.int64)
    mu = tl.zeros([BLOCK], tl.int64)
    mh = tl.zeros([BLOCK], tl.int64)
    for c0 in range(ch * CH, ch * CH + CH, BLOCK):
        col = c0 + tl.arange(0, BLOCK)
        ok = col < FF
        tg, g = _y(c_ptr, m_ptr, col, ok, row, M, 2 * FF, fmsh, mant, e)
        tu, u = _y(c_ptr, m_ptr, col + FF, ok, row, M, 2 * FF, fmsh, mant, e)
        sg = _silu(g, lut_ptr)
        h = (sg * u + (1 << (FRC - 1))) >> FRC
        tl.store(h_ptr + row.to(tl.int64) * FF + col, h, mask=ok)
        mt = tl.maximum(mt, tl.where(ok, tl.maximum(tl.abs(tg), tl.abs(tu)), 0))
        ms = tl.maximum(ms, tl.where(ok, tl.abs(sg), 0))
        mu = tl.maximum(mu, tl.where(ok, tl.abs(u), 0))
        mh = tl.maximum(mh, tl.where(ok, tl.abs(h), 0))
    p = part_ptr + row * NCH + ch
    tl.store(p, tl.max(mt, 0))
    tl.store(p + M * NCH, tl.max(ms, 0))
    tl.store(p + 2 * M * NCH, tl.max(mu, 0))
    tl.store(p + 3 * M * NCH, tl.max(mh, 0))


@triton.jit
def _quant_kernel(h_ptr, part_ptr, q_ptr, mant_ptr, e_ptr, rmax_ptr, M, NCH,
                  FF: tl.constexpr, CH: tl.constexpr, BLOCK: tl.constexpr, NB: tl.constexpr, ATOMIC: tl.constexpr):
    row = tl.program_id(0)
    ch = tl.program_id(1)
    k = tl.arange(0, NB)
    kok = k < NCH
    amax = tl.maximum(tl.max(tl.load(part_ptr + 3 * M * NCH + row * NCH + k, mask=kok, other=0), 0), 1)
    sh = _bitlen(amax) - 30                                  # quant_rows
    as_ = _rsh(amax, sh)
    d = 2 * as_
    for c0 in range(ch * CH, ch * CH + CH, BLOCK):
        col = c0 + tl.arange(0, BLOCK)
        ok = col < FF
        zs = _rsh(tl.load(h_ptr + row.to(tl.int64) * FF + col, mask=ok, other=0), sh)
        n = 2 * (zs * QMAX) + as_                            # rdiv32(zs * qmax, as_)
        q = tl.floor(n.to(tl.float32) / d.to(tl.float32)).to(tl.int64)
        q = tl.where(q * d > n, q - 1, q)
        q = tl.where((q + 1) * d <= n, q + 1, q)
        lo = ((q + 128) & 255) - 128                         # stack_limbs(q, 2)
        hi = ((((q - lo) >> 8) + 128) & 255) - 128
        tl.store(q_ptr + row.to(tl.int64) * FF + col, lo.to(tl.int8), mask=ok)
        tl.store(q_ptr + (M + row).to(tl.int64) * FF + col, hi.to(tl.int8), mask=ok)
    if ch == 0:
        b = _bitlen(as_) - 24                                # norm24(as_, sh - FR)
        tl.store(mant_ptr + row, _rsh(as_, b))
        tl.store(e_ptr + row, sh - FRC + b)
        for j in tl.static_range(3):
            mj = tl.max(tl.load(part_ptr + j * M * NCH + row * NCH + k, mask=kok, other=0), 0)
            if ATOMIC:                                       # maxima over rows of |sg|, |u| (j = 1, 2)
                if j > 0:
                    tl.atomic_max(rmax_ptr + (j - 1), mj)
            else:
                tl.store(rmax_ptr + j * M + row, mj)


def silu_quant(c, m, fmsh, mant, e, lut, check, block=512, scr=None):
    """c = torch._int_mm(2-limb input, Wgu^T) (int32, >= 2M rows), m / fmsh / mant / e as in segs.post
    -> (a8, mant', e', flag) as segs.silu_mul(segs.post(...)) with stack_limbs (flag also covers post's
    when check). scr (int64 [2], zeroed): the row maxima of |sg|, |u| go there as atomic maxima and no flag is
    computed (flag None; the caller multiplies them, tri_fuse.flag_end); check must be False."""
    M, FF = mant.shape[0], c.shape[1] // 2
    dev = c.device
    nch = max(1, min(triton.cdiv(FF, block), triton.cdiv(528, max(M, 1))))
    ch = triton.cdiv(triton.cdiv(FF, nch), block) * block
    nch = triton.cdiv(FF, ch)
    rows = 2 * M
    pad = max(24, (rows + 7) // 8 * 8) - rows
    q = torch.empty(rows + pad, FF, dtype=torch.int8, device=dev)
    q[rows:].zero_()
    h = torch.empty(M, FF, dtype=torch.long, device=dev)
    part = torch.empty(4, M, nch, dtype=torch.long, device=dev)
    mant2 = torch.empty(M, dtype=torch.long, device=dev)
    e2 = torch.empty_like(mant2)
    rmax = torch.empty(3, M, dtype=torch.long, device=dev)
    if M:
        _h_kernel[(M, nch)](c, m, fmsh, mant, e, lut, h, part, M, nch, FF=FF, CH=ch, BLOCK=block, num_warps=4)
        _quant_kernel[(M, nch)](h, part, q, mant2, e2, rmax if scr is None else scr, M, nch, FF=FF, CH=ch,
                                BLOCK=block, NB=triton.next_power_of_2(nch), ATOMIC=scr is not None, num_warps=4)
    if scr is not None:
        assert not check
        return q, mant2, e2, None
    mx = rmax.amax(1) if M else torch.zeros(3, dtype=torch.long, device=dev)
    flag = mx[1] * mx[2] >= 1 << 62
    if check:
        flag = flag | (mx[0] >= 1 << 38)
    return q, mant2, e2, flag
