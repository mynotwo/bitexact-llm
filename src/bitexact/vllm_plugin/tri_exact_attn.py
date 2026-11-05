"""Exact-mode attention on vLLM's paged limb-plane KV cache (same cache layout as tri_paged).

For queries of one KV head (M <= 4 = G query heads of one token), reads K / V in place through the block
table and computes what exact/model.py computes unfused (bmm, ops.softmax_w, bmm):

  raw[i, s] = sum_d q[i, d] K[s, d]              q int32 (|q| <= 2^22, per-row exponent sh[i]), K at 17 bits
  mx[i]     = max over s <= qpos[i] of raw[i, s]
  zf        = min(rshc(mx - raw, 17 + 3 - sh[i] - 26), 48 << 26)
  t         = rshc(zf * L2E, 28);  p = exp2neg_lut(t, lut, 22)   (table 2^-f + linear interpolation)
  den[i]    = sum_s p,   num[i, d] = sum_s p V[s, d]             V at 19 bits

All reductions are exact integer sums / maxima, so the key split cannot change a bit. The 3 limbs of the
M queries are stacked as rows of one 16-row int8 operand (as tri_paged._dec_*): per dot entry
|sum| <= 64 * 2^14 (q.K) or keys * 2^14 (p.V, asserted < 43690 keys), so the int32 partials are exact.
"""

import torch
import triton
import triton.language as tl

from bitexact.exact.consts import EXP2_BITS as _EXP2_BITS
from bitexact.exact.consts import EXP2_ONE as _EXP2_ONE
from bitexact.exact.ops import FT2 as _FT2
from bitexact.exact.ops import FZ as _FZ
from bitexact.exact.segs import FK as _FK
from bitexact.exact.segs import FP as _FP
from bitexact.vllm_plugin.tri_paged import NEG, _dec_scores, _load_v, _stack, _unstack

BQ = tl.constexpr(4)
DEC_PROGRAMS = 528
EXP2_BITS, EXP2_ONE, FT2, FZ, FK, FP = (tl.constexpr(v) for v in (_EXP2_BITS, _EXP2_ONE, _FT2, _FZ, _FK, _FP))
ZMAX = tl.constexpr(48 << _FZ)
TSH = tl.constexpr(_FZ + 28 - _FT2)                  # t = rshc(zf * L2E, TSH)


@triton.jit
def _xtile(q_ptr, sh_ptr, qpos_ptr, bt_ptr, rmap_ptr, M, sqz, sqm, spz, sbt, NSPLIT,
           NHKV: tl.constexpr, DH: tl.constexpr, BN: tl.constexpr):
    pz = tl.program_id(0).to(tl.int64)                 # (row, KV head): grid axis 0 holds up to 2^31 programs
    sp = tl.program_id(1)
    r = pz // NHKV
    rm = tl.arange(0, BQ)
    rd = tl.arange(0, DH)
    mok = rm < M
    qpos = tl.load(qpos_ptr + r * spz + rm, mask=mok, other=-1).to(tl.int64)
    rq = tl.load(q_ptr + pz * sqz + rm[:, None] * sqm + rd[None, :], mask=mok[:, None], other=0).to(tl.int32)
    sh = tl.load(sh_ptr + pz * M + rm, mask=mok, other=0)
    kend = tl.max(qpos, axis=0) + 1
    chunk = tl.cdiv(tl.cdiv(kend, NSPLIT), BN) * BN
    lo = sp * chunk
    hi = tl.minimum(lo + chunk, kend)
    brow = tl.load(rmap_ptr + r).to(tl.int64)
    return pz, pz % NHKV, rm, rd, mok, qpos, _stack(rq, BQ), sh, lo, hi, bt_ptr + brow * sbt


@triton.jit
def _rowshift(sh):
    """Per-row shift of z -> zf (ops.rshc with the tensor shift s = FK + 3 - sh - FZ clamped to +-62):
    (rounding half, right shift, left shift)."""
    s = tl.minimum(tl.maximum(FK + 3 - FZ - sh, -62), 62)
    pos = tl.maximum(s, 0)
    one = tl.full((BQ,), 1, dtype=tl.int64)
    half = tl.where(pos > 0, one << tl.maximum(pos - 1, 0), 0)
    return half, pos, tl.maximum(-s, 0)


@triton.jit
def _weights(raw, valid, mx, half, pos, neg, lut_ptr, L2E: tl.constexpr):
    z = tl.where(valid, mx[:, None] - raw, 0)
    zf = tl.minimum(((z + half[:, None]) >> pos[:, None]) << neg[:, None], ZMAX)
    t = (zf * L2E + (1 << (TSH - 1))) >> TSH
    n = t >> FT2
    f = t & ((1 << FT2) - 1)
    idx = f >> (FT2 - EXP2_BITS)
    rr = f & ((1 << (FT2 - EXP2_BITS)) - 1)
    a = tl.load(lut_ptr + idx)
    b = tl.load(lut_ptr + idx + 1)
    v = a + (((b - a) * rr + (1 << (FT2 - EXP2_BITS - 1))) >> (FT2 - EXP2_BITS))
    s = tl.minimum(EXP2_ONE - FP + n, 62)
    p = (v + (tl.full(v.shape, 1, dtype=tl.int64) << (s - 1))) >> s
    return tl.where(valid, p, 0)


@triton.jit
def _xmax_kernel(q_ptr, sh_ptr, qpos_ptr, c_ptr, bt_ptr, rmap_ptr, mx_ptr, M, NSPLIT,
                 sqz, sqm, spz, sc_nb, sc_h, sc_bs, sbt,
                 BS: tl.constexpr, NHKV: tl.constexpr, DH: tl.constexpr, BN: tl.constexpr):
    pz, h, rm, rd, mok, qpos, qs, sh, lo, hi, bt_ptr = _xtile(
        q_ptr, sh_ptr, qpos_ptr, bt_ptr, rmap_ptr, M, sqz, sqm, spz, sbt, NSPLIT, NHKV, DH, BN)
    cbase = c_ptr + h * sc_h
    mx = tl.full((BQ,), NEG, dtype=tl.int64)
    for k0 in tl.range(lo, hi, BN, num_stages=1):
        raw, valid, off, kok = _dec_scores(qs, cbase, bt_ptr, k0, hi, qpos, rd, sc_nb, sc_bs, BS, DH, BQ, BN)
        mx = tl.maximum(mx, tl.max(tl.where(valid, raw, NEG), axis=1))
    tl.atomic_max(mx_ptr + pz * M + rm, mx, mask=mok & (hi > lo))


@triton.jit
def _xacc_kernel(q_ptr, sh_ptr, qpos_ptr, c_ptr, bt_ptr, rmap_ptr, lut_ptr, mx_ptr, num_ptr, den_ptr, M, NSPLIT,
                 sqz, sqm, spz, sc_nb, sc_h, sc_bs, sbt, snz, snm,
                 BS: tl.constexpr, NHKV: tl.constexpr, DH: tl.constexpr, L2E: tl.constexpr,
                 BN: tl.constexpr, FUSED: tl.constexpr):
    pz, h, rm, rd, mok, qpos, qs, sh, lo, hi, bt_ptr = _xtile(
        q_ptr, sh_ptr, qpos_ptr, bt_ptr, rmap_ptr, M, sqz, sqm, spz, sbt, NSPLIT, NHKV, DH, BN)
    cbase = c_ptr + h * sc_h
    if FUSED:
        mx = tl.full((BQ,), NEG, dtype=tl.int64)
        for k0 in tl.range(lo, hi, BN, num_stages=1):
            raw, valid, off, kok = _dec_scores(qs, cbase, bt_ptr, k0, hi, qpos, rd, sc_nb, sc_bs, BS, DH, BQ, BN)
            mx = tl.maximum(mx, tl.max(tl.where(valid, raw, NEG), axis=1))
    else:
        mx = tl.load(mx_ptr + pz * M + rm, mask=mok, other=0)
    half, pos, neg = _rowshift(sh)
    e0 = tl.zeros((4 * BQ, DH), dtype=tl.int32)
    e1 = tl.zeros((4 * BQ, DH), dtype=tl.int32)
    e2 = tl.zeros((4 * BQ, DH), dtype=tl.int32)
    den = tl.zeros((BQ,), dtype=tl.int64)
    for k0 in tl.range(lo, hi, BN, num_stages=1):
        raw, valid, off, kok = _dec_scores(qs, cbase, bt_ptr, k0, hi, qpos, rd, sc_nb, sc_bs, BS, DH, BQ, BN)
        p = _weights(raw, valid, mx, half, pos, neg, lut_ptr, L2E)
        den += tl.sum(p, axis=1)
        ps = _stack(p.to(tl.int32), BQ)
        v0, v1, v2 = _load_v(cbase, off, kok, rd, DH)
        e0 += tl.dot(ps, v0, out_dtype=tl.int32)
        e1 += tl.dot(ps, v1, out_dtype=tl.int32)
        e2 += tl.dot(ps, v2, out_dtype=tl.int32)
    num = _unstack(e0, e1, e2, BQ)
    out = num_ptr + pz * snz + rm[:, None] * snm + rd[None, :]
    if FUSED:                                    # sole writer of these rows (zeros when no key: hi == lo)
        tl.store(out, num, mask=mok[:, None])
        tl.store(den_ptr + pz * M + rm, den, mask=mok)
    else:
        any_k = hi > lo
        tl.atomic_add(out, num, mask=mok[:, None] & any_k)
        tl.atomic_add(den_ptr + pz * M + rm, den, mask=mok & any_k)


@torch.compiler.disable
def attend(q, sh, qpos, cache, bt, rmap, lut, l2e, nsplit=None, bn=64, warps=4):
    """q int32 [R, HKV, M, HD] (M <= 4, |q| <= 2^22), sh int64 [R, HKV, M] (per-row exponent of q), qpos int64
    [R, M] (key s visible to query i iff s <= qpos[r, i]), cache int8 limb planes [nb, HKV, bs, 8 HD],
    bt [*, nblk] block table, rmap int64 [R] (block-table row of each q row), lut = exp2 table (int64),
    l2e = ops.L2E28_HD[HD]. Returns num int64 [R, HKV, M, HD], den int64 [R, HKV, M, 1]."""
    R, HKV, M, HD = q.shape
    assert M <= BQ.value and cache.shape[1] == HKV and cache.shape[3] == 8 * HD and cache.stride(3) == 1
    assert bt.shape[1] * cache.shape[2] < 43690
    q, sh, qpos = q.contiguous(), sh.contiguous(), qpos.contiguous()
    if nsplit is None:
        tiles = R * HKV
        nsplit = max(1, min(64, DEC_PROGRAMS // tiles, (bt.shape[1] * cache.shape[2] // 2048) if R > 8 else 64))
    fused = nsplit == 1
    new = torch.empty if fused else torch.zeros
    num = new(R, HKV, M, HD, dtype=torch.long, device=q.device)
    den = new(R, HKV, M, 1, dtype=torch.long, device=q.device)
    mx = num if fused else torch.full((R, HKV, M), NEG.value, dtype=torch.long, device=q.device)
    grid = (R * HKV, nsplit)
    common = (q.stride(1), q.stride(2), qpos.stride(0), cache.stride(0), cache.stride(1), cache.stride(2),
              bt.stride(0))
    if not fused:
        _xmax_kernel[grid](q, sh, qpos, cache, bt, rmap, mx, M, nsplit, *common,
                           BS=cache.shape[2], NHKV=HKV, DH=HD, BN=bn, num_warps=warps)
    _xacc_kernel[grid](q, sh, qpos, cache, bt, rmap, lut, mx, num, den, M, nsplit, *common, num.stride(1),
                       num.stride(2), BS=cache.shape[2], NHKV=HKV, DH=HD, L2E=l2e, BN=bn, FUSED=fused,
                       num_warps=warps)
    return num, den
