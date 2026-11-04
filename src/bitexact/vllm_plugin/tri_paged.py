"""Fused exact integer attention on vLLM's paged int32 KV cache.

Triton programs per (query tile, request, KV head, key split). They read K / V in place through the
block table (key s of request r, head h: cache[bt[r, s // bs], h, s % bs, :HD] = K, [HD:] = V; no
gathered copy, no score tensor in memory) and compute exactly what the int8 engine computes unfused
(int8.segs.softmax and bitexact.bmm):

  raw[i, s] = sum_d q[i, d] K[s, d]                     int8 limbs on tensor cores, int32 accumulators
  valid     = s <= qpos[i]
  mx[i]     = max over valid s of raw[i, s]             (kernel 1)
  z         = rsh(raw - mx, 2 FQ + 3 - FA)              (kernel 2, raw recomputed)
  p         = rsh(exp_lut[clamp(z) - EXP_MIN << FA], FE - FP)   (0 below the table, 0 when not valid)
  den[i]    = sum_s p,   num[i, d] = sum_s p V[s, d]    (limbs again; per tile |int32 sum| < 2^31)

All reductions are exact integer sums and maxima, so tiling, splits and order cannot change a bit.
Why two sweeps over K and not one online-softmax sweep: the engine quantizes each weight against the final
row maximum (z = rsh(raw - mx), then the exp table). Rescaling partial sums when the maximum grows would
need exp(a) * exp(b) == exp(a + b) on the quantized table, which does not hold, so a single sweep could
not reproduce the engine's integers. Kernel 1 reads only K (no exp, no V).
Keys are visited only up to the largest query position of the tile (read on the device), so the
kernels need no host-side sequence lengths and are CUDA-graph safe.
Cache layout: int8 limb planes per (block, KV head, slot) row of 512 bytes: K limbs 0, 1, 2 at bytes
[0, 64), [64, 128), [128, 192), V limbs at [192, 384) (the rest is unused; the row stays the 512 bytes vLLM
allocates). Limb i is the signed base-256 digit _split would produce of the int32 K / V value, so the kernels
read 384 instead of 512 bytes per key and skip the splitting. `store` writes the planes.
Returns num (int64 [R, HKV, M, HD]) and den (int64 [R, HKV, M, 1]); the caller rounds num / den.
"""

import torch
import triton
import triton.language as tl

from bitexact.int8.consts import EXP_MIN, FA, FE
from bitexact.int8.segs import FP, FQ, ZCLAMP, score_rule

NEG = tl.constexpr(-(1 << 62))
ZCLAMP_C = tl.constexpr(ZCLAMP)


@triton.jit
def _split(r):
    lo = ((r + 128) & 255) - 128
    return lo.to(tl.int8), (r - lo) >> 8


@triton.jit
def _limb_dot(a0, a1, a2, b0, b1, b2):
    """Exact a @ b, a and b given as 3 int8 limbs each -> int64."""
    c0 = tl.dot(a0, b0, out_dtype=tl.int32)
    c1 = tl.dot(a1, b0, out_dtype=tl.int32) + tl.dot(a0, b1, out_dtype=tl.int32)
    c2 = tl.dot(a2, b0, out_dtype=tl.int32) + tl.dot(a1, b1, out_dtype=tl.int32) + \
        tl.dot(a0, b2, out_dtype=tl.int32)
    c3 = tl.dot(a2, b1, out_dtype=tl.int32) + tl.dot(a1, b2, out_dtype=tl.int32)
    c4 = tl.dot(a2, b2, out_dtype=tl.int32)
    return c0.to(tl.int64) + (c1.to(tl.int64) << 8) + (c2.to(tl.int64) << 16) \
        + (c3.to(tl.int64) << 24) + (c4.to(tl.int64) << 32)


@triton.jit
def _limb_parts(a0, a1, a2, b0, b1, b2):
    """The five int32 limb-weight partial sums of a @ b (weights 2^0, 2^8, ..., 2^32)."""
    c0 = tl.dot(a0, b0, out_dtype=tl.int32)
    c1 = tl.dot(a1, b0, out_dtype=tl.int32) + tl.dot(a0, b1, out_dtype=tl.int32)
    c2 = tl.dot(a2, b0, out_dtype=tl.int32) + tl.dot(a1, b1, out_dtype=tl.int32) + \
        tl.dot(a0, b2, out_dtype=tl.int32)
    c3 = tl.dot(a2, b1, out_dtype=tl.int32) + tl.dot(a1, b2, out_dtype=tl.int32)
    c4 = tl.dot(a2, b2, out_dtype=tl.int32)
    return c0, c1, c2, c3, c4


@triton.jit
def _load_k(cbase, off, kok, rd, DH: tl.constexpr):
    """The 3 int8 limb planes of the keys at row offsets off ([BN]) -> three [DH, BN] int8 tiles."""
    p = cbase + off[None, :] + rd[:, None]
    m = kok[None, :]
    return tl.load(p, mask=m, other=0), tl.load(p + DH, mask=m, other=0), tl.load(p + 2 * DH, mask=m, other=0)


@triton.jit
def _load_v(cbase, off, kok, rd, DH: tl.constexpr):
    """The 3 int8 limb planes of the values -> three [BN, DH] int8 tiles."""
    p = cbase + off[:, None] + 3 * DH + rd[None, :]
    m = kok[:, None]
    return tl.load(p, mask=m, other=0), tl.load(p + DH, mask=m, other=0), tl.load(p + 2 * DH, mask=m, other=0)


@triton.jit
def _rsh(x, s: tl.constexpr):
    return (x + (1 << (s - 1))) >> s


@triton.jit
def _zscale(d, ZSH: tl.constexpr, ZMUL: tl.constexpr):
    """z = int8.segs.softmax's score scale of d <= 0: rsh(d, ZSH), or (head_dim 128, ZMUL != 0) rsh(max(d, ZCLAMP) * ZMUL,
    ZSH); the clamp keeps |d * ZMUL| < 2^62 and sends only values far below the exp table (z < -32) to z < -32."""
    if ZMUL == 0:
        return _rsh(d, ZSH)
    return _rsh(tl.maximum(d, ZCLAMP_C) * ZMUL, ZSH)


@triton.jit
def _tile(q_ptr, qpos_ptr, bt_ptr, M, sqz, sqm, spz, sbt, NSPLIT,
          NHKV: tl.constexpr, DH: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """Query tile setup and this program's key range [lo, hi) (split of [0, kend) in BN multiples)."""
    pm = tl.program_id(0)
    pz = tl.program_id(1).to(tl.int64)
    sp = tl.program_id(2)
    r = pz // NHKV
    h = pz % NHKV
    rm = pm * BM + tl.arange(0, BM)
    rd = tl.arange(0, DH)
    mok = rm < M
    qpos = tl.load(qpos_ptr + r * spz + rm, mask=mok, other=-1).to(tl.int64)
    rq = tl.load(q_ptr + pz * sqz + rm[:, None] * sqm + rd[None, :], mask=mok[:, None], other=0).to(tl.int32)
    a0, rq = _split(rq)
    a1, rq = _split(rq)
    a2, rq = _split(rq)
    kend = tl.max(qpos, axis=0) + 1
    chunk = tl.cdiv(tl.cdiv(kend, NSPLIT), BN) * BN
    lo = sp * chunk
    hi = tl.minimum(lo + chunk, kend)
    return pz, h, rm, rd, mok, qpos, a0, a1, a2, lo, hi, bt_ptr + r * sbt


@triton.jit
def _scores(a0, a1, a2, cbase, bt_ptr, k0, hi, qpos, rd, sc_nb, sc_bs,
            BS: tl.constexpr, DH: tl.constexpr, BN: tl.constexpr):
    ks = k0 + tl.arange(0, BN)
    kok = ks < hi
    blk = tl.load(bt_ptr + ks // BS, mask=kok, other=0).to(tl.int64)
    off = blk * sc_nb + (ks % BS) * sc_bs
    b0, b1, b2 = _load_k(cbase, off, kok, rd, DH)
    raw = _limb_dot(a0, a1, a2, b0, b1, b2)
    valid = (ks[None, :] <= qpos[:, None]) & kok[None, :]
    return raw, valid, off, kok


@triton.jit
def _max_kernel(q_ptr, qpos_ptr, c_ptr, bt_ptr, mx_ptr, M, NSPLIT,
                sqz, sqm, spz, sc_nb, sc_h, sc_bs, sbt,
                BS: tl.constexpr, NHKV: tl.constexpr, DH: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                SPLIT: tl.constexpr):
    pz, h, rm, rd, mok, qpos, a0, a1, a2, lo, hi, bt_ptr = _tile(
        q_ptr, qpos_ptr, bt_ptr, M, sqz, sqm, spz, sbt, NSPLIT, NHKV, DH, BM, BN)
    cbase = c_ptr + h * sc_h
    mx = tl.full((BM,), NEG, dtype=tl.int64)
    for k0 in tl.range(lo, hi, BN, num_stages=1):
        raw, valid, off, kok = _scores(a0, a1, a2, cbase, bt_ptr, k0, hi, qpos, rd, sc_nb, sc_bs, BS, DH, BN)
        mx = tl.maximum(mx, tl.max(tl.where(valid, raw, NEG), axis=1))
    if SPLIT:
        tl.atomic_max(mx_ptr + pz * M + rm, mx, mask=mok & (hi > lo))
    else:                                       # one split: the program owns the row; NEG when no key is visible
        tl.store(mx_ptr + pz * M + rm, mx, mask=mok)


@triton.jit
def _acc_kernel(q_ptr, qpos_ptr, c_ptr, bt_ptr, ptab_ptr, mx_ptr, num_ptr, den_ptr, M, NSPLIT,
                sqz, sqm, spz, sc_nb, sc_h, sc_bs, sbt, snz, snm,
                BS: tl.constexpr, NHKV: tl.constexpr, DH: tl.constexpr, ZSH: tl.constexpr, ZMUL: tl.constexpr,
                ZLO: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr):
    pz, h, rm, rd, mok, qpos, a0, a1, a2, lo, hi, bt_ptr = _tile(
        q_ptr, qpos_ptr, bt_ptr, M, sqz, sqm, spz, sbt, NSPLIT, NHKV, DH, BM, BN)
    cbase = c_ptr + h * sc_h
    mx = tl.load(mx_ptr + pz * M + rm, mask=mok, other=0)
    # p.v limb partials stay int32 over the whole key range: per tile |sum| <= 3 * BN * 2^14, so up to
    # 2^31 / (3 * 2^14) = 43690 keys (asserted by the caller).
    n0 = tl.zeros((BM, DH), dtype=tl.int32)
    n1 = tl.zeros((BM, DH), dtype=tl.int32)
    n2 = tl.zeros((BM, DH), dtype=tl.int32)
    n3 = tl.zeros((BM, DH), dtype=tl.int32)
    n4 = tl.zeros((BM, DH), dtype=tl.int32)
    den = tl.zeros((BM,), dtype=tl.int64)
    for k0 in tl.range(lo, hi, BN, num_stages=1):
        raw, valid, off, kok = _scores(a0, a1, a2, cbase, bt_ptr, k0, hi, qpos, rd, sc_nb, sc_bs, BS, DH, BN)
        z = _zscale(tl.where(valid, raw, NEG) - mx[:, None], ZSH, ZMUL)   # <= 0
        idx = (tl.maximum(z, ZLO - 1) - ZLO).to(tl.int32)               # table index, -1 = below table
        ok = valid & (idx >= 0)
        p = tl.load(ptab_ptr + tl.maximum(idx, 0), mask=ok, other=0)    # int32, = rsh(exp, FE - FP)
        den += tl.sum(p, axis=1).to(tl.int64)
        p0, p = _split(p)
        p1, p = _split(p)
        p2, p = _split(p)
        v0, v1, v2 = _load_v(cbase, off, kok, rd, DH)
        c0, c1, c2, c3, c4 = _limb_parts(p0, p1, p2, v0, v1, v2)
        n0 += c0
        n1 += c1
        n2 += c2
        n3 += c3
        n4 += c4
    num = n0.to(tl.int64) + (n1.to(tl.int64) << 8) + (n2.to(tl.int64) << 16) \
        + (n3.to(tl.int64) << 24) + (n4.to(tl.int64) << 32)
    if SPLIT:
        any_k = hi > lo
        tl.atomic_add(num_ptr + pz * snz + rm[:, None] * snm + rd[None, :], num, mask=mok[:, None] & any_k)
        tl.atomic_add(den_ptr + pz * M + rm, den, mask=mok & any_k)
    else:                                       # one split: plain stores; num / den are 0 when no key was visible
        tl.store(num_ptr + pz * snz + rm[:, None] * snm + rd[None, :], num, mask=mok[:, None])
        tl.store(den_ptr + pz * M + rm, den, mask=mok)


# Decode (M <= BQ = 4 queries per request and KV head): the 3 limbs of the BQ queries are stacked as the rows
# of one 16-row int8 operand (row i * BQ + g = limb i of query g, rows 3 * BQ.. zero), so q.K takes 3 tensor-core
# dots (one per K limb) instead of 9 with 12 of 16 rows padding; p.V likewise. Per dot entry |sum| <= DH * 2^14
# (q.K) or keys * 2^14 (p.V, < 2^31 for the asserted < 43690 keys), so the int32 partials are exact; they are
# combined as int64 sum_ij a_i b_j 2^(8 (i + j)), the same polynomial in the same int8 limbs as _limb_dot /
# _limb_parts (int64 adds and shifts are exact mod 2^64 and the result is in range), hence the same integers.
@triton.jit
def _stack(x, BQ: tl.constexpr):
    """[BQ, N] int32 (3 limbs) -> [4 BQ, N] int8, row i * BQ + g = limb i of x[g] (rows 3 BQ.. zero)."""
    N: tl.constexpr = x.shape[1]
    x = tl.reshape(tl.broadcast_to(x[None, :, :], (4, BQ, N)), (4 * BQ, N))
    i = (tl.arange(0, 4 * BQ) // BQ)[:, None]
    l0, x = _split(x)
    l1, x = _split(x)
    l2, x = _split(x)
    return tl.where(i == 0, l0, tl.where(i == 1, l1, tl.where(i == 2, l2, 0))).to(tl.int8)


@triton.jit
def _unstack(c0, c1, c2, BQ: tl.constexpr):
    """Stacked int32 dots with B limbs 0, 1, 2 ([4 BQ, N]) -> [BQ, N] int64 sum_ij c_j[i BQ + g] 2^(8 (i + j))."""
    N: tl.constexpr = c0.shape[1]
    t = c0.to(tl.int64) + (c1.to(tl.int64) << 8) + (c2.to(tl.int64) << 16)
    t = t << (8 * (tl.arange(0, 4 * BQ) // BQ)).to(tl.int64)[:, None]
    return tl.sum(tl.reshape(t, (4, BQ, N)), axis=0)


@triton.jit
def _dec_tile(q_ptr, qpos_ptr, bt_ptr, M, sqz, sqm, spz, sbt, NSPLIT,
              NHKV: tl.constexpr, DH: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr):
    pz = tl.program_id(1).to(tl.int64)
    sp = tl.program_id(2)
    r = pz // NHKV
    rm = tl.arange(0, BQ)
    rd = tl.arange(0, DH)
    mok = rm < M
    qpos = tl.load(qpos_ptr + r * spz + rm, mask=mok, other=-1).to(tl.int64)
    rq = tl.load(q_ptr + pz * sqz + rm[:, None] * sqm + rd[None, :], mask=mok[:, None], other=0).to(tl.int32)
    kend = tl.max(qpos, axis=0) + 1
    chunk = tl.cdiv(tl.cdiv(kend, NSPLIT), BN) * BN
    lo = sp * chunk
    hi = tl.minimum(lo + chunk, kend)
    return pz, pz % NHKV, rm, rd, mok, qpos, _stack(rq, BQ), lo, hi, bt_ptr + r * sbt


@triton.jit
def _dec_scores(qs, cbase, bt_ptr, k0, hi, qpos, rd, sc_nb, sc_bs,
                BS: tl.constexpr, DH: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr):
    ks = k0 + tl.arange(0, BN)
    kok = ks < hi
    blk = tl.load(bt_ptr + ks // BS, mask=kok, other=0).to(tl.int64)
    off = blk * sc_nb + (ks % BS) * sc_bs
    b0, b1, b2 = _load_k(cbase, off, kok, rd, DH)
    raw = _unstack(tl.dot(qs, b0, out_dtype=tl.int32), tl.dot(qs, b1, out_dtype=tl.int32),
                   tl.dot(qs, b2, out_dtype=tl.int32), BQ)
    valid = (ks[None, :] <= qpos[:, None]) & kok[None, :]
    return raw, valid, off, kok


@triton.jit
def _dec_max_kernel(q_ptr, qpos_ptr, c_ptr, bt_ptr, mx_ptr, M, NSPLIT,
                    sqz, sqm, spz, sc_nb, sc_h, sc_bs, sbt,
                    BS: tl.constexpr, NHKV: tl.constexpr, DH: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr):
    pz, h, rm, rd, mok, qpos, qs, lo, hi, bt_ptr = _dec_tile(
        q_ptr, qpos_ptr, bt_ptr, M, sqz, sqm, spz, sbt, NSPLIT, NHKV, DH, BQ, BN)
    cbase = c_ptr + h * sc_h
    mx = tl.full((BQ,), NEG, dtype=tl.int64)
    for k0 in tl.range(lo, hi, BN, num_stages=1):
        raw, valid, off, kok = _dec_scores(qs, cbase, bt_ptr, k0, hi, qpos, rd, sc_nb, sc_bs, BS, DH, BQ, BN)
        mx = tl.maximum(mx, tl.max(tl.where(valid, raw, NEG), axis=1))
    tl.atomic_max(mx_ptr + pz * M + rm, mx, mask=mok & (hi > lo))


@triton.jit
def _dec_acc_kernel(q_ptr, qpos_ptr, c_ptr, bt_ptr, ptab_ptr, mx_ptr, num_ptr, den_ptr, M, NSPLIT,
                    sqz, sqm, spz, sc_nb, sc_h, sc_bs, sbt, snz, snm,
                    BS: tl.constexpr, NHKV: tl.constexpr, DH: tl.constexpr, ZSH: tl.constexpr, ZMUL: tl.constexpr,
                    ZLO: tl.constexpr, BQ: tl.constexpr, BN: tl.constexpr, FUSED: tl.constexpr):
    """FUSED (NSPLIT == 1): this program sees all keys of its rows, so it takes the row maxima itself in a
    first sweep (the second sweep re-reads K, mostly from L2) and stores num / den; otherwise mx comes from
    _dec_max_kernel and the per-split sums are added atomically."""
    pz, h, rm, rd, mok, qpos, qs, lo, hi, bt_ptr = _dec_tile(
        q_ptr, qpos_ptr, bt_ptr, M, sqz, sqm, spz, sbt, NSPLIT, NHKV, DH, BQ, BN)
    cbase = c_ptr + h * sc_h
    if FUSED:
        mx = tl.full((BQ,), NEG, dtype=tl.int64)
        for k0 in tl.range(lo, hi, BN, num_stages=1):
            raw, valid, off, kok = _dec_scores(qs, cbase, bt_ptr, k0, hi, qpos, rd, sc_nb, sc_bs, BS, DH, BQ, BN)
            mx = tl.maximum(mx, tl.max(tl.where(valid, raw, NEG), axis=1))
    else:
        mx = tl.load(mx_ptr + pz * M + rm, mask=mok, other=0)
    e0 = tl.zeros((4 * BQ, DH), dtype=tl.int32)
    e1 = tl.zeros((4 * BQ, DH), dtype=tl.int32)
    e2 = tl.zeros((4 * BQ, DH), dtype=tl.int32)
    den = tl.zeros((BQ,), dtype=tl.int64)
    for k0 in tl.range(lo, hi, BN, num_stages=1):
        raw, valid, off, kok = _dec_scores(qs, cbase, bt_ptr, k0, hi, qpos, rd, sc_nb, sc_bs, BS, DH, BQ, BN)
        z = _zscale(tl.where(valid, raw, NEG) - mx[:, None], ZSH, ZMUL)   # <= 0
        idx = (tl.maximum(z, ZLO - 1) - ZLO).to(tl.int32)               # table index, -1 = below table
        ok = valid & (idx >= 0)
        p = tl.load(ptab_ptr + tl.maximum(idx, 0), mask=ok, other=0)    # int32, = rsh(exp, FE - FP)
        den += tl.sum(p, axis=1).to(tl.int64)
        ps = _stack(p, BQ)
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


@triton.jit
def _store_kernel(c_ptr, slot_ptr, k_ptr, v_ptr, skz, skh, svz, svh, sc_nb, sc_h, sc_bs,
                  BS: tl.constexpr, DH: tl.constexpr):
    n = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    slot = tl.load(slot_ptr + n)
    rd = tl.arange(0, DH)
    base = c_ptr + (slot // BS) * sc_nb + h * sc_h + (slot % BS) * sc_bs
    k = tl.load(k_ptr + n * skz + h * skh + rd).to(tl.int32)
    v = tl.load(v_ptr + n * svz + h * svh + rd).to(tl.int32)
    ok = slot >= 0                                    # dummy / padded tokens carry slot -1: not stored
    for i in tl.static_range(3):
        k0, k = _split(k)
        v0, v = _split(v)
        tl.store(base + i * DH + rd, k0, mask=ok)
        tl.store(base + (3 + i) * DH + rd, v0, mask=ok)


def store(c, slot, k, v):
    """Write K / V (int64 [N, HKV, HD], |.| within 3 limbs) of the tokens at cache slots slot [N] into the
    limb-plane cache c (int8 [nb, HKV, bs, 512]); slot < 0 is skipped."""
    N, hkv, hd = k.shape
    if N:
        _store_kernel[(N, hkv)](c, slot, k, v, k.stride(0), k.stride(1), v.stride(0), v.stride(1),
                                c.stride(0), c.stride(1), c.stride(2), BS=c.shape[2], DH=hd, num_warps=1)


def to_limbs(c):
    """int32 cache [nb, HKV, bs, 2 HD] (K | V) -> the int8 limb-plane layout (reference / tests)."""
    HD = c.shape[-1] // 2
    out = torch.zeros(*c.shape[:3], 8 * HD, dtype=torch.int8, device=c.device)
    for part, o in ((c[..., :HD], 0), (c[..., HD:], 3)):
        r = part.clone()
        for i in range(3):
            lo = ((r + 128) & 255) - 128
            out[..., (o + i) * HD:(o + i + 1) * HD] = lo.to(torch.int8)
            r = (r - lo) >> 8
    return out


def weight_table(lut):
    """The attention weight table rsh(exp_lut, FE - FP) as int32 (values <= 2^20). Built once when the
    model loads (it syncs with the host, so it must not run inside a compiled region or a graph capture)."""
    t = (lut + (1 << (FE - FP - 1))) >> (FE - FP)
    assert int(t.max()) < 1 << 31 and int(t.min()) >= 0
    return t.to(torch.int32).contiguous()


DEC_PROGRAMS = 528     # decode programs to aim for (4 per SM on H100 / H800)
SM_TARGET = 264        # programs to aim for (2 per SM on H100 / H800)


@torch.compiler.disable          # a graph break for torch.compile callers: the launch picks nsplit from shapes
def attend(q, qpos, cache, bt, ptab, nsplit=None, bm=None, bn=None, warps=4):
    """q int64 [R, HKV, M, HD] (|q| within 3 limbs), qpos int64 [R, M] (key s visible to query i iff
    s <= qpos[r, i]), cache int8 limb planes [nb, HKV, bs, 8 HD] (see `store`), bt [R, *] block-table rows, ptab = weight_table(exp table).
    Returns num int64 [R, HKV, M, HD], den int64 [R, HKV, M, 1].

    Split-KV: each (query tile, request, head) is split into nsplit key ranges, computed on the device
    from the tile's own last position, so equal work per split and no host-side lengths (graph safe).
    Kernel 1: per-split row maxima, combined with an int64 atomic max. Kernel 2: weights from the global
    maximum, per-split num / den added with int64 atomic adds. Max and integer sums are exact in any
    order, so the result does not depend on nsplit or on scheduling.
    With nsplit == 1 each output element has exactly one writer: the kernels store instead of using atomics, and
    mx / num / den are not initialised (same values: a row with no visible key gets NEG / 0 / 0 in both paths)."""
    R, nh, M, dh = q.shape
    zmul, zsh = score_rule(dh)
    HKV, HD = nh, dh
    assert cache.shape[1] == nh and cache.shape[3] == 8 * dh and cache.stride(3) == 1 and bt.shape[1] * cache.shape[2] < 43690
    q = q.contiguous()
    qpos = qpos.contiguous()
    if M <= DEC_BQ and bm is None:
        return _attend_dec(q, qpos, cache, bt, ptab, nsplit, bn or 64, warps)  # 64 beats 128 at every R once contexts are long (p3_splitkv sweep)
    BM = bm or (16 if M <= 16 else 64)
    bn = bn or 64
    tiles = triton.cdiv(M, BM) * R * HKV
    if nsplit is None:
        nsplit = max(1, min(64, triton.cdiv(SM_TARGET, tiles)))
    split = nsplit > 1
    if split:           # atomics accumulate into initialised buffers
        mx = torch.full((R, HKV, M), NEG.value, dtype=torch.long, device=q.device)
        num = torch.zeros(R, HKV, M, HD, dtype=torch.long, device=q.device)
        den = torch.zeros(R, HKV, M, 1, dtype=torch.long, device=q.device)
    else:               # every element is written by exactly one program: no initialisation (same values)
        mx = torch.empty((R, HKV, M), dtype=torch.long, device=q.device)
        num = torch.empty(R, HKV, M, HD, dtype=torch.long, device=q.device)
        den = torch.empty(R, HKV, M, 1, dtype=torch.long, device=q.device)
    grid = (triton.cdiv(M, BM), R * HKV, nsplit)
    common = (q.stride(1), q.stride(2), qpos.stride(0), cache.stride(0), cache.stride(1), cache.stride(2),
              bt.stride(0))
    _max_kernel[grid](q, qpos, cache, bt, mx, M, nsplit, *common,
                      BS=cache.shape[2], NHKV=HKV, DH=HD, BM=BM, BN=bn, SPLIT=split, num_warps=warps)
    _acc_kernel[grid](q, qpos, cache, bt, ptab, mx, num, den, M, nsplit, *common, num.stride(1),
                      num.stride(2), BS=cache.shape[2], NHKV=HKV, DH=HD, ZSH=zsh, ZMUL=zmul or 0,
                      ZLO=EXP_MIN << FA, BM=BM, BN=bn, SPLIT=split, num_warps=warps)
    return num, den


DEC_BQ = 4                       # decode: the G = 4 queries of one KV head, limbs stacked to 16 dot rows


def _attend_dec(q, qpos, cache, bt, ptab, nsplit, bn, warps):
    """attend for M <= DEC_BQ: one program per (request, KV head, key split), limb-stacked dots. With one
    split a single kernel takes the maxima and the sums and stores them (no maxima pass, no atomics, no
    zero fills); it is used once there are at least SM_TARGET / 2 programs (32 or more requests). Otherwise
    nsplit = SM_TARGET // programs, so the splits fit in one wave."""
    R, HKV, M, HD = q.shape
    zmul, zsh = score_rule(HD)
    if nsplit is None:
        tiles = R * HKV
        # R <= 8: as before but 4 programs per SM (base 2); R > 8: split only with >= 2048 table keys per split
        # (a split costs ~30 us: 3 launches + zero fills, and the table is as wide as max_model_len, not the real context)
        nsplit = max(1, min(64, DEC_PROGRAMS // tiles, (bt.shape[1] * cache.shape[2] // 2048) if R > 8 else 64))
    fused = nsplit == 1
    new = torch.empty if fused else torch.zeros
    num = new(R, HKV, M, HD, dtype=torch.long, device=q.device)
    den = new(R, HKV, M, 1, dtype=torch.long, device=q.device)
    mx = num if fused else torch.full((R, HKV, M), NEG.value, dtype=torch.long, device=q.device)
    grid = (1, R * HKV, nsplit)
    common = (q.stride(1), q.stride(2), qpos.stride(0), cache.stride(0), cache.stride(1), cache.stride(2),
              bt.stride(0))
    if not fused:
        _dec_max_kernel[grid](q, qpos, cache, bt, mx, M, nsplit, *common,
                              BS=cache.shape[2], NHKV=HKV, DH=HD, BQ=DEC_BQ, BN=bn, num_warps=warps)
    _dec_acc_kernel[grid](q, qpos, cache, bt, ptab, mx, num, den, M, nsplit, *common, num.stride(1),
                          num.stride(2), BS=cache.shape[2], NHKV=HKV, DH=HD, ZSH=zsh, ZMUL=zmul or 0,
                          ZLO=EXP_MIN << FA, BQ=DEC_BQ, BN=bn, FUSED=fused, num_warps=warps)
    return num, den
