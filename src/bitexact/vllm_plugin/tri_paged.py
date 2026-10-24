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
Returns num (int64 [R, HKV, M, HD]) and den (int64 [R, HKV, M, 1]); the caller rounds num / den.
"""

import torch
import triton
import triton.language as tl

from bitexact.int8.consts import EXP_MIN, FA, FE
from bitexact.int8.segs import FP, FQ
from bitexact.llama import HD, HKV

NEG = tl.constexpr(-(1 << 62))


@triton.jit
def _split(r):
    lo = ((r + 128) & 255) - 128
    return lo.to(tl.int8), (r - lo) >> 8


@triton.jit
def _limb_dot(a0, a1, a2, rb):
    """Exact a @ b, a given as 3 int8 limbs, b int32 within 3 limbs -> int64."""
    b0, rb = _split(rb)
    b1, rb = _split(rb)
    b2, rb = _split(rb)
    c0 = tl.dot(a0, b0, out_dtype=tl.int32)
    c1 = tl.dot(a1, b0, out_dtype=tl.int32) + tl.dot(a0, b1, out_dtype=tl.int32)
    c2 = tl.dot(a2, b0, out_dtype=tl.int32) + tl.dot(a1, b1, out_dtype=tl.int32) + \
        tl.dot(a0, b2, out_dtype=tl.int32)
    c3 = tl.dot(a2, b1, out_dtype=tl.int32) + tl.dot(a1, b2, out_dtype=tl.int32)
    c4 = tl.dot(a2, b2, out_dtype=tl.int32)
    return c0.to(tl.int64) + (c1.to(tl.int64) << 8) + (c2.to(tl.int64) << 16) \
        + (c3.to(tl.int64) << 24) + (c4.to(tl.int64) << 32)


@triton.jit
def _limb_parts(a0, a1, a2, rb):
    """The five int32 limb-weight partial sums of a @ b (weights 2^0, 2^8, ..., 2^32)."""
    b0, rb = _split(rb)
    b1, rb = _split(rb)
    b2, rb = _split(rb)
    c0 = tl.dot(a0, b0, out_dtype=tl.int32)
    c1 = tl.dot(a1, b0, out_dtype=tl.int32) + tl.dot(a0, b1, out_dtype=tl.int32)
    c2 = tl.dot(a2, b0, out_dtype=tl.int32) + tl.dot(a1, b1, out_dtype=tl.int32) + \
        tl.dot(a0, b2, out_dtype=tl.int32)
    c3 = tl.dot(a2, b1, out_dtype=tl.int32) + tl.dot(a1, b2, out_dtype=tl.int32)
    c4 = tl.dot(a2, b2, out_dtype=tl.int32)
    return c0, c1, c2, c3, c4


@triton.jit
def _rsh(x, s: tl.constexpr):
    return (x + (1 << (s - 1))) >> s


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
def _scores(a0, a1, a2, cbase, bt_ptr, k0, hi, qpos, rd, sc_nb, sc_bs, BS: tl.constexpr, BN: tl.constexpr):
    ks = k0 + tl.arange(0, BN)
    kok = ks < hi
    blk = tl.load(bt_ptr + ks // BS, mask=kok, other=0).to(tl.int64)
    off = blk * sc_nb + (ks % BS) * sc_bs
    kt = tl.load(cbase + off[None, :] + rd[:, None], mask=kok[None, :], other=0)           # [DH, BN]
    raw = _limb_dot(a0, a1, a2, kt)
    valid = (ks[None, :] <= qpos[:, None]) & kok[None, :]
    return raw, valid, off, kok


@triton.jit
def _max_kernel(q_ptr, qpos_ptr, c_ptr, bt_ptr, mx_ptr, M, NSPLIT,
                sqz, sqm, spz, sc_nb, sc_h, sc_bs, sbt,
                BS: tl.constexpr, NHKV: tl.constexpr, DH: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pz, h, rm, rd, mok, qpos, a0, a1, a2, lo, hi, bt_ptr = _tile(
        q_ptr, qpos_ptr, bt_ptr, M, sqz, sqm, spz, sbt, NSPLIT, NHKV, DH, BM, BN)
    cbase = c_ptr + h * sc_h
    mx = tl.full((BM,), NEG, dtype=tl.int64)
    for k0 in tl.range(lo, hi, BN, num_stages=1):
        raw, valid, off, kok = _scores(a0, a1, a2, cbase, bt_ptr, k0, hi, qpos, rd, sc_nb, sc_bs, BS, BN)
        mx = tl.maximum(mx, tl.max(tl.where(valid, raw, NEG), axis=1))
    tl.atomic_max(mx_ptr + pz * M + rm, mx, mask=mok & (hi > lo))


@triton.jit
def _acc_kernel(q_ptr, qpos_ptr, c_ptr, bt_ptr, ptab_ptr, mx_ptr, num_ptr, den_ptr, M, NSPLIT,
                sqz, sqm, spz, sc_nb, sc_h, sc_bs, sbt, snz, snm,
                BS: tl.constexpr, NHKV: tl.constexpr, DH: tl.constexpr, ZSH: tl.constexpr,
                ZLO: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
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
        raw, valid, off, kok = _scores(a0, a1, a2, cbase, bt_ptr, k0, hi, qpos, rd, sc_nb, sc_bs, BS, BN)
        z = _rsh(tl.where(valid, raw, NEG) - mx[:, None], ZSH)          # <= 0
        idx = (tl.maximum(z, ZLO - 1) - ZLO).to(tl.int32)               # table index, -1 = below table
        ok = valid & (idx >= 0)
        p = tl.load(ptab_ptr + tl.maximum(idx, 0), mask=ok, other=0)    # int32, = rsh(exp, FE - FP)
        den += tl.sum(p, axis=1).to(tl.int64)
        p0, p = _split(p)
        p1, p = _split(p)
        p2, p = _split(p)
        vt = tl.load(cbase + off[:, None] + DH + rd[None, :], mask=kok[:, None], other=0)  # [BN, DH]
        c0, c1, c2, c3, c4 = _limb_parts(p0, p1, p2, vt)
        n0 += c0
        n1 += c1
        n2 += c2
        n3 += c3
        n4 += c4
    num = n0.to(tl.int64) + (n1.to(tl.int64) << 8) + (n2.to(tl.int64) << 16) \
        + (n3.to(tl.int64) << 24) + (n4.to(tl.int64) << 32)
    any_k = hi > lo
    tl.atomic_add(num_ptr + pz * snz + rm[:, None] * snm + rd[None, :], num, mask=mok[:, None] & any_k)
    tl.atomic_add(den_ptr + pz * M + rm, den, mask=mok & any_k)


def weight_table(lut):
    """The attention weight table rsh(exp_lut, FE - FP) as int32 (values <= 2^20). Built once when the
    model loads (it syncs with the host, so it must not run inside a compiled region or a graph capture)."""
    t = (lut + (1 << (FE - FP - 1))) >> (FE - FP)
    assert int(t.max()) < 1 << 31 and int(t.min()) >= 0
    return t.to(torch.int32).contiguous()


SM_TARGET = 264        # programs to aim for (2 per SM on H100 / H800)


@torch.compiler.disable          # a graph break for torch.compile callers: the launch picks nsplit from shapes
def attend(q, qpos, cache, bt, ptab, nsplit=None, bm=None, bn=64, warps=4):
    """q int64 [R, HKV, M, HD] (|q| within 3 limbs), qpos int64 [R, M] (key s visible to query i iff
    s <= qpos[r, i]), cache int32 [nb, HKV, bs, 2 HD], bt [R, *] block-table rows, ptab = weight_table(exp table).
    Returns num int64 [R, HKV, M, HD], den int64 [R, HKV, M, 1].

    Split-KV: each (query tile, request, head) is split into nsplit key ranges, computed on the device
    from the tile's own last position, so equal work per split and no host-side lengths (graph safe).
    Kernel 1: per-split row maxima, combined with an int64 atomic max. Kernel 2: weights from the global
    maximum, per-split num / den added with int64 atomic adds. Max and integer sums are exact in any
    order, so the result does not depend on nsplit or on scheduling."""
    R, nh, M, dh = q.shape
    assert nh == HKV and dh == HD and cache.stride(3) == 1 and bt.shape[1] * cache.shape[2] < 43690
    q = q.contiguous()
    qpos = qpos.contiguous()
    BM = bm or (16 if M <= 16 else 64)
    tiles = triton.cdiv(M, BM) * R * HKV
    if nsplit is None:
        nsplit = max(1, min(64, triton.cdiv(SM_TARGET, tiles)))
    mx = torch.full((R, HKV, M), NEG.value, dtype=torch.long, device=q.device)
    num = torch.zeros(R, HKV, M, HD, dtype=torch.long, device=q.device)
    den = torch.zeros(R, HKV, M, 1, dtype=torch.long, device=q.device)
    grid = (triton.cdiv(M, BM), R * HKV, nsplit)
    common = (q.stride(1), q.stride(2), qpos.stride(0), cache.stride(0), cache.stride(1), cache.stride(2),
              bt.stride(0))
    _max_kernel[grid](q, qpos, cache, bt, mx, M, nsplit, *common,
                      BS=cache.shape[2], NHKV=HKV, DH=HD, BM=BM, BN=bn, num_warps=warps)
    _acc_kernel[grid](q, qpos, cache, bt, ptab, mx, num, den, M, nsplit, *common, num.stride(1),
                      num.stride(2), BS=cache.shape[2], NHKV=HKV, DH=HD, ZSH=2 * FQ + 3 - FA,
                      ZLO=EXP_MIN << FA, BM=BM, BN=bn, num_warps=warps)
    return num, den
