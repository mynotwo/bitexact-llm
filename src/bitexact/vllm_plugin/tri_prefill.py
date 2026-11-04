"""Prefill attention straight on the token-major layout of the fused forward (vLLM plugin, CUDA).

tri_paged.attend takes queries as [request, KV head, G * Tm, HD] (g-major, padded to the longest request of a
group), so the prefill path gathered q into that layout, scattered num / den back, and ran the max sweep and
the sum sweep as two launches with atomics. Here one program handles a tile of BT = BM // G consecutive query
tokens of one request and one KV head (rows t * G + g: BT tokens x the G query heads of the KV head, so a K / V
row loaded for the tile serves BM queries). It reads q from, and writes num / den to, the [N, HKV, G, HD] /
[N, HKV, G, 1] arrays tri_fuse works on (no gather, no scatter, no padding queries), and runs both sweeps over
the keys [0, last query position of the tile] itself (the max, then p.v with that max; the second K read is
mostly L2), so there are no atomics, zero fills or max array.

Same integers as tri_paged._max_kernel / _acc_kernel: the products are the same exact limb dots (_scores,
_limb_parts), the maximum and the sums are exact integer reductions, and key s is visible to a query iff
s <= its position, so the grouping of queries into tiles cannot change a bit. tests/test_tri_prefill.py
compares with tri_paged.attend.
"""

import torch
import triton
import triton.language as tl

from bitexact.int8.consts import EXP_MIN, FA
from bitexact.int8.segs import score_rule
from bitexact.vllm_plugin.tri_paged import (NEG, _limb_parts, _load_v, _scores, _split, _zscale)

BM = 64                  # query rows per tile (BM // G tokens)
BN = 64                  # keys per step


@triton.jit
def _pre_kernel(q_ptr, pos_ptr, tile_ptr, c_ptr, bt_ptr, ptab_ptr, num_ptr, den_ptr, sq_tok, sc_nb, sc_h, sc_bs, sbt,
                BS: tl.constexpr, NHKV: tl.constexpr, G: tl.constexpr, DH: tl.constexpr, ZSH: tl.constexpr,
                ZMUL: tl.constexpr, ZLO: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pt = tl.program_id(0)
    h = tl.program_id(1)
    req = tl.load(tile_ptr + 3 * pt).to(tl.int64)
    tok0 = tl.load(tile_ptr + 3 * pt + 1).to(tl.int64)
    ntok = tl.load(tile_ptr + 3 * pt + 2)
    r = tl.arange(0, BM)
    t = r // G
    g = r % G
    rd = tl.arange(0, DH)
    ok = t < ntok
    tok = tok0 + t
    qpos = tl.load(pos_ptr + tok, mask=ok, other=-1).to(tl.int64)
    qoff = tok[:, None] * sq_tok + (h * G + g)[:, None] * DH + rd[None, :]
    rq = tl.load(q_ptr + qoff, mask=ok[:, None], other=0).to(tl.int32)
    a0, rq = _split(rq)
    a1, rq = _split(rq)
    a2, rq = _split(rq)
    kend = tl.max(qpos, axis=0) + 1
    btr = bt_ptr + req * sbt
    cbase = c_ptr + h * sc_h
    mx = tl.full((BM,), NEG, dtype=tl.int64)
    for k0 in tl.range(0, kend, BN, num_stages=1):
        raw, valid, off, kok = _scores(a0, a1, a2, cbase, btr, k0, kend, qpos, rd, sc_nb, sc_bs, BS, DH, BN)
        mx = tl.maximum(mx, tl.max(tl.where(valid, raw, NEG), axis=1))
    n0 = tl.zeros((BM, DH), dtype=tl.int32)
    n1 = tl.zeros((BM, DH), dtype=tl.int32)
    n2 = tl.zeros((BM, DH), dtype=tl.int32)
    n3 = tl.zeros((BM, DH), dtype=tl.int32)
    n4 = tl.zeros((BM, DH), dtype=tl.int32)
    den = tl.zeros((BM,), dtype=tl.int64)
    for k0 in tl.range(0, kend, BN, num_stages=1):
        raw, valid, off, kok = _scores(a0, a1, a2, cbase, btr, k0, kend, qpos, rd, sc_nb, sc_bs, BS, DH, BN)
        z = _zscale(tl.where(valid, raw, NEG) - mx[:, None], ZSH, ZMUL)
        idx = (tl.maximum(z, ZLO - 1) - ZLO).to(tl.int32)
        okp = valid & (idx >= 0)
        p = tl.load(ptab_ptr + tl.maximum(idx, 0), mask=okp, other=0)
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
    tl.store(num_ptr + qoff, num, mask=ok[:, None])
    tl.store(den_ptr + tok * (NHKV * G) + h * G + g, den, mask=ok)


def tile_table(pre, G, bm=BM):
    """[T, 3] int32 (request, first token, tokens) of the BM // G-token tiles of the prefill requests
    pre = [(req, tok0, qlen, seq_len)], on the host."""
    bt_ = bm // G
    return torch.tensor([[r, t0 + s, min(bt_, ql - s)] for r, t0, ql, _ in pre for s in range(0, ql, bt_)],
                        dtype=torch.int32)


def attend(q, pos, cache, bt, ptab, tiles, num, den, bm=BM, bn=BN, warps=4):
    """q int64 [N, HKV, G, HD], pos int64 [N], cache int8 limb planes, bt [R, *] block table, tiles (tile_table,
    on the device) -> num int64 [N, HKV, G, HD] and den [N, HKV, G, 1] rows of the tiles' tokens written
    in place (num / den contiguous)."""
    N, HKV, G, HD = q.shape
    zmul, zsh = score_rule(HD)
    assert cache.shape[1] == HKV and cache.shape[3] == 8 * HD and cache.stride(3) == 1 and q.is_contiguous() \
        and num.is_contiguous() and den.is_contiguous() and bm % G == 0 and bt.shape[1] * cache.shape[2] < 43690
    _pre_kernel[(tiles.shape[0], HKV)](q, pos, tiles, cache, bt, ptab, num, den, q.stride(0), cache.stride(0),
                                       cache.stride(1), cache.stride(2), bt.stride(0), BS=cache.shape[2], NHKV=HKV,
                                       G=G, DH=HD, ZSH=zsh, ZMUL=zmul or 0, ZLO=EXP_MIN << FA, BM=bm, BN=bn,
                                       num_warps=warps)
