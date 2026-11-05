"""One Triton kernel per exact-mode linear: all 9 limb products, the sparse residual and the rounding epilogue.

Replaces wmm_triton (split-K int64 partials [split, M, N] in memory) + segs.lin_y (sum, ELL residual gather,
combine, shift_round_hl) for decode-size M. A program owns a [BM, BN] output tile and walks all of K:
  T  = sum_k q[m, k] W[n, k]        q split into 3 int8 limbs in registers, W = 3 stored limbs, 9 int8 dots
                                    into 5 int32 accumulators (|acc| <= 3 K 2^14 < 2^31), int64 at the end
  B  = sum_j q[m, ec[n, j]] ev[n, j]   sparse residual (weight bits below the row grid), exact int64
  y  = shift_round_hl(combine(T, d, B), sh[m] + g[n] - d[n] + FR)   (exact/ops.py, same int64 operations)
and the lost-bit / range flag of wmm.epilogue(checked=True) is set from the kernel. The result is the same
integers as lin_y(*Lin(q), sh) because integer sums are exact in any order.
"""

import torch
import triton
import triton.language as tl

from bitexact.exact.consts import FR

FRC = tl.constexpr(FR)


@triton.jit
def _split(r):
    lo = ((r + 128) & 255) - 128
    return lo.to(tl.int8), (r - lo) >> 8


@triton.jit
def _one(x):
    return x * 0 + 1


@triton.jit
def _round_shift(T, B, d, p):
    """wmm.epilogue's (y, H, L): round((T 2^d + B) 2^p) from int64 T, B and per-column d in [0, 30], p int64."""
    s = 32 - d
    H = T >> s
    L = ((T & ((_one(s).to(tl.int64) << s) - 1)) << d) + B
    H = H + (L >> 32)
    L = L & 0xFFFFFFFF
    k = tl.maximum(-p, 0)
    pp = tl.minimum(tl.maximum(p, 0), 30)
    up = (H << (32 + pp)) + (L << pp)
    k1 = tl.minimum(tl.maximum(k, 1), 31)
    small = (H << (32 - k1)) + ((L + (_one(k1) << (k1 - 1))) >> k1)
    k2 = tl.minimum(tl.maximum(k, 32), 94)
    carry = tl.where(k2 == 32, (L + (1 << 31)) >> 32, _one(k2) << tl.maximum(k2 - 33, 0))
    big = (H + carry) >> tl.minimum(k2 - 32, 62)
    y = tl.where(p >= 0, up, tl.where(k < 32, small, big))
    return y, H, L


@triton.jit
def _lin_kernel(q_ptr, w_ptr, d_ptr, g_ptr, sh_ptr, cnt_ptr, ec_ptr, ev_ptr, y_ptr, flag_ptr,
                M, N, K, R, sqm, swl, swn,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, HASRES: tl.constexpr, CHECK: tl.constexpr):
    pn = tl.program_id(0)
    pm = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mm = rm < M
    nm = rn < N
    c0 = tl.zeros((BM, BN), dtype=tl.int32)
    c1 = tl.zeros((BM, BN), dtype=tl.int32)
    c2 = tl.zeros((BM, BN), dtype=tl.int32)
    c3 = tl.zeros((BM, BN), dtype=tl.int32)
    c4 = tl.zeros((BM, BN), dtype=tl.int32)
    for k0 in range(0, K, BK):
        ka = k0 + rk
        ra = tl.load(q_ptr + rm[:, None] * sqm + ka[None, :], mask=mm[:, None] & (ka[None, :] < K), other=0)
        bm = (ka[:, None] < K) & nm[None, :]
        bp = w_ptr + rn[None, :] * swn + ka[:, None]
        b0 = tl.load(bp, mask=bm, other=0)
        b1 = tl.load(bp + swl, mask=bm, other=0)
        b2 = tl.load(bp + 2 * swl, mask=bm, other=0)
        a0, ra = _split(ra)
        a1, ra = _split(ra)
        a2, ra = _split(ra)
        c0 = tl.dot(a0, b0, c0, out_dtype=tl.int32)
        c1 = tl.dot(a0, b1, c1, out_dtype=tl.int32)
        c1 = tl.dot(a1, b0, c1, out_dtype=tl.int32)
        c2 = tl.dot(a0, b2, c2, out_dtype=tl.int32)
        c2 = tl.dot(a1, b1, c2, out_dtype=tl.int32)
        c2 = tl.dot(a2, b0, c2, out_dtype=tl.int32)
        c3 = tl.dot(a1, b2, c3, out_dtype=tl.int32)
        c3 = tl.dot(a2, b1, c3, out_dtype=tl.int32)
        c4 = tl.dot(a2, b2, c4, out_dtype=tl.int32)
    T = c0.to(tl.int64) + (c1.to(tl.int64) << 8) + (c2.to(tl.int64) << 16) \
        + (c3.to(tl.int64) << 24) + (c4.to(tl.int64) << 32)
    B = tl.zeros((BM, BN), dtype=tl.int64)
    if HASRES:
        cnt = tl.load(cnt_ptr + rn, mask=nm, other=0)
        for j in range(0, tl.max(cnt, axis=0)):
            ec = tl.load(ec_ptr + rn * R + j, mask=nm & (j < cnt), other=0)
            ev = tl.load(ev_ptr + rn * R + j, mask=nm & (j < cnt), other=0)
            qv = tl.load(q_ptr + rm[:, None] * sqm + ec[None, :], mask=mm[:, None] & nm[None, :], other=0)
            B += qv.to(tl.int64) * ev[None, :]
    d = tl.load(d_ptr + rn, mask=nm, other=0).to(tl.int64)[None, :]
    g = tl.load(g_ptr + rn, mask=nm, other=0).to(tl.int64)[None, :]
    sh = tl.load(sh_ptr + rm, mask=mm, other=0)[:, None]
    p = sh + (g - d) + FRC
    y, H, L = _round_shift(T, B, d, p)
    ok = mm[:, None] & nm[None, :]
    if CHECK:
        one = _one(p)
        up_limit = one << (31 - tl.minimum(tl.maximum(p, 0), 30))
        small_limit = one << (31 + tl.minimum(tl.maximum(-p, 1), 31))
        up_lost = (H >= up_limit) | (H < -up_limit) | ((p > 30) & ((H != 0) | (L != 0)))
        small_lost = (H >= small_limit) | (H < -small_limit)
        lost = tl.where(p >= 0, up_lost, (p > -32) & small_lost)
        bad = (lost | (y <= -(1 << 62)) | (y >= (1 << 62))) & ok
        tl.store(flag_ptr, 1, mask=tl.max(tl.max(bad.to(tl.int32), axis=1), axis=0) > 0)
    tl.store(y_ptr + rm[:, None].to(tl.int64) * N + rn[None, :], y, mask=ok)


class FusedLin:
    """The linear layer `lin` (exact.model.Lin on CUDA, gemm = triton) as one kernel per call: q int32
    [M, K], sh int64 [M] -> y int64 [M, N] at FR; sets flag (int32 [1]) to 1 when wmm.epilogue's checked
    condition (lost bits, |y| >= 2^62) holds anywhere."""

    # (N, K) -> [(max M, (BM, BN, BK, warps, stages))], swept per shape on an H800 (experiments/perf/p7_exactplugin/
    # sweep_gemm.py and p10_exact_gemm, 1B/8B shapes); the integers do not depend on the config.
    # Unmeasured larger batches keep the previous configuration.
    TABLE = {
        (3072, 2048): [(8, (16, 32, 128, 4, 3)), (32, (16, 64, 128, 4, 3)),
                       (128, (64, 32, 128, 4, 3)), (1 << 30, (64, 32, 128, 4, 4))],
        (2048, 2048): [(8, (16, 32, 128, 4, 3)), (32, (16, 32, 128, 4, 3)), (64, (16, 64, 128, 4, 3)),
                       (1 << 30, (64, 32, 128, 4, 4))],
        (16384, 2048): [(8, (16, 64, 128, 4, 3)), (32, (16, 128, 128, 4, 3)), (64, (64, 64, 128, 4, 2)),
                        (1 << 30, (64, 64, 64, 4, 3))],
        (2048, 8192): [(8, (16, 32, 128, 4, 3)), (32, (16, 32, 256, 4, 3)), (64, (16, 64, 256, 4, 3)),
                       (1 << 30, (64, 32, 128, 4, 4))],
        (128256, 2048): [(8, (16, 64, 128, 4, 4)), (32, (32, 64, 128, 4, 3)), (1 << 30, (64, 64, 64, 4, 4))],
        (6144, 4096): [(8, (16, 64, 128, 4, 4)), (16, (16, 32, 128, 4, 3)),
                       (32, (16, 64, 128, 4, 3)), (64, (64, 32, 128, 4, 3)),
                       (1 << 30, (64, 64, 64, 4, 3))],
        (4096, 4096): [(1, (16, 32, 128, 4, 4)), (16, (16, 32, 128, 4, 3)),
                       (32, (16, 64, 128, 4, 3)),
                       (128, (64, 32, 128, 4, 3)), (1 << 30, (64, 64, 64, 4, 3))],
        (28672, 4096): [(8, (16, 64, 128, 4, 3)), (16, (16, 32, 128, 4, 3)),
                        (32, (32, 64, 128, 4, 3)), (64, (64, 32, 128, 4, 3)),
                        (1 << 30, (64, 64, 64, 4, 3))],
        (4096, 14336): [(1, (16, 32, 256, 4, 3)), (16, (16, 32, 128, 4, 3)),
                        (32, (16, 64, 256, 4, 3)), (64, (64, 32, 128, 4, 3)),
                        (128, (64, 64, 128, 4, 3)), (1 << 30, (64, 64, 64, 4, 3))],
        (128256, 4096): [(8, (16, 64, 128, 4, 4)), (16, (16, 32, 128, 4, 3)),
                         (32, (32, 64, 128, 4, 3)), (64, (64, 32, 128, 4, 3)),
                         (128, (64, 64, 64, 4, 4)), (1 << 30, (64, 64, 64, 4, 3))],
    }
    DEFAULT = [(16, (16, 32, 128, 4, 3)), (64, (64, 32, 128, 4, 3)), (1 << 30, (64, 64, 64, 4, 3))]

    def __init__(self, lin, cfg=None):
        self.limbs, self.d, self.g, self.N = lin.limbs, lin.d, lin.g, lin.N
        rr = lin.rr
        cnt = torch.bincount(rr.cpu(), minlength=self.N) if rr.numel() else torch.zeros(self.N, dtype=torch.long)
        self.has_res = rr.numel() > 0
        self.R = int(cnt.max()) if self.has_res else 1
        dev = self.limbs.device
        self.cnt = cnt.to(torch.int32).to(dev)
        if self.has_res:
            start = torch.cumsum(cnt, 0) - cnt
            rrc = rr.cpu()
            j = torch.arange(rrc.numel()) - start[rrc]
            ec = torch.zeros(self.N, self.R, dtype=torch.int32)
            ev = torch.zeros(self.N, self.R, dtype=torch.long)
            ec[rrc, j], ev[rrc, j] = lin.rc.cpu().to(torch.int32), lin.rv.cpu()
            self.ec, self.ev = ec.to(dev), ev.to(dev)
        else:
            self.ec = self.ev = torch.zeros(1, dtype=torch.int32, device=dev)
        self.cfg = cfg

    def __call__(self, q, sh, flag, checked=True):
        M, K = q.shape
        cfg = self.cfg or next(c for mx, c in self.TABLE.get((self.N, K), self.DEFAULT) if M <= mx)
        BM, BN, BK, nw, ns = cfg
        y = torch.empty((M, self.N), dtype=torch.long, device=q.device)
        grid = (triton.cdiv(self.N, BN), triton.cdiv(M, BM))
        _lin_kernel[grid](q, self.limbs, self.d, self.g, sh, self.cnt, self.ec, self.ev, y, flag,
                          M, self.N, K, self.R, q.stride(0), self.limbs.stride(0), self.limbs.stride(1),
                          BM=BM, BN=BN, BK=BK, HASRES=self.has_res, CHECK=checked, num_warps=nw, num_stages=ns)
        return y
