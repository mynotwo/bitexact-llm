"""Exact-bf16 primitives against Python big-integer / Fraction references (CPU; GEMM paths on CUDA)."""

import random
from fractions import Fraction

import pytest
import torch

from bitexact import intops
from bitexact.exact import ops
from bitexact.exact.consts import EXP2_BITS, EXP2_ONE, FR, decompose
from bitexact.exact.wmm import HAVE_TRITON, epilogue, residual, wmm_cpu

CUDA = torch.cuda.is_available()


def rnd(v):
    """round half up of a Fraction / int."""
    v = Fraction(v)
    return (v + Fraction(1, 2)).__floor__()


def test_rshc():
    xs = [0, 1, -1, 5, -5, 6, -6, 7, -7, (1 << 50) + 3, -(1 << 50) - 3]
    for s in (-3, 0, 1, 2, 7, 40, 62):
        got = ops.rshc(torch.tensor(xs), torch.full((len(xs),), s)).tolist()
        assert got == [rnd(Fraction(x, 1) / Fraction(2) ** s) for x in xs]


def test_div_round():
    R = random.Random(0)
    a, p, r = [], [], []
    for _ in range(4000):
        rr = R.randrange(1 << 11, 1 << R.choice([12, 20, 31, 33]))
        pp = R.randrange(-70, 60)
        lim = (1 << 61) if pp <= 0 else max(1, (rr << 61) >> (pp + 1)) // max(1, rr) * rr
        a.append(R.randrange(-min(lim, 1 << 61), min(lim, 1 << 61) + 1))
        p.append(pp)
        r.append(rr)
    got = ops.div_round(torch.tensor(a), torch.tensor(p), torch.tensor(r)).tolist()
    ref = [rnd(Fraction(x) * Fraction(2) ** pp / rr) for x, pp, rr in zip(a, p, r)]
    assert got == ref


def test_fdiv():
    R = random.Random(5)
    a = [R.randrange(-(1 << 62), 1 << 62) for _ in range(20000)] + [(1 << 62) - 1, -(1 << 62), 0, 12345 * 99991]
    b = [R.randrange(1 << 11, 1 << R.choice([12, 31, 34])) for _ in range(20000)] + [1 << 11, 1 << 11, 7777, 99991]
    a = [x if abs(x) // y < 1 << 51 else x % (y << 50) for x, y in zip(a, b)]
    assert intops.fdiv(torch.tensor(a), torch.tensor(b)).tolist() == [x // y for x, y in zip(a, b)]
    assert intops.rdiv(torch.tensor(a[:5000]) >> 2, torch.tensor(b[:5000])).tolist() == \
        [rnd(Fraction(x >> 2, y)) for x, y in zip(a[:5000], b[:5000])]


def test_combine_shift():
    R = random.Random(1)
    n = 4000
    T = [R.randrange(-(1 << 58), 1 << 58) for _ in range(n)]
    d = [R.choice([0, 0, 1, 7, 16, 30]) for _ in range(n)]
    B = [R.randrange(-(1 << 50), 1 << 50) if dd else 0 for dd in d]
    p = [R.randrange(-100, 3) for _ in range(n)]
    H, L = ops.combine(torch.tensor(T), torch.tensor(d), torch.tensor(B))
    assert all(h * (1 << 32) + lo == t * (1 << dd) + b for h, lo, t, dd, b in zip(H.tolist(), L.tolist(), T, d, B))
    got = ops.shift_round_hl(H, L, torch.tensor(p)).tolist()
    ref = [rnd(Fraction(t * (1 << dd) + b) * Fraction(2) ** pp) for t, dd, b, pp in zip(T, d, B, p)]
    inside = [abs(r) < 1 << 61 for r in ref]                    # the contract: result below 2^62
    assert sum(inside) > n // 2
    assert [x for x, k in zip(got, inside) if k] == [x for x, k in zip(ref, inside) if k]


def test_quant():
    g = torch.Generator().manual_seed(0)
    P = torch.randint(-(1 << 44), 1 << 44, (6, 64), generator=g)
    P[1] //= 1 << 40
    P[2] = 0
    q, sh = ops.quant(P, -40)
    assert int(q.abs().max()) <= 1 << 22
    for i in range(6):
        for j in range(64):
            assert int(q[i, j]) == rnd(Fraction(int(P[i, j])) * Fraction(2) ** (-40 - int(sh[i])))


def test_rms():
    g = torch.Generator().manual_seed(1)
    x = torch.randint(-(200 << FR), 200 << FR, (4, 2048), generator=g)
    x[1] //= 1 << 12
    r, er = ops.rms(x)
    ref = (x.double() / 2 ** FR).pow(2).mean(-1).add(1e-5).sqrt()
    assert torch.allclose(r.double() * torch.pow(2.0, er.double()), ref, rtol=1e-7)


def _exp_lut():
    f = torch.arange((1 << EXP2_BITS) + 1, dtype=torch.float64) / (1 << EXP2_BITS)
    return torch.round(torch.pow(2.0, -f) * 2.0 ** EXP2_ONE).long()


def test_exp_silu_precision():
    lut = _exp_lut()
    t = torch.randint(0, 30 << ops.FT2, (10000,))
    v = ops.exp2neg_lut(t, lut, 40).double() / 2 ** 40
    ref = torch.pow(2.0, -t.double() / 2 ** ops.FT2)
    assert float((v - ref).abs().max()) < 3e-10           # interpolation ~2^-32
    gg = torch.randint(-(40 << FR), 40 << FR, (10000,))
    u = torch.full_like(gg, 1 << FR)
    P, eps, ok = ops.silu_mul(gg, u, lut)
    assert bool(ok)
    gd = gg.double() / 2 ** FR
    assert float((P.double() * 2.0 ** eps - gd * torch.sigmoid(gd)).abs().max()) < 1e-7


def _rand_bf16(N, K, seed):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(N, K, generator=g) * 0.02).to(torch.bfloat16)
    w[0, :5] = torch.tensor([1e-9, -3e-8, 0.0, 2.0, -0.5]).to(torch.bfloat16)   # forces a residual
    return w


def test_decompose_and_linear_exact():
    w = _rand_bf16(48, 256, 2)
    dec = decompose(w)
    assert dec["stats"]["same_val"] and dec["stats"]["nnz_res"] > 0
    q = torch.randint(-(1 << 22), (1 << 22) + 1, (5, 256))
    L = dec["limbs"].long()
    W = L[0] + (L[1] << 8) + (L[2] << 16)
    T = wmm_cpu(q, W.double().t().contiguous())
    assert T.tolist() == [[sum(int(a) * int(b) for a, b in zip(qr, wr)) for wr in W] for qr in q]
    B = residual(q, dec["rr"], dec["rc"], dec["rv"], 48)
    sh = torch.tensor([-30, -10, 0, -25, -18])
    y = epilogue(T, B, dec["d"], dec["g"], sh, FR)
    wf = [[Fraction(float(v)) for v in row] for row in w.double()]
    for m in range(5):
        for n in range(48):
            exact = sum(int(q[m, k]) * wf[n][k] for k in range(256)) * Fraction(2) ** (int(sh[m]) + FR)
            assert int(y[m, n]) == rnd(exact)


@pytest.mark.gpu
@pytest.mark.skipif(not (CUDA and HAVE_TRITON), reason="needs CUDA and Triton")
def test_wmm_gpu_paths():
    from bitexact.exact.wmm import CFG_LARGE, CFG_MID, CFG_SMALL, wmm_cublas, wmm_triton
    g = torch.Generator().manual_seed(3)
    for M, N, K in [(1, 256, 2048), (7, 3072, 2048), (37, 512, 8192), (300, 1024, 2048)]:
        q = torch.randint(-(1 << 22), (1 << 22) + 1, (M, K), generator=g)
        q[0, :] = 1 << 22                                       # adversarial: all extremes
        W = torch.randint(-8355711, 8355712, (N, K), generator=g)
        W[0, :] = 8355711
        L = torch.stack([lo.to(torch.int8) for lo in intops.split_limbs(W, 3)[0]])
        ref = wmm_cpu(q, W.double().t().contiguous())
        qc, Lc = q.cuda(), L.cuda()
        for cfg in (None, CFG_SMALL, CFG_MID, CFG_LARGE):
            assert torch.equal(wmm_triton(qc.to(torch.int32), Lc, cfg).sum(0).cpu(), ref)
        assert torch.equal(wmm_cublas(qc, Lc.reshape(3 * N, K)).cpu(), ref)
