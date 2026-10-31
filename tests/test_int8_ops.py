"""Int8-engine primitives against Python integer / float64 references (CPU)."""

import torch

from bitexact.int8.consts import FA, FR, SILU_LIM
from bitexact.int8.ops import bitlen32, imm, post_bound, qmax, quant_rows, rdiv32, rms, silu
from bitexact.intops import bitlen, rdiv, rsh


def test_quant_rows_and_imm():
    g = torch.Generator().manual_seed(0)
    for nl in (1, 2, 3):
        z = torch.randint(-(1 << 44), 1 << 44, (5, 64), generator=g)
        z[1] //= 1 << 40                                   # small row: left shift path
        q, as_, _ = quant_rows(z, nl)
        assert int(q.abs().max()) <= qmax(nl)
        assert bool(((as_ >= 1 << 29) & (as_ <= 1 << 30)).all())
        w = torch.randint(-127, 128, (7, 64), generator=g).to(torch.int8)
        ref = [[sum(a * b for a, b in zip(r, c)) for c in w.long().tolist()] for r in q.tolist()]
        assert imm(q, w.double().t()).tolist() == ref


def test_rms_close():
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(4, 2048, generator=g, dtype=torch.float64) * torch.tensor([[1e-3], [0.1], [3.0], [200.0]]))
    xi = torch.round(x * (1 << FR)).long()
    r, shx = rms(xi)
    val = r.double() * 2.0 ** (shx - FR).double()
    ref = torch.sqrt((x * x).mean(1) + 1e-5)
    assert float(((val - ref) / ref).abs().max()) < 1e-4


def test_silu():
    xs = torch.arange(-SILU_LIM << FA, (SILU_LIM << FA) + 1, dtype=torch.float64) / (1 << FA)
    lut = torch.round(xs / (1 + torch.exp(-xs)) * (1 << FR)).long()
    x = torch.linspace(-40, 40, 10001, dtype=torch.float64)
    got = silu(torch.round(x * (1 << FR)).long(), lut).double() / (1 << FR)
    assert float((got - x / (1 + torch.exp(-x))).abs().max()) < 3e-5


def rdiv_cases(g):
    """(a, b) with b in [2^29, 2^30] (quant_rows' as_) and a near every rounding boundary of a / b, |a / b| <= qmax(2)."""
    b = torch.randint(1 << 29, (1 << 30) + 1, (4000,), generator=g)
    b[:4] = torch.tensor([1 << 29, 1 << 30, (1 << 29) + 1, (1 << 30) - 1])
    j = torch.randint(-qmax(2), qmax(2) + 1, b.shape, generator=g)
    j[4:8] = torch.tensor([qmax(2), -qmax(2), 0, 1])
    d = torch.randint(-3, 4, b.shape, generator=g)
    a = j * b - b // 2 + d                                     # around (j - 1/2) b: the ties of round(a / b)
    a = torch.cat([a, j * b + d, torch.randint(-qmax(2), qmax(2) + 1, b.shape, generator=g) * b // 7])
    b = b.repeat(3)
    keep = (2 * a + b).abs() < qmax(2) * 2 * b                 # rdiv32's range: |quotient| <= qmax + 1/2
    return a[keep], b[keep]


def test_rdiv32_bitlen32_equal_reference():
    g = torch.Generator().manual_seed(4)
    a, b = rdiv_cases(g)
    assert torch.equal(rdiv32(a, b), rdiv(a, b))
    x = torch.cat([torch.tensor([0, 1, 2, 3, (1 << 24) - 1, 1 << 24, (1 << 24) + 1, (1 << 62) - 1, 1 << 62]),
                   torch.randint(0, 1 << 62, (5000,), generator=g) >> torch.randint(0, 62, (5000,), generator=g)])
    x = torch.cat([x, (1 << torch.arange(1, 62)) - 1, 1 << torch.arange(0, 62)])
    assert torch.equal(bitlen32(x), bitlen(x))


def test_post_bound_holds():
    """post_bound against the largest |rsh(q @ W^T * m, fmsh)| over sign-pattern inputs at |q| = qmax (the extremes)."""
    g = torch.Generator().manual_seed(5)
    w = torch.randint(-127, 128, (6, 50), generator=g).to(torch.int8)
    m = torch.randint(1, 1 << 17, (6,), generator=g)
    for nl, fmsh in [(1, 9), (2, 13), (2, 0), (3, -2)]:
        q = torch.sign(w.long()) * qmax(nl)                    # row c of q maximizes acc_c (> 0)
        t = rsh(q @ w.long().t() * m, fmsh)
        assert int(t.abs().amax()) == post_bound(w, m, fmsh, nl)
