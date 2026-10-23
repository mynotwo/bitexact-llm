"""Int8-engine primitives against Python integer / float64 references (CPU)."""

import torch

from bitexact.int8.consts import FA, FR, SILU_LIM
from bitexact.int8.ops import imm, qmax, quant_rows, rms, silu


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
