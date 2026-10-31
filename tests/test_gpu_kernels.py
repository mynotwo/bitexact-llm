"""CUDA kernels against exact references: attention limb bmm, compiled vs eager segments (skip without CUDA)."""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


def test_bmm_exact():
    from bitexact.bmm import bmm
    from bitexact.intops import LIM
    g = torch.Generator(device="cuda").manual_seed(0)
    for Z, M, K, N in [(3, 5, 64, 70), (16, 4, 700, 64), (2, 130, 64, 129)]:
        a = torch.randint(-LIM[3], LIM[3] + 1, (Z, M, K), device="cuda", generator=g)
        b = torch.randint(-LIM[3], LIM[3] + 1, (Z, K, N), device="cuda", generator=g).to(torch.int32)
        ref = torch.stack([torch.tensor([[sum(x * y for x, y in zip(r, c)) for c in bz.t().tolist()]
                                         for r in az.tolist()]) for az, bz in zip(a.cpu(), b.long().cpu())])
        assert torch.equal(bmm(a, b).cpu(), ref)
        assert torch.equal(bmm(a, b.transpose(1, 2).contiguous().transpose(1, 2)).cpu(), ref)


def test_int8_segments_compiled_equal():
    from bitexact.int8 import segs
    g = torch.Generator(device="cuda").manual_seed(1)
    eager, comp = segs.Segs(2, True, False), segs.Segs(2, True, True)
    for M in (3, 40):
        x = torch.randint(-(1 << 26), 1 << 26, (M, 2048), device="cuda", generator=g)
        x = x >> torch.randint(0, 20, (M, 1), device="cuda", generator=g)
        gamma = torch.randint(1, 1 << 13, (2048,), device="cuda", generator=g)
        for a, b in zip(eager.pre_norm(x, gamma), comp.pre_norm(x, gamma)):
            assert torch.equal(a, b)
        gu = torch.randint(-(1 << 22), 1 << 22, (M, 16384), device="cuda", generator=g)
        lut = torch.randint(-(1 << 20), 1 << 20, ((64 << 12) + 1,), device="cuda", generator=g)
        for a, b in zip(eager.silu_mul(gu, lut), comp.silu_mul(gu, lut)):
            assert torch.equal(a, b)


def test_int8_division_compiled_equal():
    """rdiv32 / bitlen32 in Triton (float32 division and frexp as Inductor emits them) against the int64 rules."""
    from test_int8_ops import rdiv_cases

    from bitexact.int8.ops import bitlen32, rdiv32
    from bitexact.intops import bitlen, rdiv
    a, b = (t.cuda() for t in rdiv_cases(torch.Generator().manual_seed(6)))
    assert torch.equal(torch.compile(rdiv32)(a, b), rdiv(a, b))
    x = torch.randint(0, 1 << 62, (20000,), device="cuda") >> torch.randint(0, 62, (20000,), device="cuda")
    x = torch.cat([x, (1 << torch.arange(1, 62, device="cuda")) - 1, 1 << torch.arange(0, 62, device="cuda")])
    assert torch.equal(torch.compile(bitlen32)(x), bitlen(x))


def test_exact_norm_compiled_equal():
    from bitexact.exact.segs import Segs
    g = torch.Generator(device="cuda").manual_seed(2)
    eager, comp = Segs(False), Segs(True)
    for M in (1, 37):
        x = torch.randint(-(1 << 30), 1 << 30, (M, 2048), device="cuda", generator=g)
        x = x >> torch.randint(0, 24, (M, 1), device="cuda", generator=g)
        gm = torch.randint(-255, 256, (2048,), device="cuda", generator=g)
        ge = torch.randint(-14, -6, (2048,), device="cuda", generator=g)
        for a, b in zip(eager.pre_norm(x, gm, ge), comp.pre_norm(x, gm, ge)):
            assert torch.equal(a.cpu(), b.cpu())
