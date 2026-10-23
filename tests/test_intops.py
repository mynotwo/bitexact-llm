"""Shared integer rules against Python integer references (CPU)."""

import math

import torch

from bitexact.intops import LIM, bitlen, first_argmax, isqrt, join_limbs, rsh, split_limbs, stack_limbs


def test_bitlen_rsh():
    v = [0, 1, 2, 3, 255, 256, (1 << 40) - 1, 1 << 40, (1 << 62) + 5]
    assert bitlen(torch.tensor(v)).tolist() == [x.bit_length() for x in v]
    x = torch.tensor([5, -5, 6, -6, 7, -7, 1 << 50])
    for s in [-3, 0, 1, 2]:
        ref = [math.floor(a / 2 ** s + 0.5) if s > 0 else a * 2 ** -s for a in x.tolist()]
        assert rsh(x, s).tolist() == ref
        assert rsh(x, torch.full_like(x, s)).tolist() == ref


def test_isqrt():
    xs = [0, 1, 2, 3, 4, 15, 16, 17, (1 << 60) - 1, 1 << 60, (1 << 62) - 1, 12345678987654321]
    assert isqrt(torch.tensor(xs)).tolist() == [math.isqrt(x) for x in xs]


def test_limbs_roundtrip():
    g = torch.Generator().manual_seed(0)
    for n in (1, 2, 3):
        x = torch.randint(-LIM[n], LIM[n] + 1, (5, 40), generator=g)
        x[0, 0], x[0, 1] = LIM[n], -LIM[n]
        lo, r = split_limbs(x, n)
        assert not bool(r.any()) and all(int(t.abs().max()) <= 128 for t in lo)
        assert torch.equal(sum(t << (8 * i) for i, t in enumerate(lo)), x)
        a = stack_limbs(x, n)
        assert a.dtype == torch.int8 and a.shape[0] % 8 == 0 and a.shape[0] >= 24
        w = torch.randint(-127, 128, (40, 7), generator=g)
        assert torch.equal(join_limbs(a.long() @ w, 5, n), x @ w)
    _, r = split_limbs(torch.tensor([LIM[2] + 1]), 2)
    assert bool(r.any())


def test_first_argmax_ties():
    v = torch.tensor([[1, 3, 3, 2], [5, 5, 5, 5]])
    assert first_argmax(v).tolist() == [1, 0]
