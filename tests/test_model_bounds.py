"""Invalid exact linear outputs must not hide behind residual cancellation or int64 wrap."""

import pytest
import torch

from bitexact.exact.consts import FR
from bitexact.exact.model import ExactLlama
from bitexact.exact.segs import Segs, lin_add
from bitexact.exact.wmm import epilogue


def parts(value, shift):
    return (torch.tensor([[value]]), (torch.zeros(1, 1, dtype=torch.long),),
            torch.zeros(1, dtype=torch.long), torch.tensor([shift - FR]), torch.zeros(1, dtype=torch.long))


@pytest.mark.parametrize("value,shift", [(1 << 40, 22), (-(1 << 40), 22), (1 << 40, 30), (-(1 << 40), 30)])
def test_epilogue_rejects_bound_and_wrapped_results(value, shift):
    c, rs, d, g, sh = parts(value, shift)
    _, flag = epilogue(c, rs[0], d, g, sh, FR, checked=True)
    assert bool(flag)


def test_bad_projection_cannot_hide_in_residual_sum():
    args = parts(1 << 40, 22)
    out, flag = lin_add(*args, torch.tensor([[-(1 << 62)]]))
    assert out.item() == 0
    assert bool(flag)


def test_logits_latches_bound_violation():
    model = ExactLlama.__new__(ExactLlama)
    model.sg = Segs()
    model.err = torch.tensor(False)
    args = parts(1 << 40, 30)
    model.lm = lambda q: args[:4]
    model.logits_q(torch.zeros(1, 1, dtype=torch.int32), args[4])
    with pytest.raises(OverflowError):
        model.check()


@pytest.mark.parametrize("shift", [-90, -33, -32, -31, -1, 0, 1, 20, 30])
def test_checked_epilogue_preserves_valid_integer_results(shift):
    from fractions import Fraction

    for value in [-1234567, -1, 0, 1, 1234567]:
        c, rs, d, g, sh = parts(value, shift)
        y, flag = epilogue(c, rs[0], d, g, sh, FR, checked=True)
        expected = (Fraction(value) * Fraction(2) ** shift + Fraction(1, 2)).__floor__()
        assert y.item() == expected
        assert not bool(flag)


def test_checked_epilogue_against_independent_big_integer_oracle():
    import random
    from fractions import Fraction

    rng = random.Random(19)
    cases = [(rng.randrange(-(1 << 59), 1 << 59), rng.randrange(-(1 << 50), 1 << 50),
              rng.randrange(31), rng.randrange(-100, 31)) for _ in range(2000)]
    t, b, d, shifts = zip(*cases)
    td = torch.tensor(d)
    y, flag = epilogue(torch.tensor([t]), torch.tensor([b]), td, torch.tensor(shifts) + td - FR,
                       torch.zeros(1, dtype=torch.long), FR, checked=True)
    expected = [(Fraction(v * (1 << dd) + residual) * Fraction(2) ** shift + Fraction(1, 2)).__floor__()
                for v, residual, dd, shift in cases]
    # Per-column calls check each rejection, rather than trusting only an aggregate flag.
    assert bool(flag)
    for i, want in enumerate(expected):
        _, bad = epilogue(torch.tensor([[t[i]]]), torch.tensor([[b[i]]]), td[i:i + 1],
                          torch.tensor([shifts[i] + d[i] - FR]), torch.zeros(1, dtype=torch.long), FR,
                          checked=True)
        assert bool(bad) == (abs(want) >= 1 << 62)
        if not bool(bad):
            assert y[0, i].item() == want
