"""The packed scorer (int8/scorer.py) gives, for every response token, the same log p bits as the eager padded
reference, for any batch composition and order, and on CPU."""

import random

import numpy as np
import pytest
import torch

from bitexact.int8.model import IntLlama
from bitexact.int8.scorer import Scorer, score_padded

pytestmark = pytest.mark.slow
CUDA = torch.cuda.is_available()


def _batch(seed, n, lo, hi):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        P, R = rng.randint(lo, hi), rng.randint(2, hi)
        out.append(([128000] + [rng.randrange(1000, 100000) for _ in range(P - 1)],
                    [rng.randrange(1000, 100000) for _ in range(R)]))
    return out


def _bits(res):
    return [x[0].view(np.int64) for x in res]


@pytest.fixture
def int8_const(const):
    kind, c = const
    if kind != "int8":
        pytest.skip("int8 engine only")
    return c


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@torch.no_grad()
def test_packed_scorer_equals_eager_reference(int8_const):
    ref_m = IntLlama(int8_const, "cuda", compiled=False, fused=False)
    seqs = _batch(0, 7, 5, 150) + _batch(1, 2, 200, 330)                   # short, 64-tile and 16-tile tails, > 256
    ref = _bits(score_padded(ref_m, seqs))
    m = IntLlama(int8_const, "cuda")
    for rows in (64, 256, 2048):                                           # chunk boundaries inside sequences
        sc = Scorer(m, rows=rows)
        got = _bits(sc.score(seqs))
        for a, b in zip(ref, got):
            assert np.array_equal(a, b)
        order = list(range(len(seqs)))
        random.Random(3).shuffle(order)
        got = _bits(sc.score([seqs[i] for i in order]))
        for j, i in enumerate(order):
            assert np.array_equal(ref[i], got[j])
        for i in (0, 4):                                                   # alone
            assert np.array_equal(ref[i], _bits(sc.score([seqs[i]]))[0])


@torch.no_grad()
def test_cpu_scorer_is_reference_flow(int8_const):
    m = IntLlama(int8_const, "cpu")
    assert not m.fused
    seqs = _batch(2, 2, 4, 12)
    res = Scorer(m).score(seqs, want_hash=True)
    for (lp, hs), (p, r) in zip(res, seqs):
        assert len(lp) == len(hs) == len(r)
