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


@torch.no_grad()
def test_scorer_empty_prompt_raises_and_empty_responses(int8_const):
    m = IntLlama(int8_const, "cpu")
    sc = Scorer(m)
    with pytest.raises(ValueError, match="empty prompt"):
        sc.score([([1, 2], [3]), ([], [4])])
    with pytest.raises(ValueError, match="empty prompt"):
        score_padded(m, [([], [4])])
    out = sc.score([([128000, 5], []), ([128000], [])], want_hash=True)           # all responses empty
    assert [len(lp) for lp, _ in out] == [0, 0] and [h for _, h in out] == [[], []]
    assert [len(lp) for lp, _ in sc.score([([128000, 5], []), ([128000], [])])] == [0, 0]
    assert sc.score([]) == []
    mixed = sc.score([([128000, 5, 6], [7, 8]), ([128000], []), ([128000], [9])])  # empty response between others
    assert [len(lp) for lp, _ in mixed] == [2, 0, 1]


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@torch.no_grad()
def test_packed_scorer_short_sequences(int8_const):
    """1-token prompt, 1-token response, empty response (alone and among others) equal the eager reference."""
    ref_m = IntLlama(int8_const, "cuda", compiled=False, fused=False)
    m = IntLlama(int8_const, "cuda")
    cases = [([128000], [11]), ([128000, 7], [12]), ([128000], [13, 14, 15]), ([128000, 3, 4], []),
             ([128000], [])]
    sc = Scorer(m)
    for batch in (cases, cases[:1], cases[3:4] + cases[:1], cases[::-1]):
        got = sc.score(batch, want_hash=True)
        assert [len(lp) for lp, _ in got] == [len(r) for _, r in batch]
        for (lp, hs), (p, r) in zip(got, batch):
            if r:
                one = score_padded(ref_m, [(p, r)], want_hash=True)[0]
                assert np.array_equal(lp.view(np.int64), one[0].view(np.int64)) and hs == one[1]
