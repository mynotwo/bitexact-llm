"""Integer Gumbel noise against the float64 formula, and the sampling rule (CPU)."""

import numpy as np
import torch

from bitexact import sampling
from bitexact.llama import V, ln_table


def test_gumbel_matches_float():
    lnt = ln_table()
    for seed, t in [(0, 0), (3, 17), (99, 511)]:
        raw = sampling.raw_rows([(seed, t)])
        g = sampling.gumbel(raw, lnt, 16).double()[0] / 65536
        u = np.clip(np.random.default_rng([sampling.SEED_BASE, seed, t]).random(V), 1e-12, 1 - 1e-12)
        assert float((g - torch.from_numpy(-np.log(-np.log(u)))).abs().max()) < 2e-5
    # extreme u: near 0 and near 1
    w = np.array([[0, 1 << 11, ((1 << 53) - 1) << 11, ((1 << 53) - (1 << 45)) << 11, (1 << 63) - 1,
                   ((1 << 53) - (1 << 45) - 1) << 11, 12345 << 11]], dtype=np.uint64)
    raw = torch.from_numpy(w.view(np.int64))
    g = sampling.gumbel(raw, lnt, 16).double()[0] / 65536
    u = ((raw >> 11) & ((1 << 53) - 1)).double()[0].clamp(sampling.UMIN, 2.0 ** 53 - sampling.UMIN) / 2.0 ** 53
    assert float((g - (-torch.log(-torch.log(u)))).abs().max()) < 2e-5


def test_gumbel_bits_consistent():
    lnt = ln_table()
    raw = sampling.raw_rows([(7, 3)])
    g16, g24 = sampling.gumbel(raw, lnt, 16), sampling.gumbel(raw, lnt, 24)
    assert int((g24 - (g16 << 8)).abs().max()) <= 1 << 7


def test_sample_rule():
    li = torch.tensor([[10, 30, 30, 20]])
    assert sampling.sample(li, 0, 1).tolist() == [1]
    g = torch.tensor([[0, 0, 0, 100]])
    assert sampling.sample(li, 1, 1, g).tolist() == [3]
    assert sampling.temp_ratio(0.7) == (7, 10) and sampling.temp_ratio(0) == (0, 1)
