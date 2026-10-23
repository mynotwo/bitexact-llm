"""End-to-end invariance of the integer engines: the logits of a token are the same bits whether it is
computed alone, inside a right-padded batch, through the KV cache one token at a time, with fused
(torch.compile) segments, with the other GEMM implementation, or on the CPU."""

import pytest
import torch

from bitexact import load
from bitexact.llama import PAD

pytestmark = pytest.mark.slow
CUDA = torch.cuda.is_available()
SEQS = [[128000, 791, 23070, 19044, 6267, 304, 813, 1515, 430], [128000, 755, 11117, 5401],
        [128000, 644, 220, 9674, 17, 279, 33715, 8813, 279, 18671, 13962, 11, 1405]]


def _full(m, seqs):
    """Hidden-state logits of every position, all sequences in one right-padded batch."""
    Lm = max(map(len, seqs))
    ids = torch.full((len(seqs), Lm), PAD, dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = torch.tensor(s)
    pos = torch.arange(Lm).expand(len(seqs), Lm).contiguous()
    h = m.forward(ids.to(m.dev), pos.to(m.dev), m.new_cache(len(seqs), Lm))
    return [m.logits(h[r, :len(s)]).cpu() for r, s in enumerate(seqs)]


def _incr(m, s):
    cache = m.new_cache(1, len(s))
    rows = []
    for t, tok in enumerate(s):
        h = m.forward(torch.tensor([[tok]], device=m.dev), torch.tensor([[t]], device=m.dev), cache)
        rows.append(m.logits(h[:, 0]).cpu())
    return torch.cat(rows)


def _variants(kind):
    out = [("cpu", {})]
    if CUDA:
        out.append(("cuda", {}))
        out.append(("cuda", {"compiled": True}))
        if kind == "exact":
            out.append(("cuda", {"gemm": "cublas"}))
    return out


@torch.no_grad()
def test_engine_invariance(const):
    kind, c = const
    ref = None
    for dev, kw in _variants(kind):
        m = load(c, dev, **kw)
        full = _full(m, SEQS)
        alone = [_full(m, [s])[0] for s in SEQS]
        incr = [_incr(m, s) for s in SEQS]
        m.check()
        ref = full if ref is None else ref
        for r in range(len(SEQS)):
            assert torch.equal(full[r], ref[r]), (dev, kw, "batch vs reference device")
            assert torch.equal(alone[r], ref[r]), (dev, kw, "alone")
            assert torch.equal(incr[r], ref[r]), (dev, kw, "kv cache")
