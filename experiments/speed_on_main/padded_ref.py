"""score_padded of bitexact.int8.scorer (exp/speed-final), for trees without that module: the reference flow."""

import hashlib

import numpy as np
import torch

from bitexact.llama import PAD


def sha_rows(lg):
    a = lg.long().cpu().numpy()
    return [hashlib.sha256(a[t].tobytes()).hexdigest()[:16] for t in range(a.shape[0])]


def logp_rows(lg, tgt):
    """log p of tgt [n] under int64 logits rows lg [n, V] at 16 fractional bits (float64 log-softmax)."""
    return (lg.double() * 2.0 ** -16).log_softmax(-1)[torch.arange(lg.shape[0], device=lg.device), tgt]


def score_padded(m, batch, chunk=256, want_hash=False):
    """Reference flow: right-padded batch, prefill in position chunks, lm_head + log-softmax per response chunk."""
    _check_prompts(batch)
    seqs = [p + r for p, r in batch]
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
    pos = torch.arange(L).expand(len(seqs), L).contiguous()
    ids, pos = ids.to(m.dev), pos.to(m.dev)
    cache = m.new_cache(len(batch), L)
    h = torch.cat([m.forward(ids[:, c:min(L, c + chunk)], pos[:, c:min(L, c + chunk)], cache, S=min(L, c + chunk))
                   for c in range(0, L, chunk)], 1)
    out = []
    for j, (p, r) in enumerate(batch):
        lps, hs = [], []
        for c0 in range(0, len(r), chunk):
            c1 = min(len(r), c0 + chunk)
            lg = m.logits(h[j, len(p) - 1 + c0:len(p) - 1 + c1])
            hs += sha_rows(lg) if want_hash else []
            lps.append(logp_rows(lg, torch.tensor(r[c0:c1], device=lg.device)))
        out.append((torch.cat(lps).cpu().numpy() if lps else np.empty(0), hs if want_hash else None))
    m.check()
    return out


def _check_prompts(batch):
    """The log p of the first response token is read at the last prompt token: a prompt needs one."""
    for i, (p, _) in enumerate(batch):
        if len(p) == 0:
            raise ValueError(f"batch[{i}]: empty prompt (the first response token has no context to be scored from)")
