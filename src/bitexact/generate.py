"""Loading an engine from its constants file, generation with the integer sampler, and logits hashes."""

import hashlib

import torch

from bitexact.sampling import gumbel, raw_rows, sample


def load(const, device="cuda", **kw):
    """Constants (a path or a dict from exact.consts.build / int8.consts.build) -> ExactLlama or IntLlama.
    Keyword arguments go to the model (gemm=, compiled=, n_layers=, nl=)."""
    c = torch.load(const, map_location="cpu", weights_only=False) if isinstance(const, str) else const
    if "lm.limbs" in c:
        from bitexact.exact.model import ExactLlama
        return ExactLlama(c, device, **kw)
    from bitexact.int8.model import IntLlama
    return IntLlama(c, device, **kw)


def logits_sha(row):
    """SHA-256 (hex) of one int64 logits row: equal hashes <=> bit-identical logits."""
    return hashlib.sha256(row.contiguous().cpu().numpy().tobytes()).hexdigest()


def choose(model, logits, keys, tn, td):
    """Tokens for logits rows [M, V]: greedy (tn = 0) or Gumbel-max at T = tn / td with the noise of
    keys [(seed, t)] (one per row)."""
    g = gumbel(raw_rows(keys).to(logits.device), model.ln, model.FR) if tn else None
    return sample(logits, tn, td, g)


@torch.no_grad()
def generate(model, ids, max_new, tn=0, td=1, seed=0):
    """One prompt (list of token ids): prefill, then max_new tokens through the KV cache. Generated token t
    uses the noise keyed (seed, t). Returns (tokens, logits hashes per generated position)."""
    dev = model.dev
    P = len(ids)
    cache = model.new_cache(1, P + max_new)
    h = model.forward(torch.tensor([ids], device=dev), torch.arange(P, device=dev)[None], cache)[:, -1]
    toks, hashes = [], []
    for t in range(max_new):
        lg = model.logits(h)
        y = int(choose(model, lg, [(seed, t)], tn, td)[0])
        toks.append(y)
        hashes.append(logits_sha(lg[0]))
        if t + 1 < max_new:
            h = model.forward(torch.tensor([[y]], device=dev), torch.tensor([[P + t]], device=dev), cache)[:, -1]
    model.check()
    return toks, hashes
