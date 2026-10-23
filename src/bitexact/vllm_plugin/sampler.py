"""Integer sampler for the int8 engine in vLLM.

vLLM's sampler works on float32 logits with its own float Gumbel noise. We keep it but make its choice
forced: the model's compute_logits stores the exact integer logits (STASH), and this logits processor
(vLLM V2 model runner interface) replaces every row by 0 at the integer-sampled token and -inf
elsewhere. Temperature, top-k/top-p and vLLM's random draw then can only return that token.

Integer sampling rule (bitexact.sampling):
  T = 0 (vLLM greedy): first argmax of the integer logits.
  T > 0: first argmax of round(l * td / tn) + g, T = tn / td (rational, denominator <= 100), g =
         integer Gumbel noise at 16 bits from the PCG64 stream keyed (SEED_BASE, seed, t) (generated on the
         GPU by gpu_pcg.py, the same words as numpy's stream), where seed is the request's
         SamplingParams.seed (0 when unset) and t is the index of the generated token (0 for the first
         one). Same key, same noise, on any device and in any batch.
top_k / top_p / min_p / penalties are not part of the integer rule (ignored by design).

Logit hashes: with BITEXACT_HASH_LOG=<file> every sampled row appends "seed t sha16 token" (sha16 =
first 16 hex digits of SHA-256 over the int64 logits row), for invariance tests.
"""

import hashlib
import os

import torch
from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor

from bitexact.int8.consts import FR
from bitexact.intops import first_argmax
from bitexact.sampling import gumbel, sample, temp_ratio


class _Stash:
    logits = None            # int64 [rows, V] from the latest compute_logits
    active = False           # set when the integer model is loaded in this process


STASH = _Stash()


_NOISY = None


def _noisy_guess(li, raw, lnt, tn, td):
    """Gumbel-max choice (sampling.sample + gumbel), compiled into fused kernels. Integer ops only
    (bitlen's float64 frexp is corrected to the exact integer), so it equals the eager result."""
    global _NOISY
    if _NOISY is None:
        def f(li, raw, lnt, tn: int, td: int):
            return sample(li, tn, td, gumbel(raw, lnt, FR))
        _NOISY = torch.compile(f, dynamic=True) if os.environ.get("BITEXACT_COMPILE", "1") == "1" else f
    return _NOISY(li, raw, lnt, tn, td)


def integer_sample(li, seeds, ts, temps, lnt):
    """li int64 [M, V]; per row seed, generated index t, (tn, td). Returns int64 [M] token ids."""
    from bitexact.vllm_plugin import gpu_pcg
    out = torch.empty(li.shape[0], dtype=torch.long, device=li.device)
    groups = {}
    for r, tt in enumerate(temps):
        groups.setdefault(tt, []).append(r)
    for (tn, td), rows in groups.items():
        idx = torch.tensor(rows, device=li.device)
        if tn == 0:
            out[idx] = first_argmax(li[idx])
            continue
        raw = gpu_pcg.raw_rows([(seeds[r], max(ts[r], 0)) for r in rows], li.device)
        out[idx] = _noisy_guess(li[idx], raw, lnt, tn, td)
    return out


class IntSampler(LogitsProcessor):
    def __init__(self, vllm_config, req_states):
        self.rs = req_states
        n = req_states.max_num_reqs
        self.seed = [0] * n
        self.temp = [(0, 1)] * n
        self.lnt = None
        self.log = os.environ.get("BITEXACT_HASH_LOG")

    def add_request(self, req_idx, sampling_params):
        if not STASH.active:
            return False
        self.seed[req_idx] = int(sampling_params.seed or 0)
        self.temp[req_idx] = temp_ratio(sampling_params.temperature)
        return True

    def apply(self, logits, ctx):
        if not STASH.active:
            return logits
        li = STASH.logits
        assert li is not None and li.shape[0] == logits.shape[0], "integer logits rows != vLLM rows"
        if self.lnt is None:
            from bitexact.vllm_plugin.model import MODEL
            self.lnt = MODEL.m.ln
        slots = ctx.expanded_idx_mapping.tolist()
        pos = ctx.pos.tolist()
        plen = self.rs.prompt_len.np
        ts = [pos[r] + 1 - int(plen[s]) for r, s in enumerate(slots)]
        seeds = [self.seed[s] for s in slots]
        y = integer_sample(li, seeds, ts, [self.temp[s] for s in slots], self.lnt)
        if self.log:
            self._log(li, seeds, ts, y)
        logits.fill_(float("-inf"))
        logits.scatter_(1, y[:, None], 0.0)
        return logits

    def _log(self, li, seeds, ts, y):
        rows = li.cpu().numpy()
        ys = y.tolist()
        with open(self.log, "a") as f:
            for r in range(len(seeds)):
                if ts[r] >= 0:                           # rows of unfinished prefill chunks are discarded
                    h = hashlib.sha256(rows[r].tobytes()).hexdigest()[:16]
                    f.write(f"{seeds[r]} {ts[r]} {h} {ys[r]}\n")
