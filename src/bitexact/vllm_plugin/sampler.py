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

No host-device sync: rows map to request slots through vLLM's host copy of the batch (idx_mapping_np), and
t comes from its host sequence lengths (seq_lens_upper_bound_np, exact without speculative decoding), so the
step's tokens stay on the device and vLLM's async scheduling can overlap host work with the forward pass.

Logit hashes: with BITEXACT_HASH_LOG=<file> every sampled row appends "seed t sha16 token" (sha16 =
first 16 hex digits of SHA-256 over the int64 logits row), for invariance tests. That mode also checks the
host t against vLLM's device positions (a sync, test runs only).
"""

import hashlib
import os

import numpy as np
import torch
from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor

from bitexact.int8.consts import FR
from bitexact.intops import first_argmax
from bitexact.vllm_plugin import tri_sample
from bitexact.sampling import gumbel, sample, temp_ratio


class _Stash:
    logits = None            # int64 [rows, V] from the latest compute_logits
    greedy = None            # int64 [rows], first argmax of each logits row (same compute_logits)
    active = False           # set when the integer model is loaded in this process
    head = None              # tri_sample.Head of the latest compute_logits (the int64 logits are not stored)
    strict = False           # this step: some request needs the real float logits (see _strict)
    sampler = None           # the IntSampler of the process


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


def integer_sample(li, seeds, ts, temps, lnt, greedy=None):
    """li int64 [M, V]; per row seed, generated index t, (tn, td). Returns int64 [M] token ids. greedy: the
    first argmax of each row when already computed (tri_logits)."""
    from bitexact.vllm_plugin import gpu_pcg
    out = torch.empty(li.shape[0], dtype=torch.long, device=li.device)
    groups = {}
    for r, tt in enumerate(temps):
        groups.setdefault(tt, []).append(r)
    for (tn, td), rows in groups.items():
        whole = len(rows) == li.shape[0]                 # one temperature in the batch: no gather / scatter
        idx = None if whole else torch.tensor(rows).pin_memory().to(li.device, non_blocking=True)
        lg = li if whole else li[idx]
        if tn == 0:
            y = first_argmax(lg) if greedy is None else greedy if whole else greedy[idx]
        else:
            raw = gpu_pcg.raw_rows([(seeds[r], max(ts[r], 0)) for r in rows], li.device)
            y = _noisy_guess(lg, raw, lnt, tn, td)
        if whole:
            return y
        out[idx] = y
    return out


def _strict(sp):
    """True if the request uses something that reads or edits vLLM's float logits before the token is drawn
    (logprobs, penalties, bias, allowed / bad words, min_tokens, structured output, extra processors).
    Without such requests compute_logits hands vLLM one-hot logits at the integer argmax and greedy batches
    skip the logits processing step altogether."""
    return bool(sp.logprobs is not None or sp.prompt_logprobs is not None or sp.presence_penalty
                or sp.frequency_penalty or sp.repetition_penalty != 1.0 or sp.logit_bias or sp.allowed_token_ids
                or sp.bad_words or getattr(sp, "bad_words_token_ids", None) or sp.min_tokens or getattr(sp, "structured_outputs", None) is not None
                or getattr(sp, "logits_processors", None) or getattr(sp, "extra_args", None))


def _request_state(lp):
    """The model runner's RequestState behind the LogitsProcRequestState view (they share their tensors), or
    False. vLLM gives logits processors no callback when a request finishes."""
    import gc
    for o in gc.get_referrers(lp.total_len):
        if getattr(o, "total_len", None) is lp.total_len and hasattr(o, "req_id_to_index"):
            return o
    return False


class IntSampler(LogitsProcessor):
    def __init__(self, vllm_config, req_states):
        self.rs = req_states
        n = req_states.max_num_reqs
        self.seed = np.zeros(n, dtype=np.int64)         # per request slot
        self.tn = np.zeros(n, dtype=np.int64)           # temperature T = tn / td (tn = 0: greedy)
        self.td = np.ones(n, dtype=np.int64)
        self.lnt = None
        self.log = os.environ.get("BITEXACT_HASH_LOG")
        self.strict = np.zeros(n, dtype=bool)
        self.active = None
        STASH.sampler = self

    def add_request(self, req_idx, sampling_params):
        if not STASH.active:
            return False
        self.seed[req_idx] = int(sampling_params.seed or 0)
        self.tn[req_idx], self.td[req_idx] = temp_ratio(sampling_params.temperature)
        self.strict[req_idx] = bool(_strict(sampling_params) or self.log)
        return bool(self.strict[req_idx] or self.tn[req_idx] != 0)

    def any_noisy(self):
        """Does a running request sample with temperature? (then the noisy kernel covers all rows)."""
        if self.active is None:
            self.active = _request_state(self.rs)
        if self.active is False:
            return True
        return bool(self.tn[self._running()].any())

    def any_strict(self):
        """Does a running request need the real float logits? Host only. vLLM's warm-up requests (every feature on)
        leave flags in slots that are free, so only the slots of running requests count; if the runner's
        RequestState cannot be found, every flag counts (the float path: slower, never wrong)."""
        if self.active is None:
            self.active = _request_state(self.rs)
        if self.active is False:
            return bool(self.strict.any())
        return bool(self.strict[self._running()].any())

    def _running(self):
        d = self.active.req_id_to_index
        return np.fromiter(d.values(), dtype=np.intp, count=len(d))

    def apply(self, logits, ctx):
        if not STASH.active:
            return logits
        head = STASH.head
        li = STASH.logits if head is None else None
        n = head.M if head is not None else li.shape[0]
        assert (head is not None or li is not None) and n == logits.shape[0], "integer logits rows != vLLM rows"
        if self.lnt is None:
            from bitexact.vllm_plugin.model import MODEL
            self.lnt = MODEL.m.ln
        slots = ctx.idx_mapping_np
        plen = self.rs.prompt_len.np
        if len(slots) == n:                              # one logits row per request
            rows = slots
            ts = ctx.seq_lens_upper_bound_np[:n].astype(np.int64) - plen[slots].astype(np.int64)
        else:
            # Speculative decoding: a request owns 1 + k rows (its last token and k drafts). The host does not
            # know how many drafts were accepted last step, so the generated index of each row comes from the
            # device positions (row at position p samples the token at p + 1), one sync per step. Row j of a
            # request is sampled exactly as plain decoding samples that position (same seed, same t), so a draft
            # is accepted iff it equals the token plain decoding would have produced there.
            rows = ctx.expanded_idx_mapping.cpu().numpy().astype(np.intp)
            ts = ctx.pos.cpu().numpy().astype(np.int64) + 1 - plen[rows].astype(np.int64)
        seeds = self.seed[rows]
        tn, td = self.tn[rows], self.td[rows]
        wide = bool(seeds.max(initial=0) >= 1 << 32 or ts.max(initial=0) >= 1 << 32)   # GPU seeding needs < 2^32
        if head is None or wide:
            temps = list(zip(tn.tolist(), td.tolist()))
            keys = list(zip(seeds.tolist(), np.maximum(ts, 0).tolist()))
        if head is None:
            y = integer_sample(li, seeds.tolist(), ts.tolist(), temps, self.lnt, STASH.greedy)
        else:       # one pass over the GEMM output (rows with tn = 0 are greedy) that also writes vLLM's logits
            assert logits.is_contiguous()
            st = tri_sample.row_state(keys, temps) if wide else np.stack([seeds, np.maximum(ts, 0), tn, td], 1)
            y = tri_sample.noisy(head, st, self.lnt, out=logits)
        if self.log:
            li = head.logits() if li is None else li
            pos = ctx.pos.tolist()
            if len(slots) == n:
                assert ctx.expanded_idx_mapping.tolist() == slots.tolist(), "host slots != device slots"
                assert ts.tolist() == [pos[r] + 1 - int(plen[s]) for r, s in enumerate(slots)], "host t != device t"
            self._log(li, seeds.tolist(), ts.tolist(), y)
        if head is None:
            logits.fill_(float("-inf"))
            logits.scatter_(1, y[:, None], 0.0)
        return logits                                    # head path: noisy() left -inf rows with 0 at the token

    def _log(self, li, seeds, ts, y):
        rows = li.cpu().numpy()
        ys = y.tolist()
        with open(self.log, "a") as f:
            for r in range(len(seeds)):
                if ts[r] >= 0:                           # rows of unfinished prefill chunks are discarded
                    h = hashlib.sha256(rows[r].tobytes()).hexdigest()[:16]
                    f.write(f"{seeds[r]} {ts[r]} {h} {ys[r]}\n")
