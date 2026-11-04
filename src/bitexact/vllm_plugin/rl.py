"""RL support for the vLLM plugin: weights replaced in place between steps, integer log p of sampled tokens,
and teacher-forced integer scoring inside the prefill pass.

  update_weights(m, c)   copy a new int8 constants set (bitexact.int8.consts layout, tables excluded) into the
                         live IntLlama; same storage, so CUDA graphs and compiled regions see the new values.
                         The load-time bound (ops.post_bound < 2^38 for every linear layer) is checked on the new
                         weights first; on failure OverflowError is raised and nothing is written.
  int_logp(lg, tok, lnt) integer log p of tok from int64 logits rows (exact conversion to float64 of an integer
                         at 2^-40); a pure function of the logits row.
  install()              (in the process that runs the engine, V1 multiprocessing off) enables
                           - RL.lp: list collecting (seeds, ts, log p) of every sampled row (sampler.py hook);
                           - scoring requests: SamplingParams(max_tokens=1, skip_reading_prefix_cache=True,
                             extra_args={"bitexact_score_from": P}) over prompt + response (P = prompt length)
                             get int_logp of tokens P..L-1 in RL.scores[request_id] (device float64), computed
                             from the same prefill hidden states with model.m.logits.
"""

import numpy as np
import torch

from bitexact.int8.consts import FR
from bitexact.int8.ops import post_bound
from bitexact.intops import rsh
from bitexact.sampling import FL, LN2Q, ln_q

LSE_FE, LSE_MIN = 44, -40
_LSE_TAB = {}
POST_LIMIT = 1 << 38
LIN = ("wqkv", "wo", "wgu", "wd")


def lse_table(device):
    """e^x at LSE_FE = 44 bits for x in [LSE_MIN, 0] at step 2^-12 (float64 on the CPU, rounded once).
    Finer than the engine's FE = 30 table so that the ~10^5 tiny vocabulary terms of a peaked row do not
    underflow to 0 (with FE = 30 that biases log p by up to ~1e-5); V * 2^44 < 2^62."""
    k = str(device)
    if k not in _LSE_TAB:
        xs = torch.arange(LSE_MIN << 12, 1, dtype=torch.float64) / 4096
        _LSE_TAB[k] = torch.round(torch.exp(xs) * 2.0 ** LSE_FE).long().to(device)
    return _LSE_TAB[k]


def int_logp(lg, tok, lnt):
    """Integer-defined log p of `tok` [n] from int64 logits rows lg [n, V] at FR bits (T = 1).

    log-sum-exp from lse_table (linear interpolation from 16 to 12 fractional bits, rounded) and the
    engine's ln table (ln_q, FL = 40 bits); the result is an integer at 2^-40 returned as float64 (exact
    conversion). A pure integer function of the logits row: equal rows give equal bits on any device or
    batch. Approximation error vs float64 log-softmax ~1e-8."""
    expt = lse_table(lg.device)
    mx = lg.amax(-1, keepdim=True)
    z = lg - mx
    zc = z.clamp(min=LSE_MIN << FR)
    zi = zc >> (FR - 12)
    r = zc - (zi << (FR - 12))
    i0 = zi - (LSE_MIN << 12)
    last = expt.shape[0] - 1
    a = expt[i0]
    b = expt[(i0 + 1).clamp(max=last)]
    e = a + rsh((b - a) * r, FR - 12)
    S = e.sum(-1)                                                # >= 2^44 (the max entry), < 2^62
    lse = ln_q(S, lnt) - LSE_FE * LN2Q                           # ln(sum e^z) at FL bits
    zt = z.gather(1, tok[:, None])[:, 0].clamp(min=-(1 << 36))   # (zt << 24) stays < 2^62
    return ((zt << (FL - FR)) - lse).double() / 2.0 ** FL


@torch.no_grad()
def update_weights(m, c, limit=POST_LIMIT):
    """Copy constants `c` (emb, {i}.ga/gm/wqkv/wo/wgu/wd[.m|.fm], gf, lm[.m|.fm]) into IntLlama `m` in place.
    Returns the fraction of int8 weight entries that changed. Raises OverflowError (before writing anything)
    if post_bound >= limit for a linear layer of the new weights, which is what the engine's load-time
    check (m.post_check False) relies on; clears the sticky device-side error flag."""
    lins = [(f"{i}.{n}") for i in range(m.L) for n in LIN] + ["lm"]
    for k in lins:
        pb = post_bound(c[k], c[k + ".m"], int(m._fmsh(c[k + ".fm"])), m.nl)
        if pb >= limit:
            raise OverflowError(f"post_bound {pb} >= 2^{limit.bit_length() - 1} for {k}")
    changed = total = 0
    m.emb.copy_(c["emb"])
    for i, ly in enumerate(m.layers):
        ly["ga"].copy_(c[f"{i}.ga"])
        ly["gm"].copy_(c[f"{i}.gm"])
        for n in LIN:
            w, mm, fmsh = ly[n]
            changed += int((w != c[f"{i}.{n}"]).sum())
            total += w.numel()
            w.copy_(c[f"{i}.{n}"])
            mm.copy_(c[f"{i}.{n}.m"])
            fmsh.copy_(m._fmsh(c[f"{i}.{n}.fm"]))
    m.gf.copy_(c["gf"])
    w, mm, fmsh = m.lm
    changed += int((w != c["lm"]).sum())
    total += w.numel()
    w.copy_(c["lm"])
    mm.copy_(c["lm.m"])
    fmsh.copy_(m._fmsh(c["lm.fm"]))
    m.err.zero_()
    return changed / total


class _RL:
    def __init__(self):
        self.lp = None                  # list of (seeds, ts, device float64 log p) while collecting
        self.start = {}                 # request id -> first scored token index
        self.scores = {}                # request id -> device float64 [n scored tokens]
        self.cov = {}                   # request id -> bool numpy coverage of scores
        self.installed = False

    def take_lp(self):
        """{(seed, t): log p} of the sampled rows since collecting started; stops collecting."""
        out, lst = {}, self.lp
        self.lp = None
        if lst:
            seeds = np.concatenate([s for s, _, _ in lst])
            ts = np.concatenate([t for _, t, _ in lst])
            lp = torch.cat([x for _, _, x in lst]).cpu().numpy()
            for s, t, x in zip(seeds.tolist(), ts.tolist(), lp.tolist()):
                if t >= 0:
                    out[(s, t)] = x
        return out

    def take_scores(self, req_id):
        """Scores of the request with the client's id (the worker's id is that plus a '-suffix' vLLM adds)."""
        key = next(k for k in self.scores if k == req_id or k.rsplit("-", 1)[0] == req_id)
        assert self.cov.pop(key).all(), "scoring request: some rows were not computed"
        self.start.pop(key, None)
        return self.scores.pop(key)


RL = _RL()


def install():
    """Hook vLLM's PromptLogprobsWorker (scoring requests) once per process."""
    if RL.installed:
        return
    from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
    add, cpl = PromptLogprobsWorker.add_request, PromptLogprobsWorker.compute_prompt_logprobs

    def add_request(self, req_id, req_idx, sampling_params):
        ex = sampling_params.extra_args or {}
        if "bitexact_score_from" in ex:
            RL.start[req_id] = int(ex["bitexact_score_from"])
        return add(self, req_id, req_idx, sampling_params)

    def compute_prompt_logprobs(self, logits_fn, hidden_states, input_batch, all_token_ids, num_computed_tokens,
                                prompt_lens):
        out = cpl(self, logits_fn, hidden_states, input_batch, all_token_ids, num_computed_tokens, prompt_lens)
        if RL.start:
            _score_rows(hidden_states, input_batch, all_token_ids, prompt_lens)
        return out

    PromptLogprobsWorker.add_request = add_request
    PromptLogprobsWorker.compute_prompt_logprobs = compute_prompt_logprobs
    RL.installed = True


@torch.no_grad()
def _score_rows(hidden_states, ib, all_token_ids, prompt_lens, chunk=1024):
    from bitexact.vllm_plugin.model import MODEL
    rows, toks, jobs = [], [], []
    for i, rid in enumerate(ib.req_ids):
        P = RL.start.get(rid)
        if P is None:
            continue
        slot = int(ib.idx_mapping_np[i])
        L = int(prompt_lens[slot])
        c0 = int(ib.num_computed_prefill_tokens_np[i])
        c1 = c0 + int(ib.num_scheduled_tokens[i])
        lo, hi = max(P - 1, c0), min(c1, L - 1)          # row r predicts token r + 1; the last row is the sampled one
        if rid not in RL.scores:
            RL.scores[rid] = torch.empty(L - P, dtype=torch.float64, device=hidden_states.device)
            RL.cov[rid] = np.zeros(L - P, dtype=bool)
        if lo >= hi:
            continue
        base = int(ib.query_start_loc_np[i]) - c0
        rows.append(torch.arange(base + lo, base + hi, device=hidden_states.device))
        toks.append(all_token_ids[slot, lo + 1:hi + 1])
        jobs.append((rid, lo - (P - 1), hi - lo))
    if not jobs:
        return
    h = hidden_states[torch.cat(rows)].long()
    t = torch.cat(toks).long()
    lp = torch.cat([int_logp(MODEL.m.logits(h[a:a + chunk]), t[a:a + chunk], MODEL.m.ln)
                    for a in range(0, h.shape[0], chunk)])
    MODEL._check_late()
    o = 0
    for rid, off, n in jobs:
        RL.scores[rid][off:off + n] = lp[o:o + n]
        RL.cov[rid][off:off + n] = True
        o += n
