"""Integer side of the A/B/C RL experiment: per-step BitExact int8 quantization of the fp32 master weights,
the matching dequantized float weights, an integer log p, and batched integer sampling / scoring.

Quantization is bitexact.int8.consts._q_linear applied on the GPU (float64) to the current master weights,
so every RL step yields a valid int8 constants set with freshly recomputed multipliers m_c, exponents fm
(the |acc * m| < 2^62 bound is re-derived per matrix) and gammas at FG bits. The tables (exp, SiLU, RoPE,
ln) do not depend on the weights and are taken from the original constants file.

The float arms use the same quantized weights: W_float = q * m_c * 127 / 2^fm (the integer engine's
effective per-channel scale), RMSNorm gammas round(g * 2^FG) / 2^FG, input embedding round(E * 2^FR) / 2^FR,
lm_head = dequantized int8 embedding (untied, as in the int8 engine).
"""

import math

import torch

from bitexact.int8.consts import FG, FR, QMAX_BITS
from bitexact.int8.model import IntLlama
from bitexact.intops import first_argmax, rsh
from bitexact.llama import PAD, Cache
from bitexact.sampling import FL, LN2Q, ln_q

EOS = 128001
LIN = {"wqkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "wo": ["self_attn.o_proj"],
       "wgu": ["mlp.gate_proj", "mlp.up_proj"], "wd": ["mlp.down_proj"]}


def _q_linear(w):
    """bitexact.int8.consts._q_linear on a GPU tensor with the same results as the CPU builder: the per-row
    vectors (s = amax / 127, m) are computed on the CPU (CUDA turns tensor / scalar into a multiply by the
    reciprocal, which changes s by an ulp and flips ~1e-4 of the rounded weights)."""
    amax = w.abs().amax(1).clamp_min(1e-30).cpu()
    s = amax / 127.0
    q = torch.round(w / s.to(w.device)[:, None]).clamp(-127, 127)
    l1 = q.abs().sum(1).cpu()
    worst = float((l1 * s / 127.0).max()) * 2.0 ** QMAX_BITS * 127.0
    fm = min(60, int(math.floor(62 - math.log2(worst))) - 1)
    m = torch.round(s / 127.0 * 2.0 ** fm).long().to(w.device)
    return q.to(torch.int8), m, fm


def _deq(q, m, fm):
    return (q.double() * (m.double() * 127.0 / 2.0 ** fm)[:, None])


@torch.no_grad()
def quantize(P, n_layers=16):
    """fp32 master params (HF names, tied embedding) -> (int consts dict without tables, float weights dict
    {HF name: fp32 tensor} with lm_head.weight untied, stats)."""
    c, fw, st = {"n_layers": n_layers}, {}, {"fm": []}
    E = P["model.embed_tokens.weight"].double()
    assert float(E.abs().max()) < 2 ** (31 - FR), "embedding out of int32 range"
    c["emb"] = torch.round(E * (1 << FR)).to(torch.int32)
    fw["model.embed_tokens.weight"] = (c["emb"].double() / (1 << FR)).float()
    for i in range(n_layers):
        p = f"model.layers.{i}."
        for name, parts in LIN.items():
            ws = [P[p + n + ".weight"] for n in parts]
            q, m, fm = _q_linear(torch.cat(ws).double())
            c[f"{i}.{name}"], c[f"{i}.{name}.m"], c[f"{i}.{name}.fm"] = q, m, fm
            st["fm"].append(fm)
            d = _deq(q, m, fm).float()
            o = 0
            for n, w in zip(parts, ws):
                fw[p + n + ".weight"] = d[o:o + w.shape[0]]
                o += w.shape[0]
        for key, n in (("ga", "input_layernorm"), ("gm", "post_attention_layernorm")):
            g = torch.round(P[p + n + ".weight"].double() * (1 << FG)).long()
            assert int(g.abs().max()) < 1 << 20, (i, key)
            c[f"{i}.{key}"] = g
            fw[p + n + ".weight"] = (g.double() / (1 << FG)).float()
    g = torch.round(P["model.norm.weight"].double() * (1 << FG)).long()
    assert int(g.abs().max()) < 1 << 20, "gf"
    c["gf"] = g
    fw["model.norm.weight"] = (g.double() / (1 << FG)).float()
    c["lm"], c["lm.m"], c["lm.fm"] = _q_linear(E)
    st["fm"].append(c["lm.fm"])
    fw["lm_head.weight"] = _deq(c["lm"], c["lm.m"], c["lm.fm"]).float()
    return c, fw, st


TABLES = ("exp", "silu", "cos", "sin", "ln")


class IntEngine:
    """One IntLlama whose weights are replaced in place every step (tables loaded once)."""

    def __init__(self, const_path, c0, device, compiled=True):
        base = torch.load(const_path, map_location="cpu", weights_only=False)
        full = dict(c0)
        full.update({k: base[k] for k in TABLES})
        full["sha256"] = None
        del base
        self.m = IntLlama(full, device, compiled=compiled)
        self.prev = None

    @torch.no_grad()
    def load(self, c):
        """Swap in new constants; returns the fraction of int8 weight entries that changed since last load."""
        m, changed, total = self.m, 0, 0
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
        return changed / total

    def check(self):
        """Raises OverflowError if a device-side bound check failed since the last reset."""
        try:
            self.m.check()
        finally:
            self.m.err.zero_()


LSE_FE, LSE_MIN = 44, -40
_LSE_TAB = {}


def lse_table(device):
    """e^x at LSE_FE = 44 bits for x in [LSE_MIN, 0] at step 2^-12 (float64 on the CPU, rounded once).
    Finer than the engine's FE = 30 table so that the ~10^5 tiny vocabulary terms of a peaked row do not
    underflow to 0 (with FE = 30 that biases log p by up to ~1e-5); V * 2^44 < 2^62."""
    k = str(device)
    if k not in _LSE_TAB:
        xs = torch.arange(LSE_MIN << 12, 1, dtype=torch.float64) / 4096
        _LSE_TAB[k] = torch.round(torch.exp(xs) * 2.0 ** LSE_FE).long().to(device)
    return _LSE_TAB[k]


def _lse_parts(lg, expt):
    """(row max, sum_v e) of the log-sum-exp terms, fused into one pass over lg [n, V] by torch.compile. Pure int64
    arithmetic, so the sum is exact in any order."""
    mx = lg.amax(-1, keepdim=True)
    zc = (lg - mx).clamp(min=LSE_MIN << FR)
    zi = zc >> (FR - 12)
    r = zc - (zi << (FR - 12))
    i0 = zi - (LSE_MIN << 12)
    a = expt[i0]
    b = expt[(i0 + 1).clamp(max=expt.shape[0] - 1)]
    e = a + rsh((b - a) * r, FR - 12)
    return mx, e.sum(-1)


_lse_parts_c = {}


def int_logp(lg, tok, lnt):
    """Same value as int_logp_ref (bit for bit), without the [n, V] int64 temporaries: the vocabulary part is one
    compiled kernel pair on CUDA (eager on CPU)."""
    expt = lse_table(lg.device)
    f = _lse_parts
    if lg.device.type == "cuda":
        f = _lse_parts_c.setdefault("f", torch.compile(_lse_parts, dynamic=True))
    mx, S = f(lg, expt)
    lse = ln_q(S, lnt) - LSE_FE * LN2Q
    zt = (lg.gather(1, tok[:, None]) - mx)[:, 0].clamp(min=-(1 << 36))
    return ((zt << (FL - FR)) - lse).double() / 2.0 ** FL


def int_logp_ref(lg, tok, lnt):
    """Reference (eager, ~10 passes over [n, V] int64) of int_logp. Integer-defined log p of `tok` [n] from int64 logits rows lg [n, V] at FR bits (T = 1).

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



def _pad(rows, dev):
    L = max(map(len, rows))
    ids = torch.full((len(rows), L), PAD, dtype=torch.long)
    for i, s in enumerate(rows):
        ids[i, :len(s)] = torch.tensor(s)
    return ids.to(dev), torch.arange(L).expand(len(rows), L).contiguous().to(dev), L


class _SubCache:
    def __init__(self, cache, r0, r1):
        self.k = [t[r0:r1] for t in cache.k]
        self.v = [t[r0:r1] for t in cache.v]

    def write(self, layer, k, v, pos):
        Cache.write(self, layer, k, v, pos)


@torch.no_grad()
def generate(eng, prompts, seeds, max_new, greedy=False, bs=128, prefill_bs=16):
    """Batched integer sampling. prompts: list of token lists; seeds: request seeds (Gumbel key (seed, t),
    as the vLLM plugin). Returns (tokens, sampler log p) per request; stops a row at EOS."""
    from bitexact.vllm_plugin.sampler import integer_sample
    m = eng.m
    toks, lps = [None] * len(prompts), [None] * len(prompts)
    for b0 in range(0, len(prompts), bs):
        idx = list(range(b0, min(b0 + bs, len(prompts))))
        B = len(idx)
        Ps = torch.tensor([len(prompts[i]) for i in idx], device=m.dev)
        Lp = int(Ps.max())
        cache = m.new_cache(B, Lp + max_new)
        h = torch.empty(B, m.emb.shape[1], dtype=torch.long, device=m.dev)
        for r0 in range(0, B, prefill_bs):
            r1 = min(B, r0 + prefill_bs)
            ids, pos, _ = _pad([prompts[i] for i in idx[r0:r1]], m.dev)
            hh = m.forward(ids, pos, _SubCache(cache, r0, r1))
            h[r0:r1] = hh[torch.arange(r1 - r0, device=m.dev), Ps[r0:r1] - 1]
        out_t = torch.full((B, max_new), EOS, dtype=torch.long, device=m.dev)
        out_l = torch.zeros((B, max_new), dtype=torch.float64, device=m.dev)
        live = torch.ones(B, dtype=torch.bool, device=m.dev)
        n = max_new
        for t in range(max_new):
            lg = m.logits(h)
            if greedy:
                y = first_argmax(lg)
            else:
                y = integer_sample(lg, [seeds[i] for i in idx], [t] * B, [(1, 1)] * B, m.ln)
            out_t[:, t] = torch.where(live, y, torch.full_like(y, EOS))
            out_l[:, t] = int_logp(lg, y, m.ln)
            live &= y != EOS
            if t % 16 == 15 and not bool(live.any()):
                n = t + 1
                break
            if t + 1 == max_new:
                break
            h = m.forward(y.view(B, 1), (Ps + t).view(B, 1), cache, S=Lp + t + 1)[:, 0]
        ot, ol = out_t[:, :n].tolist(), out_l[:, :n].tolist()
        for j, i in enumerate(idx):
            k = ot[j].index(EOS) + 1 if EOS in ot[j] else n
            toks[i], lps[i] = ot[j][:k], ol[j][:k]
    eng.check()
    return toks, lps


@torch.no_grad()
def score(eng, prompts, resps, bs=16):
    """Teacher-forced integer log p of each response token (C's trainer side)."""
    m = eng.m
    out = [None] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]) + len(resps[i]))
    for b0 in range(0, len(order), bs):
        idx = order[b0:b0 + bs]
        ids, pos, L = _pad([prompts[i] + resps[i] for i in idx], m.dev)
        h = m.forward(ids, pos, m.new_cache(len(idx), L))
        rows, toks, spans = [], [], []
        for j, i in enumerate(idx):
            P, n = len(prompts[i]), len(resps[i])
            rows.append(h[j, P - 1:P - 1 + n])
            toks += resps[i]
            spans.append(n)
        hs, ts = torch.cat(rows), torch.tensor(toks, device=m.dev)
        lp = torch.cat([int_logp(m.logits(hs[k:k + 1024]), ts[k:k + 1024], m.ln)
                        for k in range(0, hs.shape[0], 1024)]).split(spans)
        for j, i in enumerate(idx):
            out[i] = lp[j].tolist()
    eng.check()
    return out
