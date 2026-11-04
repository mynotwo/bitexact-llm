"""Tensor-parallel sharding of the BitExact int8 engine.

Question: if the integer engine is sharded across ranks, does the output stay bit-identical to the
unsharded engine? `experiments/portability/RESULTS.md` lists exactly this as an untested inference:

  "An integer engine sharded by tensor parallelism would stay invariant by construction. The
   int32/int64 partial sums are exact, so any summation order gives the same integer. BitExact has no
   sharded implementation, so this is untested."

Sharding plan (Megatron-style, chosen so every split lands on an integer boundary):

  wqkv   N-split by head      rank r owns query heads [r*h, (r+1)*h) and KV heads [r*hkv, (r+1)*hkv)
  wo     K-split by head      input columns = that rank's own attention output
  wgu    N-split by FF        gate row j and up row j go to the same rank, so silu_mul is rank-local
  wd     K-split by FF        all-reduce of the partial int32 accumulator over its K range
  lm     N-split by vocabulary; rows are all-gathered before hashing

The one non-obvious point: `quant_rows` picks its per-row shift from `amax` over the *whole* row. A
K-split therefore has to quantize with the global row maximum, or the two ranks would use different
scales and the integers would differ. Each pre-segment is replicated locally; for the two K-split inputs
(the attention output and the SiLU product) the global amax is a single all-reduce of a small vector,
otherwise the full row is available locally anyway.

Every arithmetic step is the unsharded step: the only new operation is summing exact integer partials.
"""

import torch
import torch.distributed as dist

from bitexact.int8.consts import FR

from bitexact.int8 import segs as S
from bitexact.int8.model import IntLlama
from bitexact.int8.ops import norm24, qmax, quant_rows, rms, silu
from bitexact.int8.segs import FQ, FV, NEG, post, softmax
from bitexact.intops import bitlen, rdiv, rsh, stack_limbs, join_limbs
from bitexact.llama import Cache, D, FF, HD, HKV, H, V
from bitexact.bmm import bmm


def _red(x):
    """Exact integer all-reduce: SUM over the process group, in int64."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    y = x.to(torch.int64).contiguous()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


def _red_max(x):
    """Global row maxima across ranks (int64, exact)."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    y = x.to(torch.int64).contiguous()
    dist.all_reduce(y, op=dist.ReduceOp.MAX)
    return y


def quant_rows_shared(z, amax, nl):
    """`quant_rows` with the row amax supplied by the caller (so a K-split can share one scale)."""
    amax = amax.clamp(min=1)
    sh = bitlen(amax) - 30
    as_ = rsh(amax, sh)
    zs = rsh(z, sh[:, None])
    return rdiv(zs * qmax(nl), as_[:, None]), as_, sh


def pre_plain_shared(x, amax, nl):
    """`pre_plain` with a caller-supplied (global) row amax: same mant/e as the unsharded path."""
    q, as_, sh = quant_rows_shared(x, amax, nl)
    mant, e = norm24(as_, sh - FR)
    return q, mant, e


def shard(c, rank, world):
    """Constants dict -> this rank's shard (weights sliced, scales replicated where needed)."""
    assert H % world == 0 and HKV % world == 0 and FF % world == 0 and V % world == 0, world
    h, hkv, ff, v = H // world, HKV // world, FF // world, V // world
    q0, q1 = rank * h * HD, (rank + 1) * h * HD
    k0 = H * HD + rank * hkv * HD
    k1 = H * HD + (rank + 1) * hkv * HD
    w0 = H * HD + HKV * HD + rank * hkv * HD
    w1 = H * HD + HKV * HD + (rank + 1) * hkv * HD
    out = {"n_layers": c["n_layers"], "cfg": c["cfg"], "sha256": c.get("sha256"), "world": world, "rank": rank}
    for key in ("emb", "exp", "silu", "cos", "sin", "ln", "gf"):
        out[key] = c[key]
    out["emb"] = c["emb"]
    for i in range(c["n_layers"]):
        # qkv: three disjoint row ranges, kept in one weight so the layer does one GEMM per part
        out[f"{i}.wq"] = c[f"{i}.wqkv"][q0:q1]
        out[f"{i}.wq.m"] = c[f"{i}.wqkv.m"][q0:q1]
        out[f"{i}.wk"] = c[f"{i}.wqkv"][k0:k1]
        out[f"{i}.wk.m"] = c[f"{i}.wqkv.m"][k0:k1]
        out[f"{i}.wv"] = c[f"{i}.wqkv"][w0:w1]
        out[f"{i}.wv.m"] = c[f"{i}.wqkv.m"][w0:w1]
        out[f"{i}.wqkv.fm"] = c[f"{i}.wqkv.fm"]
        # wo: K-split on this rank's own attention output columns
        out[f"{i}.wo"] = c[f"{i}.wo"][:, q0:q1]
        out[f"{i}.wo.m"] = c[f"{i}.wo.m"]      # full output-channel scales (output is not split)
        out[f"{i}.wo.fm"] = c[f"{i}.wo.fm"]
        # wgu: gate rows and up rows of the same FF channels
        out[f"{i}.wg"] = c[f"{i}.wgu"][rank * ff:(rank + 1) * ff]
        out[f"{i}.wg.m"] = c[f"{i}.wgu.m"][rank * ff:(rank + 1) * ff]
        out[f"{i}.wu"] = c[f"{i}.wgu"][FF + rank * ff:FF + (rank + 1) * ff]
        out[f"{i}.wu.m"] = c[f"{i}.wgu.m"][FF + rank * ff:FF + (rank + 1) * ff]
        out[f"{i}.wgu.fm"] = c[f"{i}.wgu.fm"]
        out[f"{i}.wo.m"] = c[f"{i}.wo.m"]
        # wd: K-split on this rank's own FF channels
        out[f"{i}.wd"] = c[f"{i}.wd"][:, rank * ff:(rank + 1) * ff]
        out[f"{i}.wd.m"] = c[f"{i}.wd.m"]
        out[f"{i}.wd.fm"] = c[f"{i}.wd.fm"]
        out[f"{i}.ga"] = c[f"{i}.ga"]
        out[f"{i}.gm"] = c[f"{i}.gm"]
    out["lm"] = c["lm"][rank * v:(rank + 1) * v]
    out["lm.m"] = c["lm.m"][rank * v:(rank + 1) * v]
    out["lm.fm"] = c["lm.fm"]
    return out

class ShardCache:
    """Integer K / V for this rank's KV heads only: int32 [B, hkv, L, HD] per layer."""

    def __init__(self, n_layers, B, L, hkv, device):
        z = lambda: torch.zeros(B, hkv, L, HD, dtype=torch.int32, device=device)
        self.k = [z() for _ in range(n_layers)]
        self.v = [z() for _ in range(n_layers)]

    def write(self, layer, k, v, pos):
        b = torch.arange(k.shape[0], device=k.device)[:, None].expand_as(pos)
        self.k[layer].permute(0, 2, 1, 3)[b, pos] = k.to(torch.int32)
        self.v[layer].permute(0, 2, 1, 3)[b, pos] = v.to(torch.int32)


def _rope(x, c, s):
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    from bitexact.int8.consts import FT
    sh = FT + FR - FQ
    return torch.cat([rsh(x1 * c - x2 * s, sh), rsh(x2 * c + x1 * s, sh)], -1)


def shard_qkv(y, c, s, h, hkv):
    """`segs.qkv` restricted to this rank's heads; y is [B, T, (h + 2 hkv) * HD] at FR."""
    B, T = y.shape[0], y.shape[1]
    q = _rope(y[..., :h * HD].reshape(B, T, h, HD), c, s)
    e = h * HD
    k = _rope(y[..., e:e + hkv * HD].reshape(B, T, hkv, HD), c, s)
    v = rsh(y[..., e + hkv * HD:].reshape(B, T, hkv, HD), FR - FV)
    g = H // HKV
    qg = q.view(B, T, hkv, g, HD).permute(0, 2, 3, 1, 4).reshape(B * hkv, g * T, HD)
    # bounds are checked globally in the caller (the unsharded check is over the full q/k/v)
    return qg, k, v


def attn_out_shard(num, den, T, amax, nl):
    """`segs.attn_out` for this rank's heads, quantizing with the global row amax."""
    B = num.shape[0]
    g = H // HKV
    hkv = num.shape[1]
    o = rdiv(num << (FR - FV), den)
    o = o.view(B, hkv, g, T, HD).permute(0, 3, 1, 2, 4).reshape(B * T, hkv * g * HD)
    return pre_plain_shared(o, amax, nl)


class TPIntLlama:
    """The int8 engine, sharded over `world` ranks. TP=1 reproduces `IntLlama` exactly."""

    FR = FR

    def __init__(self, c, device, rank=0, world=1, nl=2):
        self.dev = torch.device(device)
        self.rank, self.world, self.nl = rank, world, nl
        self.h, self.hkv, self.ff = H // world, HKV // world, FF // world
        self.L = c["n_layers"]
        t = lambda k: c[k].to(device)
        self.emb = t("emb")
        self.layers = []
        for i in range(self.L):
            ly = {"ga": t(f"{i}.ga"), "gm": t(f"{i}.gm")}
            for n in ("wq", "wk", "wv", "wo", "wg", "wu", "wd"):
                cm = {"wq": "wqkv", "wk": "wqkv", "wv": "wqkv", "wo": "wo", "wg": "wgu", "wu": "wgu", "wd": "wd"}[n]
                fm = c[f"{i}.{cm}.fm"]
                ly[n] = (t(f"{i}.{n}"), t(f"{i}.{n}.m"), self._fmsh(fm))
            self.layers.append(ly)
        self.gf = t("gf")
        self.lm = (t("lm"), t("lm.m"), self._fmsh(c["lm.fm"]))
        self.exp, self.silu_t, self.cos, self.sin, self.ln = t("exp"), t("silu"), t("cos"), t("sin"), t("ln")
        self.err = False
        self.sha = c.get("sha256")

    def _fmsh(self, fm):
        return torch.tensor(fm + 8 * (self.nl - 1) - 24, device=self.dev)

    def _flag(self, cond):
        if bool(cond):
            self.err = True

    def _gemm(self, a, w):
        return torch._int_mm(a, w.t())

    def _cjoin(self, acc, M):
        return join_limbs(acc, M, self.nl)

    def _lin(self, pre, lin):
        """pre = (q limbs, mant, e) for a K-split input -> (y at FR, acc joined) locally."""
        w, m, fmsh = lin
        c = self._gemm(pre[0], w)
        M = pre[1].shape[0]
        return self._cjoin(c, M), m, fmsh, pre[1], pre[2]

    def norm_pre(self, x, gamma):
        """RMSNorm + per-row quantization: identical on every rank (depends on x, gamma only)."""
        r, shx = rms(x)
        q, as_, sh = quant_rows(x * gamma, self.nl)
        from bitexact.int8.consts import FG
        mant, e = norm24(rdiv(as_ << 30, r), sh - FG - shx - 30)
        return stack_limbs(q, self.nl), mant, e

    def attention(self, i, x, pos, cache, S):
        B, T, _ = x.shape
        ly = self.layers[i]
        pre = self.norm_pre(x.reshape(B * T, D), ly["ga"])
        mant, e = pre[1], pre[2]
        # three row-slices share one qkv GEMM in the unsharded engine; here one GEMM per part
        parts = []
        for n in ("wq", "wk", "wv"):
            w, mm, f = ly[n]
            c = self._gemm(pre[0], w)
            parts.append(post(self._cjoin(c, B * T), mm, f, mant, e)[0])
        y = torch.cat(parts, -1).view(B, T, (self.h + 2 * self.hkv) * HD)
        qg, k, v = shard_qkv(y, self.cos[pos][:, :, None], self.sin[pos][:, :, None], self.h, self.hkv)
        cache.write(i, k, v, pos)
        K, Vv = cache.k[i][:, :, :S], cache.v[i][:, :, :S]
        g = H // HKV
        raw = bmm(qg, K.transpose(-1, -2).reshape(B * self.hkv, HD, S)).view(B, self.hkv, g * T, S)
        qpos = pos[:, None, None, :].expand(B, 1, g, T).reshape(B, 1, g * T, 1)
        p, den = softmax(raw, qpos, self.exp)
        num = bmm(p.view(B * self.hkv, g * T, S), Vv.reshape(B * self.hkv, S, HD)).view(B, self.hkv, g * T, HD)
        # global row amax of the attention output (the unsharded quant_rows uses the full row)
        o_local = rdiv(num << (FR - FV), den)
        o_local = o_local.view(B, self.hkv, g, T, HD).permute(0, 3, 1, 2, 4).reshape(B * T, self.hkv * g * HD)
        amax = _red_max(o_local.abs().amax(-1))
        q_o, mant_o, e_o = pre_plain_shared(o_local, amax, self.nl)
        w, m, fmsh = ly["wo"]
        c = self._gemm(stack_limbs(q_o, self.nl), w)
        acc = self._cjoin(c, B * T)
        acc = _red(acc)
        out, flag = post(acc, m, fmsh, mant_o, e_o)
        self._flag(flag)
        return x + out.view(B, T, D)

    def mlp(self, i, x):
        B, T, _ = x.shape
        ly = self.layers[i]
        pre = self.norm_pre(x.reshape(B * T, D), ly["gm"])
        mant, e = pre[1], pre[2]
        gs, us = [], []
        for n in ("wg", "wu"):
            w, m, fmsh = ly[n]
            c = self._gemm(pre[0], w)
            gs.append(post(self._cjoin(c, B * T), m, fmsh, mant, e)[0])
        gu = torch.cat(gs, -1)
        sg = silu(gu[:, :self.ff], self.silu_t)
        prod = rsh(sg * gu[:, self.ff:], FR)
        amax = _red_max(prod.abs().amax(-1))
        q, mant2, e2 = pre_plain_shared(prod, amax, self.nl)
        w, m, fmsh = ly["wd"]
        c = self._gemm(stack_limbs(q, self.nl), w)
        acc = _red(self._cjoin(c, B * T))
        out, flag = post(acc, m, fmsh, mant2, e2)
        self._flag(flag)
        return x + out.view(B, T, D)

    def forward(self, ids, pos, cache, S=None):
        x = self.emb[ids].long()
        S = int(pos.max()) + 1 if S is None else S
        for i in range(self.L):
            x = self.attention(i, x, pos, cache, S)
            x = self.mlp(i, x)
        self._flag(bool(x.abs().amax() >= 1 << 42))
        return x

    def logits(self, x):
        """[M, D] -> this rank's int64 logits rows [M, V / world] at FR."""
        M = x.reshape(-1, D).shape[0]
        pre = self.norm_pre(x.reshape(-1, D), self.gf)
        w, m, fmsh = self.lm
        c = self._gemm(pre[0], w)
        return post(self._cjoin(c, M), m, fmsh, pre[1], pre[2])[0]

    def new_cache(self, B, L):
        return ShardCache(self.L, B, L, self.hkv, self.dev)

    def check(self):
        if dist.is_initialized() and self.world > 1:
            flag = torch.tensor([int(self.err)], device=self.dev)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            self.err = bool(flag.item())
        if self.err:
            raise OverflowError("integer bound violated (tp_shard/tpmodel.py)")


def gather_logits(lg, world, device):
    """[M, V/world] per rank -> [M, V] in vocabulary order (rank r holds columns [r*V/world, ...))."""
    if world == 1:
        return lg
    lg = lg.contiguous()
    parts = [torch.empty_like(lg) for _ in range(world)]
    dist.all_gather(parts, lg)
    return torch.cat(parts, dim=1)
