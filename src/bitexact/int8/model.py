"""Int8 engine: Llama-3.2-1B with int8 weights, integer activations and an integer KV cache.

Each row of the batch is computed with row-local integer rules only (per-row quantization, per-row RMS,
per-query softmax), and every reduction is an exact integer sum. So the logits of a token do not depend
on which other rows share the batch, on padding, or on whether earlier tokens came from a cached prefix
or from the same forward call.

Fixed-point layout (fractional bits): residual, q/k/v projections, attention output, MLP values and
logits at FR = 16; q and k at FQ = 14 after RoPE, v at FV = 14; attention weights at FP = 20.
Matrix products against weights: quant_rows (int8 limbs) + int8 GEMM (ops.imm). Attention products
q.k and p.v: exact limb GEMMs (tri_attn.bmm; float64 on CPU). Bounds are checked on the device; a
violation sets self.err, which callers test once at the end (model.check).
"""

import torch

from bitexact.bmm import bmm
from bitexact.int8.consts import FR
from bitexact.int8.ops import EXACT, imm, post_bound
from bitexact.int8.segs import Segs
from bitexact.llama import HD, HKV, Cache, D, H


class IntLlama:
    """nl: activation limbs per linear input (2: |q| <= 32512). compiled: torch.compile the element-wise
    segments (same integers)."""

    FR = FR

    def __init__(self, c, device, nl=2, n_layers=None, compiled=False):
        self.dev = torch.device(device)
        self.nl = nl
        self.L = c["n_layers"] if n_layers is None else n_layers
        t = lambda k: c[k].to(device)
        self.emb = c["emb"].to(device)
        self.layers = []
        for i in range(self.L):
            ly = {"ga": t(f"{i}.ga"), "gm": t(f"{i}.gm")}
            for n in ("wqkv", "wo", "wgu", "wd"):
                ly[n] = (t(f"{i}.{n}"), t(f"{i}.{n}.m"), self._fmsh(c[f"{i}.{n}.fm"]))
            self.layers.append(ly)
        self.gf = t("gf")
        self.lm = (t("lm"), t("lm.m"), self._fmsh(c["lm.fm"]))
        self.exp, self.silu_t, self.cos, self.sin, self.ln = t("exp"), t("silu"), t("cos"), t("sin"), t("ln")
        self.err = torch.zeros((), dtype=torch.bool, device=device)
        self.sha = c.get("sha256")
        # |t| < 2^38 after every GEMM for any input (post_bound): then segs.post skips its run-time check.
        lins = [ly[n] for ly in self.layers for n in ("wqkv", "wo", "wgu", "wd")] + [self.lm]
        self.post_check = any(post_bound(w, m, int(f), nl) >= 1 << 38 for w, m, f in lins)
        self.sg = Segs(nl, self.dev.type == "cuda", compiled, self.post_check)
        self._w64 = {}                                      # CPU: float64 copies of the int8 weights

    def _fmsh(self, fm):
        return torch.tensor(fm + 8 * (self.nl - 1) - 24, device=self.dev)

    # ---- linear layers ----
    def _flag(self, cond):
        if cond is not None:
            self.err |= cond

    def _gemm(self, a, w):
        if a.is_cuda:
            return torch._int_mm(a, w.t())                  # int8 limbs, int32 accumulation: exact
        if w.data_ptr() not in self._w64:
            self._w64[w.data_ptr()] = w.double().t().contiguous()
        return imm(a, self._w64[w.data_ptr()])              # float64, exact

    def _linear(self, pre, lin, x=None):
        """pre = (limbs or q, mant, e); returns the layer output at FR (plus x when given)."""
        w, m, fmsh = lin
        c = self._gemm(pre[0], w)
        y, flag = self.sg.post(c, m, fmsh, pre[1], pre[2]) if x is None else \
            self.sg.post_add(c, m, fmsh, pre[1], pre[2], x)
        self._flag(flag)
        return y

    def norm_linear(self, x, gamma, lin):
        """RMSNorm(x) * gamma @ W^T at FR. x int64 [M, D] at FR."""
        return self._linear(self.sg.pre_norm(x, gamma), lin)

    # ---- layers ----
    def attention(self, i, x, pos, cache, S):
        B, T, _ = x.shape
        ly = self.layers[i]
        y = self.norm_linear(x.reshape(B * T, D), ly["ga"], ly["wqkv"]).view(B, T, -1)
        qg, k, v, flag = self.sg.qkv(y, self.cos[pos][:, :, None], self.sin[pos][:, :, None])
        self._flag(flag)
        cache.write(i, k, v, pos)
        K, V = cache.k[i][:, :, :S], cache.v[i][:, :, :S]
        g = H // HKV
        raw = bmm(qg, K.transpose(-1, -2).reshape(B * HKV, HD, S)).view(B, HKV, g * T, S)   # q.k at 2 FQ
        qpos = pos[:, None, None, :].expand(B, 1, g, T).reshape(B, 1, g * T, 1)
        p, den = self.sg.softmax(raw, qpos, self.exp)
        if not x.is_cuda:                                                      # float64 path: sums < 2^53
            self._flag(V.abs().amax().long() * den.amax() >= EXACT)
        num = bmm(p.view(B * HKV, g * T, S), V.reshape(B * HKV, S, HD)).view(B, HKV, g * T, HD)
        return self._linear(self.sg.attn_out(num, den, T), ly["wo"], x)

    def mlp(self, i, x):
        B, T, _ = x.shape
        ly = self.layers[i]
        gu = self.norm_linear(x.reshape(B * T, D), ly["gm"], ly["wgu"])
        a8, mant, e, flag = self.sg.silu_mul(gu, self.silu_t)
        self._flag(flag)
        return self._linear((a8, mant, e), ly["wd"], x)

    # ---- full model ----
    def forward(self, ids, pos, cache, S=None):
        """ids, pos: int64 [B, T]. Writes K/V at pos into cache rows 0..B-1. Returns the final residual
        stream [B, T, D] at FR (before the final norm). S: keys attended (default max(pos) + 1; a fixed S
        avoids the host sync, e.g. for CUDA graphs; keys beyond a query's position are masked anyway)."""
        x = self.emb[ids].long()
        S = int(pos.max()) + 1 if S is None else S
        for i in range(self.L):
            x = self.attention(i, x, pos, cache, S)
            x = self.mlp(i, x)
        self._flag(x.abs().amax() >= 1 << 42)
        return x

    def logits(self, x):
        """Final residual rows [M, D] -> logits int64 [M, V] at FR."""
        return self.norm_linear(x, self.gf, self.lm)

    def new_cache(self, B, L):
        return Cache(self.L, B, L, self.dev)

    def check(self):
        if bool(self.err):
            raise OverflowError("integer bound violated (int8/model.py)")
