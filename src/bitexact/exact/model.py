"""Exact-bf16 Llama (3.2-1B, 3.1-8B; geometry from the constants' config): original bf16 weights, integer
activations, exact products.

Fixed point (fractional bits): residual stream, linear outputs, attention output and logits at FR = 24;
normalized activations at FN = 40; k at FK = 17, v at FV = 19 (KV cache int32, |.| <= 8355711, checked);
attention weights at FP = 22. Linear inputs are quantized per row with a power-of-two scale (|q| <= 2^22),
so a linear output is one rounding of the exact value (q @ W_bf16) * 2^(sh + FR). q rows of attention are
quantized per (token, head) the same way. Every rule is row-local, so outputs do not depend on the batch,
on padding, or on whether keys came from the cache. Bounds are checked on the device (self.err).
"""

import torch

from bitexact.bmm import bmm
from bitexact.exact.consts import FR
from bitexact.exact.segs import Segs
from bitexact.exact.wmm import HAVE_TRITON, residual, wmm_cpu, wmm_cublas, wmm_triton
from bitexact.intops import split_limbs
from bitexact.llama import Cache, dims


def _pv_cpu(p, V):
    """Exact p @ V on CPU: V split into 3 limbs so partial sums stay < 2^53."""
    out = None
    for i, lo in enumerate(split_limbs(V, 3)[0]):
        t = (p.double() @ lo.double()).long() << (8 * i)
        out = t if out is None else out + t
    return out


class Lin:
    def __init__(self, c, name, device, gemm, ell, R=0, cpu_lazy=False):
        t = lambda k: c[f"{name}.{k}"].to(device)
        self.limbs = t("limbs")
        self.N = self.limbs.shape[1]
        self.g, self.d, self.rr, self.rc, self.rv = t("g"), t("d"), t("rr"), t("rc"), t("rv")
        self.gemm = gemm if self.limbs.is_cuda else "cpu"
        self.ell = ell and self.gemm != "cpu"
        if self.ell:                                          # padded per-row residual lists
            rr = c[f"{name}.rr"]
            cnt = torch.bincount(rr, minlength=self.N)
            R = max(R, int(cnt.max()) if rr.numel() else 0)     # common R per kind: one compiled shape
            start = torch.cumsum(cnt, 0) - cnt
            j = torch.arange(rr.numel()) - start[rr]
            ec = torch.zeros(self.N, R, dtype=torch.long)
            ev = torch.zeros(self.N, R, dtype=torch.long)
            ec[rr, j], ev[rr, j] = c[f"{name}.rc"], c[f"{name}.rv"]
            self.ec, self.ev = ec.to(device), ev.to(device)
        if self.gemm == "cublas":
            self.wcat = self.limbs.reshape(3 * self.N, -1)
        self.lazy = cpu_lazy and self.gemm == "cpu"
        if self.gemm == "cpu" and not self.lazy:
            self.w64 = self._w64()

    def _w64(self):
        L = self.limbs.long()
        return (L[0] + (L[1] << 8) + (L[2] << 16)).double().t().contiguous()

    def __call__(self, q):
        """(C, rs, d, g): exact dense products and the residual for segs._lin."""
        if self.gemm == "triton":
            C = wmm_triton(q, self.limbs)
        elif self.gemm == "cublas":
            C = wmm_cublas(q, self.wcat)
        else:
            C = wmm_cpu(q, self._w64() if self.lazy else self.w64)
        rs = (q, self.ec, self.ev) if self.ell else (residual(q, self.rr, self.rc, self.rv, self.N),)
        return C, rs, self.d, self.g


class ExactLlama:
    """gemm: "triton" (default; falls back to "cublas" without Triton) or "cublas" on CUDA; CPU tensors
    always use the float64 path. compiled: torch.compile the element-wise stages (same integers).
    cpu_lazy: on CPU, build each layer's float64 weight copy per call instead of keeping all of them
    (8 bytes per weight; needed for Llama-3.1-8B on a 48 GB machine; same integers, slower)."""

    FR = FR

    def __init__(self, c, device, n_layers=None, gemm="triton", compiled=False, cpu_lazy=False):
        self.dev = torch.device(device)
        if gemm == "triton" and not HAVE_TRITON:
            gemm = "cublas"
        self.L = c["n_layers"] if n_layers is None else n_layers
        t = lambda k: c[k].to(device)
        self.emb = t("emb")
        self.layers = []
        rmax = lambda n: max(int(torch.bincount(c[f"{i}.{n}.rr"]).max()) if c[f"{i}.{n}.rr"].numel() else 0
                             for i in range(self.L))
        R = {n: rmax(n) for n in ("wqkv", "wo", "wgu", "wd")}
        for i in range(self.L):
            ly = {n: Lin(c, f"{i}.{n}", device, gemm, compiled, R[n], cpu_lazy) for n in ("wqkv", "wo", "wgu", "wd")}
            for n in ("ga", "gm"):
                ly[n] = (t(f"{i}.{n}.m"), t(f"{i}.{n}.e"))
            self.layers.append(ly)
        self.gf = (t("gf.m"), t("gf.e"))
        self.lm = Lin(c, "lm", device, gemm, compiled, 0, cpu_lazy)
        self.cos, self.sin, self.exp2, self.ln = t("cos"), t("sin"), t("exp2"), t("ln")
        self.err = torch.zeros((), dtype=torch.bool, device=device)
        self.sha = c.get("sha256")
        self.H, self.HKV, self.HD, self.D, _ = dims(c.get("cfg", {}))
        assert self.emb.shape[1] == self.D, "constants config does not match the embedding width"
        self.sg = Segs(compiled, (self.H, self.HKV, self.HD))

    def _layer(self, i, x, q, sh, pos, cache, S, err):
        """One decoder layer from x and its normalized input (q, sh); returns the next (x, q, sh, err)."""
        B, T, _ = x.shape
        ly = self.layers[i]
        qg, shg, k, v, err = self.sg.qkv(*ly["wqkv"](q), sh, pos, self.cos, self.sin, B, T, err)
        cache.write(i, k, v, pos)
        Kc, Vc = cache.k[i][:, :, :S], cache.v[i][:, :, :S]
        Z, hd = B * self.HKV, self.HD
        raw = bmm(qg, Kc.transpose(-1, -2).reshape(Z, hd, S))         # exact q.k
        g = qg.shape[1] // T
        qpos = pos[:, None, None, :].expand(B, self.HKV, g, T).reshape(Z, g * T, 1)
        p, den = self.sg.softmax(raw, qpos, shg, self.exp2)
        Vr = Vc.reshape(Z, S, hd)
        num = bmm(p, Vr) if x.is_cuda else _pv_cpu(p, Vr)
        qo, sho = self.sg.attn_out(num, den, T)
        x, q, sh, err = self.sg.add_norm(*ly["wo"](qo), sho, x, *ly["gm"], err)
        q, sh, err = self.sg.gu_silu(*ly["wgu"](q), sh, self.exp2, err)
        nxt = self.layers[i + 1]["ga"] if i + 1 < self.L else self.gf
        return self.sg.add_norm(*ly["wd"](q), sh, x, *nxt, err)

    def forward_q(self, ids, pos, cache, S=None):
        """ids, pos int64 [B, T]; writes K/V at pos. Returns the final residual stream [B, T, D] at FR and
        its final-norm quantization (q, sh), the input of the logits. S: keys attended (default
        max(pos) + 1; a fixed S avoids the host sync, e.g. for CUDA graphs)."""
        x = self.emb[ids].long()
        S = int(pos.max()) + 1 if S is None else S
        q, sh = self.sg.pre_norm(x.view(-1, self.D), *self.layers[0]["ga"])
        err = self.err
        for i in range(self.L):
            x, q, sh, err = self._layer(i, x, q, sh, pos, cache, S, err)
        self.err = err
        return x, q, sh

    def forward(self, ids, pos, cache, S=None):
        return self.forward_q(ids, pos, cache, S)[0]

    def logits_q(self, q, sh):
        """Final-norm rows (q, sh) -> logits int64 [M, V] at FR."""
        y, flag = self.sg.lin_out(*self.lm(q), sh)
        self.err = self.err | flag
        return y

    def logits(self, x):
        """Final residual rows [M, D] -> logits int64 [M, V] at FR."""
        return self.logits_q(*self.sg.pre_norm(x, *self.gf))

    def new_cache(self, B, L):
        return Cache(self.L, B, L, self.dev, self.HKV, self.HD)

    def check(self):
        if bool(self.err):
            raise OverflowError("integer bound violated (exact/model.py)")
