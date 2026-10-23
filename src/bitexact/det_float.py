"""Batch-invariant bf16 Llama-3.x forward (float, not integer): bit-exact replay on one software stack.

Every op that reduces over a dimension is a Triton kernel with one fixed tile
config, so the reduction order for a given output element never depends on how
many rows are in the batch, which rows they are, or whether attention runs over
a KV cache or a full sequence:

- mm:      C[M,N] = A[M,K] @ W[N,K]^T, no split-K, K loop always in the same order.
- rmsnorm: one program per row, fp32.
- attn:    causal flash-style attention, each query loops over KV blocks of
           fixed size starting at position 0; positions > own pos are masked
           with -inf, which contributes exactly 0.
Elementwise ops (RoPE, SiLU, residual) are left to torch: they have no reduction.

Sampling noise is a counter-based hash of (seed, sid, step, vocab index), so it
depends only on the sample and its generation step.

Unlike the integer engines this relies on floating-point kernels compiled by Triton: outputs are fixed for
one kernel configuration, but a different GPU architecture or Triton version may round differently.
cublas=True switches to an ordinary (not batch-invariant) cuBLAS / SDPA path, as a control.
"""
import json
import os

import torch
import triton
import triton.language as tl
from safetensors import safe_open

MM_BM, MM_BN, MM_BK, MM_WARPS, MM_STAGES = 64, 64, 64, 4, 4
AT_BM, AT_BN = 64, 64


@triton.jit(do_not_specialize=["M"])
def _mm_kernel(a_ptr, w_ptr, c_ptr, M, N, K,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_p = a_ptr + rm[:, None].to(tl.int64) * K + rk[None, :]
    w_p = w_ptr + rn[None, :].to(tl.int64) * K + rk[:, None]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(a_p, mask=rm[:, None] < M, other=0.0)
        w = tl.load(w_p, mask=rn[None, :] < N, other=0.0)
        acc = tl.dot(a, w, acc)
        a_p += BK
        w_p += BK
    c_p = c_ptr + rm[:, None].to(tl.int64) * N + rn[None, :]
    tl.store(c_p, acc.to(c_ptr.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


def mm(a, w, out_dtype=torch.bfloat16):
    """a [M,K] bf16 contiguous, w [N,K] bf16 contiguous -> [M,N]."""
    M, K = a.shape
    N = w.shape[0]
    assert K % MM_BK == 0 and a.is_contiguous() and w.is_contiguous()
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    grid = (triton.cdiv(N, MM_BN), triton.cdiv(M, MM_BM))
    _mm_kernel[grid](a, w, c, M, N, K, BM=MM_BM, BN=MM_BN, BK=MM_BK,
                     num_warps=MM_WARPS, num_stages=MM_STAGES)
    return c


@triton.jit
def _rms_kernel(x_ptr, w_ptr, y_ptr, eps, H: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, H)
    x = tl.load(x_ptr + row * H + cols).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)
    y = (x * rstd).to(tl.bfloat16)
    w = tl.load(w_ptr + cols)
    tl.store(y_ptr + row * H + cols, (w.to(tl.float32) * y.to(tl.float32)).to(tl.bfloat16))


def rmsnorm(x, w, eps):
    M, H = x.shape
    y = torch.empty_like(x)
    _rms_kernel[(M,)](x, w, y, eps, H=H, num_warps=8)
    return y


@triton.jit
def _attn_kernel(q_ptr, k_ptr, v_ptr, o_ptr, pos_ptr, Sq, Lmax, qk_scale,
                 HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // HQ
    h = bh % HQ
    hk = h // (HQ // HKV)
    rm = pid_m * BM + tl.arange(0, BM)
    rd = tl.arange(0, D)
    mmask = rm < Sq
    pos = tl.load(pos_ptr + b * Sq + rm, mask=mmask, other=0)
    q = tl.load(q_ptr + ((b * HQ + h) * Sq + rm[:, None]).to(tl.int64) * D + rd[None, :],
                mask=mmask[:, None], other=0.0)
    kv_off = ((b * HKV + hk) * Lmax).to(tl.int64) * D
    m_i = tl.full((BM,), float("-inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    hi = tl.max(pos, axis=0) + 1
    for start in range(0, hi, BN):
        rn = start + tl.arange(0, BN)
        nmask = rn < Lmax
        k = tl.load(k_ptr + kv_off + rn[None, :] * D + rd[:, None], mask=nmask[None, :], other=0.0)
        s = tl.dot(q, k) * qk_scale
        s = tl.where(rn[None, :] <= pos[:, None], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + kv_off + rn[:, None] * D + rd[None, :], mask=nmask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    o = acc / l_i[:, None]
    # output layout [B, Sq, HQ, D] so it feeds o_proj without a transpose
    o_p = o_ptr + ((b * Sq + rm[:, None]).to(tl.int64) * HQ + h) * D + rd[None, :]
    tl.store(o_p, o.to(tl.bfloat16), mask=mmask[:, None])


def attention(q, k_cache, v_cache, pos):
    """q [B,HQ,Sq,D] contiguous; caches [B,HKV,Lmax,D]; pos [B,Sq] int32 -> [B*Sq, HQ*D]."""
    B, HQ, Sq, D = q.shape
    HKV, Lmax = k_cache.shape[1], k_cache.shape[2]
    o = torch.empty((B, Sq, HQ, D), device=q.device, dtype=torch.bfloat16)
    qk_scale = (D ** -0.5) * 1.4426950408889634
    grid = (triton.cdiv(Sq, AT_BM), B * HQ)
    _attn_kernel[grid](q, k_cache, v_cache, o, pos, Sq, Lmax, qk_scale,
                       HQ=HQ, HKV=HKV, D=D, BM=AT_BM, BN=AT_BN, num_warps=4, num_stages=2)
    return o.view(B * Sq, HQ * D)


def llama3_inv_freq(cfg):
    """Llama-3.1 rope scaling, computed on CPU in float64 then cast (deterministic)."""
    import math
    D = cfg["hidden_size"] // cfg["num_attention_heads"]
    base = cfg["rope_theta"]
    inv = [1.0 / base ** (i / D) for i in range(0, D, 2)]
    rs = cfg.get("rope_scaling")
    if rs:
        f, lo, hi, old = rs["factor"], rs["low_freq_factor"], rs["high_freq_factor"], \
            rs["original_max_position_embeddings"]
        out = []
        for x in inv:
            wl = 2 * math.pi / x
            if wl < old / hi:
                out.append(x)
            elif wl > old / lo:
                out.append(x / f)
            else:
                sm = (old / wl - lo) / (hi - lo)
                out.append((1 - sm) * x / f + sm * x)
        inv = out
    return torch.tensor(inv, dtype=torch.float32)


class DetLlama:
    def __init__(self, path, device, cublas=False):
        self.cfg = json.load(open(os.path.join(path, "config.json")))
        c = self.cfg
        self.dev = device
        self.H, self.HQ, self.HKV = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"]
        self.D = self.H // self.HQ
        self.eps = c["rms_norm_eps"]
        self.n_layers = c["num_hidden_layers"]
        self.cublas = cublas
        w = {}
        index = os.path.join(path, "model.safetensors.index.json")
        files = sorted(set(json.load(open(index))["weight_map"].values())) if os.path.exists(index) \
            else ["model.safetensors"]
        for fn in files:
            with safe_open(os.path.join(path, fn), framework="pt", device=str(device)) as f:
                for k in f.keys():
                    w[k] = f.get_tensor(k).to(torch.bfloat16)
        self.embed = w["model.embed_tokens.weight"]
        self.lm_head = w.get("lm_head.weight", self.embed)
        self.norm = w["model.norm.weight"]
        self.layers = []
        for i in range(self.n_layers):
            p = f"model.layers.{i}."
            self.layers.append(dict(
                ln1=w[p + "input_layernorm.weight"], ln2=w[p + "post_attention_layernorm.weight"],
                qkv=torch.cat([w[p + "self_attn.q_proj.weight"], w[p + "self_attn.k_proj.weight"],
                               w[p + "self_attn.v_proj.weight"]]).contiguous(),
                o=w[p + "self_attn.o_proj.weight"],
                gu=torch.cat([w[p + "mlp.gate_proj.weight"], w[p + "mlp.up_proj.weight"]]).contiguous(),
                down=w[p + "mlp.down_proj.weight"]))
            for k in ("q_proj", "k_proj", "v_proj"):
                del w[p + "self_attn." + k + ".weight"]
            del w[p + "mlp.gate_proj.weight"], w[p + "mlp.up_proj.weight"]
        self.inv_freq = llama3_inv_freq(c).to(device)
        self.I = self.layers[0]["down"].shape[1]
        torch.cuda.empty_cache()

    def _mm(self, a, w, out_dtype=torch.bfloat16):
        if self.cublas:
            return torch.nn.functional.linear(a, w).to(out_dtype)
        return mm(a, w, out_dtype)

    def _rms(self, x, w):
        if self.cublas:
            xf = x.float()
            xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
            return w * xf.to(torch.bfloat16)
        return rmsnorm(x, w, self.eps)

    def new_cache(self, B, Lmax):
        shape = (self.n_layers, B, self.HKV, Lmax, self.D)
        return (torch.zeros(shape, device=self.dev, dtype=torch.bfloat16),
                torch.zeros(shape, device=self.dev, dtype=torch.bfloat16))

    def _rope(self, x, cos, sin):
        h = x.shape[-1] // 2
        rot = torch.cat([-x[..., h:], x[..., :h]], dim=-1)
        return x * cos + rot * sin

    def forward(self, tokens, pos, cache, out_rows=None):
        """tokens, pos: [B,S] (int64 / int32). Writes K/V at `pos` into cache.

        Returns fp32 logits for the flattened rows `out_rows` (indices into B*S),
        or for all rows when None.
        """
        B, S = tokens.shape
        kc, vc = cache
        HQ, HKV, D = self.HQ, self.HKV, self.D
        x = self.embed[tokens.reshape(-1)]
        freqs = pos.reshape(-1).float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(torch.bfloat16).view(B, S, 1, D).transpose(1, 2)
        sin = emb.sin().to(torch.bfloat16).view(B, S, 1, D).transpose(1, 2)
        bidx = torch.arange(B, device=self.dev)[:, None].expand(B, S)
        pos32 = pos.to(torch.int32).contiguous()
        posl = pos.long()
        for li, L in enumerate(self.layers):
            h = self._rms(x, L["ln1"])
            qkv = self._mm(h, L["qkv"]).view(B, S, HQ + 2 * HKV, D).transpose(1, 2)
            q = self._rope(qkv[:, :HQ], cos, sin).contiguous()
            k = self._rope(qkv[:, HQ:HQ + HKV], cos, sin)
            v = qkv[:, HQ + HKV:]
            # cache[l, b, h, pos[b,s]] = k[b, h, s]
            kc[li][bidx, :, posl] = k.transpose(1, 2)
            vc[li][bidx, :, posl] = v.transpose(1, 2)
            if self.cublas:
                o = self._sdpa(q, kc[li], vc[li], posl)
            else:
                o = attention(q, kc[li], vc[li], pos32)
            x = x + self._mm(o, L["o"])
            h = self._rms(x, L["ln2"])
            gu = self._mm(h, L["gu"])
            a = torch.nn.functional.silu(gu[:, :self.I]) * gu[:, self.I:]
            x = x + self._mm(a.contiguous(), L["down"])
        if out_rows is not None:
            x = x[out_rows]
        x = self._rms(x.contiguous(), self.norm)
        return self._mm(x, self.lm_head, torch.float32)

    def _sdpa(self, q, kc, vc, posl):
        """Non-invariant reference path: torch SDPA over the cache with a position mask."""
        B, HQ, S, D = q.shape
        rep = HQ // self.HKV
        k = kc.repeat_interleave(rep, dim=1)
        v = vc.repeat_interleave(rep, dim=1)
        hi = int(posl.max()) + 1
        mask = torch.arange(hi, device=self.dev)[None, None, :] <= posl[:, :, None]
        o = torch.nn.functional.scaled_dot_product_attention(
            q, k[:, :, :hi], v[:, :, :hi], attn_mask=mask[:, None])
        return o.transpose(1, 2).reshape(B * S, HQ * D)


M32 = 0xFFFFFFFF


def _mul32(x, c):
    lo = x & 0xFFFF
    hi = x >> 16
    return (lo * c + (((hi * c) & 0xFFFF) << 16)) & M32


def _h32(x):
    x = x & M32
    x = x ^ (x >> 16)
    x = _mul32(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = _mul32(x, 0x846CA68B)
    x = x ^ (x >> 16)
    return x


def gumbel(seed, sids, step, V, device):
    """float64 Gumbel noise [len(sids), V], a pure function of (seed, sid, step, v)."""
    s = torch.as_tensor(sids, device=device, dtype=torch.int64)[:, None]
    key = _h32(_h32(_h32(torch.full_like(s, seed)) ^ s) ^ step)
    v = torch.arange(V, device=device, dtype=torch.int64)[None, :]
    h = _h32(key ^ _h32(v + 0x9E3779B9))
    u = (h.double() + 0.5) / 4294967296.0
    return -torch.log(-torch.log(u))


def sample(logits, T, seed, sids, step):
    g = gumbel(seed, sids, step, logits.shape[-1], logits.device)
    return torch.argmax(logits.double() / T + g, dim=-1)


def checksum(logits):
    """Three int64 checksums per row of fp32 logits; equal iff bitwise equal (w.h.p.)."""
    bits = logits.contiguous().view(torch.int32).long()
    idx = torch.arange(1, logits.shape[-1] + 1, device=logits.device, dtype=torch.int64)
    return torch.stack([bits.sum(-1), ((bits & 0xFFFF) * idx).sum(-1),
                        ((bits >> 16) * idx).sum(-1)], dim=-1)
