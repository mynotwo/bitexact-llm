"""Exact mode (original bf16 weights, bitexact.exact) on vLLM's flattened token batch and paged KV cache.

Every token is a row: it is stored into the paged limb-plane cache (tri_paged.store: K at 17 bits, V at 19
bits, 3 int8 limbs each) and attends its own keys through its request's block table (tri_exact_attn). The
linears, norms, RoPE, SiLU and the attention output division are exact/segs.py's stages unchanged, so the
integers equal the standalone ExactLlama's; every rule is row-local, so batch, padding and chunking cannot
change a bit. The forward uses device tensors only (CUDA-graph safe).
"""

import torch

from bitexact.exact import ops
from bitexact.exact.model import ExactLlama
from bitexact.vllm_plugin import paged, tri_exact_attn, tri_paged


class ExactEngine:
    def __init__(self, c, dev, compiled=False, gemm="triton"):
        self.x = m = ExactLlama(c, dev, compiled=compiled, gemm=gemm)
        self.dev = m.dev
        self.L, self.H, self.HKV, self.HD, self.D = m.L, m.H, m.HKV, m.HD, m.D
        self.FR, self.ln, self.emb, self.sha = m.FR, m.ln, m.emb, m.sha
        self.l2e = ops.L2E28_HD[m.HD]

    @property
    def err(self):
        return self.x.err

    def check(self):
        self.x.check()

    def forward(self, kvs, ids, pos, slot, bt, rmap):
        """ids, pos, slot [N]: the step's tokens (slot < 0: padding, redirected to the null block 0);
        bt [*, nblk] block table and rmap [N] the table row of each token. kvs None: no cache (memory
        profiling run, attention output 0). Returns the residual stream [N, D] int64 at FR."""
        m = self.x
        N = ids.shape[0]
        G = self.H // self.HKV
        pos = pos.long()
        slot = torch.where(slot < 0, torch.zeros_like(slot), slot).long()
        qpos = pos.view(-1, 1).expand(-1, G)
        x = m.emb[ids].long()
        q, sh = m.sg.pre_norm(x, *m.layers[0]["ga"])
        err = torch.zeros((), dtype=torch.bool, device=x.device)
        for i in range(self.L):
            ly = m.layers[i]
            qg, shg, k, v, err = m.sg.qkv(*ly["wqkv"](q), sh, pos.view(N, 1), m.cos, m.sin, N, 1, err)
            if kvs is None:
                num = torch.zeros(N * self.HKV, G, self.HD, dtype=torch.long, device=x.device)
                den = torch.ones(N * self.HKV, G, 1, dtype=torch.long, device=x.device)
            else:
                c = paged.kv_view(kvs[i])
                tri_paged.store(c, slot, k[:, 0], v[:, 0])
                num, den = tri_exact_attn.attend(qg.view(N, self.HKV, G, self.HD), shg.view(N, self.HKV, G),
                                                 qpos, c, bt, rmap, m.exp2, self.l2e)
                num, den = num.view(N * self.HKV, G, self.HD), den.view(N * self.HKV, G, 1)
            qo, sho = m.sg.attn_out(num, den, 1)
            x, q, sh, err = m.sg.add_norm(*ly["wo"](qo), sho, x, *ly["gm"], err)
            q, sh, err = m.sg.gu_silu(*ly["wgu"](q), sh, m.exp2, err)
            nxt = m.layers[i + 1]["ga"] if i + 1 < self.L else m.gf
            x, q, sh, err = m.sg.add_norm(*ly["wd"](q), sh, x, *nxt, err)
        m.err |= err
        return x

    def logits(self, h):
        """Final residual rows [M, D] -> int64 logits [M, V] at FR."""
        m = self.x
        q, sh = m.sg.pre_norm(h, *m.gf)
        y, flag = m.sg.lin_out(*m.lm(q), sh)
        m.err |= flag
        return y
