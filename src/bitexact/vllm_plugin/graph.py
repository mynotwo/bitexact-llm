"""Decode-only steps without host syncs, as CUDA graphs.

decode_forward runs the whole integer forward for rows that are each one new token of their own
request, using only device tensors (token, position, slot, block-table row). The fused attention
kernel bounds its key loop by each tile's largest position on the device, so no sequence length is
needed on the host. Two users:
  - vLLM's own CUDA graphs (no enforce_eager): vLLM captures uniform-decode batches (FULL_DECODE_ONLY,
    since the model has no torch.compile support); model.forward calls decode_forward while capturing.
  - enforce_eager=True: DecodeGraphs captures it per padded row count (1, 2, 4, ..., 128).
Padding rows use slot 0 / position 0 (vLLM's null block 0 is never given to a request, so the junk K/V
written there is never read). Replays run the same kernels on the same integers as eager mode and every
row is row-local, so padding and replay do not change any bit (the invariance tests run through both).
"""

import os

import torch

from bitexact.vllm_plugin import paged

BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128]


def _layer(m, ly, x, pos, kv, slot, bt):
    return paged.mlp(m, ly, paged.decode_attention(m, ly, x, pos, kv, slot, bt))


_LAYER = {}


def _layer_fn():
    """One decoder layer, optionally as one torch.compile region (BITEXACT_COMPILE_LAYER=1, experimental,
    off by default): with vLLM 0.32 / torch 2.13 the compiled layer replayed inside the plugin's CUDA graphs
    gave logits different from the eager engine, so it is not part of the tested configuration."""
    if "f" not in _LAYER:
        on = os.environ.get("BITEXACT_COMPILE_LAYER", "0") == "1"
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        _LAYER["f"] = torch.compile(_layer, dynamic=None) if on else _layer
    return _LAYER["f"]


def decode_forward(m, kvs, ids, pos, slot, bt):
    """ids, pos, slot: [B] device tensors (slot < 0 is redirected to the null block); bt [B, *]."""
    x = m.emb[ids].long()
    pos = pos.long()
    slot = torch.where(slot < 0, torch.zeros_like(slot), slot).long()
    f = _layer_fn()
    for i in range(m.L):
        x = f(m, m.layers[i], x, pos, kvs[i], slot, bt)
    m._flag(x.abs().amax() >= 1 << 42)
    return x


class DecodeGraphs:
    def __init__(self, m, kvs, max_len):
        self.m, self.kvs = m, kvs
        dev = m.dev
        Bm = BUCKETS[-1]
        bs = paged.kv_view(kvs[0]).shape[2]
        self.tok = torch.zeros(Bm, dtype=torch.long, device=dev)
        self.pos = torch.zeros(Bm, dtype=torch.long, device=dev)
        self.slot = torch.zeros(Bm, dtype=torch.long, device=dev)
        self.bt = torch.zeros(Bm, (max_len + bs - 1) // bs, dtype=torch.long, device=dev)
        self.graphs = {}
        self.pool = None

    def _core(self, B):
        return decode_forward(self.m, self.kvs, self.tok[:B], self.pos[:B], self.slot[:B], self.bt[:B])

    def _capture(self, B):
        # Warm-up and capture write K/V of the current buffer contents: all-zero slots = null block 0.
        for t in (self.slot, self.pos, self.bt):
            t.zero_()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                self._core(B)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            out = self._core(B)
        self.pool = g.pool()
        return g, out

    def applies(self, plan, N):
        return plan is not None and not plan.pre and plan.n == N and 0 < N <= BUCKETS[-1]

    def run(self, ids, positions, plan):
        N = plan.n
        B = next(b for b in BUCKETS if b >= N)
        if B not in self.graphs:
            self.graphs[B] = self._capture(B)
        g, out = self.graphs[B]
        nblk = min(self.bt.shape[1], plan.block_table.shape[1])
        self.tok[:N].copy_(ids[:N])
        self.pos[:N].copy_(positions[:N])
        self.slot[:N].copy_(plan.slot)
        self.bt[:N, :nblk].copy_(plan.block_table[plan.dec_req, :nblk])
        if N < B:
            for t in (self.tok, self.pos, self.slot, self.bt):
                t[N:B].zero_()
        g.replay()
        return out[:N].clone()
