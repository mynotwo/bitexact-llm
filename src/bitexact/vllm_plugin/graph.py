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

from bitexact.vllm_plugin import paged, tri_fuse, tri_paged

BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128]
FUSE = os.environ.get("BITEXACT_FUSE", "1") == "1"


def _pre(m, ly, x, pos):
    return paged._qkv(m, ly, x, pos)


def _post(m, ly, num, den, x):
    return paged.mlp(m, ly, paged._out(m, ly, paged.attn_div(num, den), x))


_FN = {}


def _fns():
    """The integer ops before and after attention, each as one torch.compile region (BITEXACT_COMPILE_LAYER=1,
    default): Inductor fuses the elementwise ops around the int8 GEMMs. The KV-cache write and the attention
    kernel stay outside the compiled regions: the cache is a dtype view (bf16 storage seen as int32) that a
    compiled region must not mutate (a compiled full-layer region lost the written K/V, and decode logits
    then differed from the eager engine)."""
    if not _FN:
        on = os.environ.get("BITEXACT_COMPILE_LAYER", "1") == "1"
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        wrap = (lambda f: torch.compile(f, dynamic=None)) if on else (lambda f: f)
        _FN["pre"], _FN["post"] = wrap(_pre), wrap(_post)
    return _FN["pre"], _FN["post"]


def decode_forward(m, kvs, ids, pos, slot, bt):
    """ids, pos, slot: [B] device tensors (slot < 0 is redirected to the null block); bt [B, *]."""
    # tri_fuse's kernels take the geometry from the model (1B and 8B); D must be a power of two (one row per program)
    if (FUSE and m.D & (m.D - 1) == 0 and m.H * m.HD == m.D and paged.TRI_MLP and m.nl == 2
            and m.dev.type == "cuda" and not m.post_check):
        return _decode_fused(m, kvs, ids, pos, slot, bt)
    x = m.emb[ids].long()
    pos = pos.long()
    slot = torch.where(slot < 0, torch.zeros_like(slot), slot).long()
    pre, post = _fns()
    for i in range(m.L):
        ly = m.layers[i]
        q, k, v = pre(m, ly, x, pos)
        num, den = paged.decode_store_attend(m, q, k, v, pos, kvs[i], slot, bt)
        x = post(m, ly, num, den, x)
    m._flag(x.abs().amax(1).amax() >= 1 << 42)                # = x.abs().amax(), row-parallel
    return x


def _decode_fused(m, kvs, ids, pos, slot, bt):
    """decode_forward with tri_fuse's kernels between the GEMMs (about 11 kernels per layer instead of 28; same
    integers). No torch.compile regions: the ops that were compiled are now these kernels."""
    pos = pos.long()
    slot = torch.where(slot < 0, torch.zeros_like(slot), slot).long()
    qpos = pos.view(-1, 1).expand(-1, m.H // m.HKV).contiguous()
    ly = m.layers[0]
    x, a, mant, e = tri_fuse.embed_norm(m, ids, ly["ga"])
    for i in range(m.L):
        ly = m.layers[i]
        q = tri_fuse.qkv(m, m._gemm(a, ly["wqkv"][0]), ly["wqkv"], mant, e, pos, slot, paged.kv_view(kvs[i]))
        num, den = tri_paged.attend(q, qpos, paged.kv_view(kvs[i]), bt, m.ptab)
        a, mant, e = tri_fuse.attn_out(m, num, den)
        x, a, mant, e = tri_fuse.add_norm(m, m._gemm(a, ly["wo"][0]), ly["wo"], mant, e, x, ly["gm"])
        a, mant, e = tri_fuse.mlp_mid(m, ly, m._gemm(a, ly["wgu"][0]), mant, e, i)
        nxt = m.layers[i + 1]["ga"] if i + 1 < m.L else None
        x, a, mant, e = tri_fuse.add_norm(m, m._gemm(a, ly["wd"][0]), ly["wd"], mant, e, x, nxt)
    tri_fuse.flag_end(m)
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
