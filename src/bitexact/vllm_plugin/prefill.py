"""Prefill and mixed steps on tri_fuse's kernels (vLLM plugin, CUDA, nl = 2): the same forward as graph._decode_fused,
for flattened token batches with any mix of single-token (decode) and multi-token (prefill / chunk) requests.

The integer ops between the GEMMs are tri_fuse's four kernels per layer (add + norm, qkv + RoPE + cache write,
attention output + quantization, silu + quantization) instead of the ~28 Inductor / ATen kernels of paged.attention
+ paged.mlp, and the prefill attention reads q and writes num / den token-major (tri_prefill, no gather or scatter).
Decode rows use tri_paged.attend as in decode_forward. Every kernel computes the integers of the unfused segments
(tests/test_tri_fuse.py, tests/test_tri_prefill.py), and the rows are independent, so the step is bit-identical to
paged.attention / paged.mlp.
"""

import os

import torch

from bitexact.vllm_plugin import paged, tri_fuse, tri_paged, tri_prefill

ON = os.environ.get("BITEXACT_PREFILL_FUSE", "1") == "1"
MIN_TILES = int(os.environ.get("BITEXACT_PREFILL_MIN_TILES", "132"))   # prefill programs (tiles x KV heads) below which the split-KV attention is used


def applies(m, plan, n_tokens):
    """The fused forward needs tri_fuse's conditions and int32-safe offsets in its kernels (tokens x columns of the
    widest GEMM output: they address the limb-1 half of the GEMM output as M * N)."""
    if not (ON and plan is not None and plan.pre and m.nl == 2 and m.dev.type == "cuda" and not m.post_check
            and paged.TRI_MLP and m.D & (m.D - 1) == 0 and m.H * m.HD == m.D):
        return False
    return n_tokens * max(m.layers[0]["wgu"][0].shape[0], m.layers[0]["wqkv"][0].shape[0]) < 1 << 31


def _prefill_attn(m, i, q, pos, c, plan, tiles, num, den):
    if tiles is not None:
        tri_prefill.attend(q, pos, c, plan.block_table, m.ptab, tiles, num, den)
        return
    for grp in plan.groups():               # few tiles: split-KV kernels (more programs per tile)
        o, d = paged._prefill_group_nd(m, q, c, plan, grp)
        num[grp.dst], den[grp.dst] = o, d


def forward(m, kvs, ids, pos, plan):
    """ids, pos [N]: the step's tokens (rows >= plan.n are padding). Returns the residual stream [N, D] at FR."""
    N = ids.shape[0]
    n = plan.n
    G = m.H // m.HKV
    pos = pos.long()
    slot = torch.full((N,), -1, dtype=torch.long, device=m.dev)
    slot[:n] = plan.slot
    num = torch.zeros(N, m.HKV, G, m.HD, dtype=torch.long, device=m.dev)     # padding rows: o = 0
    den = torch.ones(N, m.HKV, G, 1, dtype=torch.long, device=m.dev)
    tiles = None
    if G in (1, 2, 4, 8) and sum(-(-t[2] // (tri_prefill.BM // G)) for t in plan.pre) * m.HKV >= MIN_TILES:
        tiles = tri_prefill.tile_table(plan.pre, G).to(m.dev, non_blocking=True)
    dq = None
    if plan.dec_req.numel():
        dq = (plan.block_table[plan.dec_req], pos[plan.dec_tok].view(-1, 1).expand(-1, G).contiguous())
    ly = m.layers[0]
    x, a, mant, e = tri_fuse.embed_norm(m, ids, ly["ga"])
    for i in range(m.L):
        ly = m.layers[i]
        c = paged.kv_view(kvs[i])
        q = tri_fuse.qkv(m, m._gemm(a, ly["wqkv"][0]), ly["wqkv"], mant, e, pos, slot, c)
        if dq is not None:
            num[plan.dec_tok], den[plan.dec_tok] = tri_paged.attend(q[plan.dec_tok], dq[1], c, dq[0], m.ptab)
        _prefill_attn(m, i, q, pos, c, plan, tiles, num, den)
        a, mant, e = tri_fuse.attn_out(m, num, den)
        x, a, mant, e = tri_fuse.add_norm(m, m._gemm(a, ly["wo"][0]), ly["wo"], mant, e, x, ly["gm"])
        a, mant, e = tri_fuse.mlp_mid(m, ly, m._gemm(a, ly["wgu"][0]), mant, e, i)
        nxt = m.layers[i + 1]["ga"] if i + 1 < m.L else None
        x, a, mant, e = tri_fuse.add_norm(m, m._gemm(a, ly["wd"][0]), ly["wd"], mant, e, x, nxt)
    tri_fuse.flag_end(m)
    return x
