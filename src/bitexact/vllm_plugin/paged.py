"""Integer attention on vLLM's paged KV cache.

vLLM allocates the KV cache for each layer from the Attention layer we declare (8 KV heads, head size
128, bf16). vLLM 0.32 hands each layer a view [blocks, heads, block_size, K|V] with K and V of one head
side by side in the last dim (2 * 128 bf16 = 512 bytes, unit stride). Viewed as int32 that last dim is
[K (64 int32) | V (64 int32)], exactly our integer K (FQ) and V (FV) of one KV head. So vLLM's block manager, block
tables, slot mapping and prefix cache drive our cache unchanged.

Why the result does not depend on scheduling: the K/V row of a token is a row-local integer function
of the token and its position (same bits whether it came from a cached prefix block, an earlier prefill
chunk or this step). Each query's scores q.k are exact integers; masked keys (beyond the query's
position, padding, stale block contents) get weight 0 before any reduction; the softmax max, the sum
of weights and p.v are exact integer reductions over the valid keys only. So how queries are grouped
(decode batch, prefill chunks, query tiles here) and how many keys are read never changes a bit.
q.k and p.v read K / V in place through the block table (tri_paged.py), no gathered copy.
"""

from dataclasses import dataclass

import torch

from bitexact.int8.consts import FR
from bitexact.int8.segs import FV
from bitexact.intops import rdiv
from bitexact.llama import HD, HKV, H
from bitexact.vllm_plugin import tri_paged

G = H // HKV


@dataclass
class Plan:
    """Host-side batch layout for one forward call (one device-to-host copy per call)."""
    n: int                      # real tokens (rest is padding)
    slot: torch.Tensor          # [n] int64 slot of each token
    block_table: torch.Tensor   # [R, max_blocks] int64
    dec_req: torch.Tensor       # [Rd] requests with one query token (device)
    dec_tok: torch.Tensor       # [Rd] their token index
    pre: list                   # [(req, tok0, qlen, seq_len)] requests with > 1 query token
    pre_groups: list            # batched query tiles of the prefill requests (PreGroup)


@dataclass
class PreGroup:
    req: torch.Tensor           # [n] request of each tile
    idx: torch.Tensor           # [n, Tm] token index of each query (padding: the tile's first token)
    pos: torch.Tensor           # [n, Tm] its position
    valid: torch.Tensor         # [n, Tm] bool
    dst: torch.Tensor           # [n_valid] token index of each valid query (row-major over valid)


GROUP_BYTES = 1 << 28           # cap on the padded int64 query tensor of one prefill group


def _pre_groups(pre, positions, dev):
    """Prefill requests (shortest first) in groups padded to the group's longest query count."""
    groups, cur = [], []

    def flush():
        if not cur:
            return
        Tm = max(t[2] for t in cur)
        ar = torch.arange(Tm)
        start = torch.tensor([t[1] for t in cur])
        valid = ar[None] < torch.tensor([t[2] for t in cur])[:, None]
        idx = torch.where(valid, start[:, None] + ar[None], start[:, None])
        to = lambda v: v.to(dev, non_blocking=True)
        idx_d = to(idx)
        groups.append(PreGroup(req=to(torch.tensor([t[0] for t in cur])), idx=idx_d, pos=positions[idx_d].long(),
                               valid=to(valid), dst=to(idx[valid])))
        cur.clear()

    for t in sorted(pre, key=lambda t: t[2]):
        if cur and (len(cur) + 1) * t[2] * H * HD * 8 > GROUP_BYTES:
            flush()
        cur.append(t)
    flush()
    return groups


def make_plan(meta, n_tokens, positions):
    qsl = meta.query_start_loc.cpu().tolist()
    sl = meta.seq_lens.cpu().tolist()
    R = len(sl)
    n = qsl[R]
    dev = meta.block_table.device
    dec, pre = [], []
    for r in range(R):
        ql = qsl[r + 1] - qsl[r]
        if ql == 1:
            dec.append((r, qsl[r], sl[r]))
        elif ql > 1:
            pre.append((r, qsl[r], ql, sl[r]))
    t = lambda xs: torch.tensor(xs, dtype=torch.long, device=dev)
    return Plan(n=min(n, n_tokens), slot=meta.slot_mapping[:n].long(), block_table=meta.block_table.long(),
                dec_req=t([d[0] for d in dec]), dec_tok=t([d[1] for d in dec]),
                pre=pre, pre_groups=_pre_groups(pre, positions, dev))


def kv_view(kv):
    """vLLM's bf16 cache view [nb, HKV, bs, 2 * 2 * HD] -> int32 [nb, HKV, bs, 2 * HD] (K | V)."""
    assert kv.dim() == 4 and kv.shape[1] == HKV and kv.shape[3] == 4 * HD and kv.stride(3) == 1, \
        (kv.shape, kv.stride())
    return kv.view(torch.int32)


def _store(c, slot, k, v):
    keep = slot >= 0                                                 # dummy / padded tokens: -1
    if not bool(keep.all()):
        slot, k, v = slot[keep], k[keep], v[keep]
    bs = c.shape[2]
    blk, off = slot // bs, slot % bs
    c[blk, :, off] = torch.cat([k, v], -1).to(torch.int32)          # [n, HKV, 2 HD]


def _attend(m, q, qpos, c, bt):
    """q [R, HKV, M, HD] (M = G * T, g-major), qpos [R, M]; keys read in place from the paged cache c
    through block-table rows bt [R, :] -> o = round(sum p v / sum p) at FR, [R, HKV, M, HD]."""
    num, den = tri_paged.attend(q, qpos, c, bt, m.ptab)
    return rdiv(num << (FR - FV), den)


def _prefill_group(m, q, c, plan, grp):
    """Several prefill requests in one kernel call, padded to the longest (Tm queries). Padding queries
    repeat the request's first query (results dropped)."""
    n, Tm = grp.idx.shape
    qt = q[grp.idx].permute(0, 2, 3, 1, 4).reshape(n, HKV, G * Tm, HD)
    qp = grp.pos.view(n, 1, Tm).expand(n, G, Tm).reshape(n, G * Tm)
    ot = _attend(m, qt, qp, c, plan.block_table[grp.req])
    return ot.view(n, HKV, G, Tm, HD).permute(0, 3, 1, 2, 4)[grp.valid]


def _qkv(m, ly, x, pos):
    N = x.shape[0]
    y = m.norm_linear(x, ly["ga"], ly["wqkv"]).view(N, 1, -1)
    qg, k, v, flag = m.sg.qkv(y, m.cos[pos][:, None, None], m.sin[pos][:, None, None])
    m._flag(flag)
    return qg.view(N, HKV, G, HD), k[:, 0], v[:, 0]


def _out(m, ly, o, x):
    return m._linear(m.sg.pre_plain(o.reshape(x.shape[0], H * HD)), ly["wo"], x)


def mlp(m, ly, x):
    """IntLlama.mlp for flattened rows x [N, D], with the layer's tensors passed in (ly)."""
    gu = m.norm_linear(x, ly["gm"], ly["wgu"])
    a8, mant, e, flag = m.sg.silu_mul(gu, m.silu_t)
    m._flag(flag)
    return m._linear((a8, mant, e), ly["wd"], x)


@torch.compiler.disable
def decode_store_attend(m, q, k, v, pos, kv, slot, blocks):
    """All rows are single-token decodes, no host syncs (CUDA-graph safe): write K / V at slot, then attend.
    Kept out of torch.compile (the cache is a dtype view and must be written in place). Padding rows: slot 0
    (vLLM's null block 0, never given to a request) and position 0."""
    c = kv_view(kv)
    bs = c.shape[2]
    c[slot // bs, :, slot % bs] = torch.cat([k, v], -1).to(torch.int32)
    return _attend(m, q, pos.view(-1, 1).expand(-1, G), c, blocks)


def attention(m, i, x, pos, kv, plan):
    """One attention block of IntLlama m (layer i) for flattened tokens x [N, D] at positions pos [N].
    Returns x + attn(x) at FR. plan None (memory profiling run): K/V are not stored, attention is 0."""
    N = x.shape[0]
    p = pos.long()
    ly = m.layers[i]
    q, k, v = _qkv(m, ly, x, p)
    o = torch.zeros(N, HKV, G, HD, dtype=torch.long, device=x.device)
    if plan is not None:
        c = kv_view(kv)
        n = plan.n
        _store(c, plan.slot, k[:n], v[:n])
        if plan.dec_req.numel():
            blocks = plan.block_table[plan.dec_req]
            qp = p[plan.dec_tok].view(-1, 1).expand(-1, G)
            o[plan.dec_tok] = _attend(m, q[plan.dec_tok], qp, c, blocks)
        for grp in plan.pre_groups:
            o[grp.dst] = _prefill_group(m, q, c, plan, grp)
    return _out(m, ly, o, x)
