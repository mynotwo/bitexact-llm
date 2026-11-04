"""Packed (varlen) teacher-forced scorer for the int8 engine.

Scorer.score(batch) returns the log p of every response token, from the same integer logits the plugin produces.
The batch is one flat stream of real tokens (no padding rows). Every operation of the engine is row-local except
attention, and attention reads the K / V of its own sequence only, so the order and grouping of rows cannot change
a bit. Per chunk of `rows` stream tokens:

  * linear layers and element-wise segments run on [n, ...] rows (compiled segments of IntLlama);
  * K / V go to a paged int8 limb-plane cache (tri_paged.store; one run of blocks per sequence) and attention is
    the plugin's fused two-sweep
    kernel tri_paged.attend on query tiles of 64 tokens (the tail of a sequence in tiles of 16); keys past a tile's
    last position are never read and the scores never reach memory;
  * only the rows that predict a response token leave the last layer; their float64 log-softmax is accumulated on
    the device and copied to the host once per batch.
The whole host-side layout (ids, positions, cache slots, tile gather / scatter indices) is built with numpy for the
batch and uploaded in one copy, and no step reads a device value on the host, so a chunk is a fixed sequence of kernel
launches (the only syncs are the final copy and the overflow-flag check).

On CPU score() runs the padded reference flow (float64 GEMM), which gives the same integer logits (bit for bit,
the row hashes are equal). The log p are computed from them in float64 on the device in use, so a CPU and a GPU
log p may differ in the last bits (about 1e-13); on one device they are bit-equal for any batch composition.
"""

import hashlib

import numpy as np
import torch

from bitexact.int8.segs import Segs
from bitexact.llama import PAD

BUCKET = 128                          # chunk row counts are multiples of this (static compiled shapes)
TILES = (64, 16)                      # query tile sizes: full tiles, then the tail of a sequence


def sha_rows(lg):
    a = lg.long().cpu().numpy()
    return [hashlib.sha256(a[t].tobytes()).hexdigest()[:16] for t in range(a.shape[0])]


def logp_rows(lg, tgt):
    """log p of tgt [n] under int64 logits rows lg [n, V] at 16 fractional bits (float64 log-softmax)."""
    return (lg.double() * 2.0 ** -16).log_softmax(-1)[torch.arange(lg.shape[0], device=lg.device), tgt]


def score_padded(m, batch, chunk=256, want_hash=False):
    """Reference flow: right-padded batch, prefill in position chunks, lm_head + log-softmax per response chunk."""
    _check_prompts(batch)
    seqs = [p + r for p, r in batch]
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
    pos = torch.arange(L).expand(len(seqs), L).contiguous()
    ids, pos = ids.to(m.dev), pos.to(m.dev)
    cache = m.new_cache(len(batch), L)
    h = torch.cat([m.forward(ids[:, c:min(L, c + chunk)], pos[:, c:min(L, c + chunk)], cache, S=min(L, c + chunk))
                   for c in range(0, L, chunk)], 1)
    out = []
    for j, (p, r) in enumerate(batch):
        lps, hs = [], []
        for c0 in range(0, len(r), chunk):
            c1 = min(len(r), c0 + chunk)
            lg = m.logits(h[j, len(p) - 1 + c0:len(p) - 1 + c1])
            hs += sha_rows(lg) if want_hash else []
            lps.append(logp_rows(lg, torch.tensor(r[c0:c1], device=lg.device)))
        out.append((torch.cat(lps).cpu().numpy() if lps else np.empty(0), hs if want_hash else None))
    m.check()
    return out


def _check_prompts(batch):
    """The log p of the first response token is read at the last prompt token: a prompt needs one."""
    for i, (p, _) in enumerate(batch):
        if len(p) == 0:
            raise ValueError(f"batch[{i}]: empty prompt (the first response token has no context to be scored from)")


def _tiles(a, b):
    """Query tiles covering stream tokens [a, b) of one sequence: (first, length, tile size) arrays."""
    nfull = (b - a) // TILES[0]
    f1 = a + TILES[0] * np.arange(nfull)
    a2 = a + TILES[0] * nfull
    f2 = a2 + TILES[1] * np.arange(-(-(b - a2) // TILES[1]))
    return [(f1, np.full(nfull, TILES[0]), TILES[0]), (f2, np.minimum(TILES[1], b - f2), TILES[1])]


class _Batch:
    """Host-side layout of a batch (numpy); upload() puts it on the device as one int64 buffer."""

    def __init__(self, batch, rows, bs, H, HKV):
        G = H // HKV
        seqs = [p + r for p, r in batch]
        lens = np.array([len(s) for s in seqs])
        offs = np.cumsum(lens) - lens
        self.N = int(lens.sum())
        nbl = -(-lens // bs)
        base = np.cumsum(nbl) - nbl                                    # first block of each sequence
        self.NB = int(nbl.sum())
        sid = np.repeat(np.arange(len(seqs)), lens)
        pos = np.arange(self.N) - offs[sid]
        slot = base[sid] * bs + pos
        need = np.concatenate([offs[j] + len(p) - 1 + np.arange(len(r)) for j, (p, r) in enumerate(batch)])
        self.nr = [len(r) for _, r in batch]
        self.pieces = []                                               # int64 arrays, in upload order
        self.idx = {}

        def put(name, a):
            a = np.asarray(a, dtype=np.int64).reshape(-1)
            self.idx[name] = (sum(map(len, self.pieces)), len(a))
            self.pieces.append(a)

        ids = np.concatenate([np.asarray(s, dtype=np.int64) for s in seqs])
        put("tgt", np.concatenate([np.asarray(r, dtype=np.int64) for _, r in batch]))
        self.chunks = []
        nch = -(-self.N // rows)
        per = -(-self.N // nch)
        n = -(-per // BUCKET) * BUCKET                                 # static row count of every chunk
        for c0 in range(0, self.N, per):
            c1 = min(self.N, c0 + per)
            pad = n - (c1 - c0)
            put(f"c{c0}i", np.concatenate([ids[c0:c1], np.full(pad, PAD)]))
            put(f"c{c0}p", np.concatenate([pos[c0:c1], np.zeros(pad)]))
            ov = np.nonzero((offs < c1) & (offs + lens > c0))[0]
            groups = []
            for k in range(2):
                f, ln, bb = [], [], []
                for j in ov:
                    t = _tiles(max(c0, offs[j]), min(c1, offs[j] + lens[j]))[k]
                    f.append(t[0] - c0)
                    ln.append(t[1])
                    bb.append(np.full(len(t[0]), base[j]))
                f, ln, bb = (np.concatenate(x) for x in (f, ln, bb))
                T, nt = TILES[k], len(f)
                if nt == 0:
                    continue
                ar = np.arange(T)[None]
                valid = ar < ln[:, None]
                tok = np.where(valid, f[:, None] + ar, f[:, None])                     # [nt, T] chunk-local token
                row = tok[:, None, None, :] * H + np.arange(HKV)[None, :, None, None] * G \
                    + np.arange(G)[None, None, :, None]                                # [nt, HKV, G, T] rows of q
                dst = np.where(np.broadcast_to(valid[:, None, None, :], row.shape), row, n * H)
                qpos = np.broadcast_to((pos[c0 + tok])[:, None, :], (nt, G, T)).reshape(nt, G * T)
                nm = f"c{c0}t{T}"
                put(nm + "g", row)
                put(nm + "d", dst)
                put(nm + "q", qpos)
                put(nm + "b", bb)
                groups.append((T, nt, nm, int(nbl[ov].max())))
            nm = f"c{c0}"
            put(nm + "w", np.concatenate([slot[c0:c1], np.full(pad, -1)]))         # padding rows: not stored
            put(nm + "n", need[(need >= c0) & (need < c1)] - c0)
            self.chunks.append((c0, n, groups))

    def upload(self, dev):
        buf = torch.from_numpy(np.concatenate(self.pieces)).pin_memory().to(dev, non_blocking=True)
        return lambda name: buf[self.idx[name][0]:self.idx[name][0] + self.idx[name][1]]


class Scorer:
    def __init__(self, m, rows=2048, bs=16, lm_rows=512, attn_kw=None):
        if m.sg.compiled and m.sg.dynamic is not False:                  # specialised kernels per bucketed shape
            g = m.sg
            m.sg = Segs(g.nl, g.cuda, True, g.post_check, g.dm, dynamic=False)
        self.m, self.rows, self.bs, self.lm_rows = m, rows, bs, lm_rows
        if m.dev.type == "cuda" and not hasattr(m, "ptab"):              # IntLlama(fused=False): table built here
            from bitexact.vllm_plugin import tri_paged
            m.ptab = tri_paged.weight_table(m.exp)
        self.attn_kw = attn_kw or {}                                     # launch parameters of tri_paged.attend
        self.c = []                                                    # paged K | V per layer, grown on demand

    def _cache(self, NB):
        m = self.m
        if not self.c or self.c[0].shape[0] < NB:                          # stale blocks are never read: no zeroing
            self.c = [torch.zeros(NB, m.HKV, self.bs, 6 * m.HD, dtype=torch.int8, device=m.dev) for _ in range(m.L)]

    def score(self, batch, want_hash=False):
        """batch: [(prompt ids, response ids)] -> [(log p float64 [len(response)], row hashes or None)].
        A prompt must have at least one token (ValueError); a response may be empty (empty result)."""
        m = self.m
        _check_prompts(batch)
        if not any(len(r) for _, r in batch):
            return [(np.empty(0), [] if want_hash else None) for _ in batch]
        if m.dev.type != "cuda":
            return score_padded(m, batch, want_hash=want_hash)
        b = _Batch(batch, self.rows, self.bs, m.H, m.HKV)
        self._cache(b.NB)
        get = b.upload(m.dev)
        tgt = get("tgt")
        h = torch.cat([self._chunk(get, c0, n, groups) for c0, n, groups in b.chunks])
        lps, hs = [], []
        for a in range(0, h.shape[0], self.lm_rows):
            k = min(self.lm_rows, h.shape[0] - a)
            hb = h[a:a + k]
            lg = m.logits(torch.cat([hb, hb.new_zeros(-(-k // 128) * 128 - k, m.D)]))[:k]       # static row counts
            if want_hash:
                hs += sha_rows(lg)
            lps.append(logp_rows(lg, tgt[a:a + k]))
        lp = torch.cat(lps).cpu().numpy()
        m.check()
        out, o = [], 0
        for n in b.nr:
            out.append((lp[o:o + n], hs[o:o + n] if want_hash else None))
            o += n
        return out

    def _chunk(self, get, c0, n, groups):
        """Layers over one chunk of n stream tokens (padding rows included) -> final residual rows of the
        response-predicting tokens."""
        from bitexact.vllm_plugin import tri_paged
        m = self.m
        H, HKV, HD = m.H, m.HKV, m.HD
        G = H // HKV
        ids, pos = get(f"c{c0}i"), get(f"c{c0}p")
        x = m.emb[ids].long()
        cos, sin = m.cos[pos][:, None, None], m.sin[pos][:, None, None]
        slot = get(f"c{c0}w")
        grp = []
        for T, nt, nm, nw in groups:
            bt = get(nm + "b")[:, None] + torch.arange(nw, device=m.dev)[None]
            grp.append((nt, get(nm + "g"), get(nm + "d"), get(nm + "q").view(nt, G * T), bt))
        for i in range(m.L):
            ly = m.layers[i]
            y = m.norm_linear(x, ly["ga"], ly["wqkv"])
            qr, k, v, flag = m.sg.qkv_rows(y, cos, sin)
            m._flag(flag)
            c = self.c[i]
            tri_paged.store(c, slot, k, v)
            # rows of padding tokens stay 0; the last row takes the writes of padded queries
            o = torch.zeros(n * H + 1, HD, dtype=torch.long, device=m.dev)
            for nt, gidx, didx, qpos, bt in grp:
                qt = qr.index_select(0, gidx).view(nt, HKV, -1, HD)
                num, den = tri_paged.attend(qt, qpos, c, bt, m.ptab, **self.attn_kw)
                o.index_copy_(0, didx, m.sg.attn_rows(num, den).view(-1, HD))
            x = m._linear(m.sg.pre_plain(o[:n * H].view(n, H * HD)), ly["wo"], x)
            gu = m.norm_linear(x, ly["gm"], ly["wgu"])
            a8, mant, e, flag = m.sg.silu_mul(gu, m.silu_t)
            m._flag(flag)
            x = m._linear((a8, mant, e), ly["wd"], x)
        m._flag(x.abs().amax() >= 1 << 42)
        return x[get(f"c{c0}n")]
