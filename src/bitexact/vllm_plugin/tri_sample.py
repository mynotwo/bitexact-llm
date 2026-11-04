"""LM-head post-processing, greedy argmax and integer Gumbel sampling straight from the GEMM output.

Before: segs.post (Inductor) joined the two int32 limb products of the LM-head GEMM into int64 logits [M, V]
(write 8 B / logit), tri_logits read them again for vLLM's float copy and the greedy argmax, and a T > 0
step generated the PCG words (write 8 B), ran the Gumbel/rdiv kernels (read and write intermediates) and
another argmax over the logits. Here the int64 logits are never stored:

  head   (_head_kernel, _comb_kernel): one pass over c = torch._int_mm(limbs, Wlm^T): each element goes
         through segs.post (join limbs, rsh(acc * m, fmsh), rsh(t * mant, 24 - FR - e)) and gives the
         first argmax of its row (same rule as tri_logits). It writes either the float logits li * 2^-16
         (as tri_logits did; needed when a request asks for logprobs) or the row "one-hot" float logits
         (-inf everywhere, 0 at the argmax), which is what vLLM's own greedy argmax needs and nothing more.
  noisy  (_noise_kernel, _comb_kernel): same pass for T > 0: PCG64 word i of the row (gpu_pcg's closed form),
         integer Gumbel noise (sampling.gumbel), rdiv(l * td, tn) + g, first argmax. Rows with tn = 0 take the
         plain logits (greedy). The row's seed state, tn, td come from the host in one int64 [M, 6] tensor.

Same integers as the unfused code: every element runs the same int64 operations in the same order
(+, * wrap alike; shifts follow torch's rules via tri_mlp._rsh), ln_q's bitlen is 64 - clz (exact), rdiv is
the exact floor of (2a + b) / 2b (Triton's integer division truncates; the remainder test floors it), and the
argmax is the exact first maximum (see tri_logits). tests/test_vllm_kernels.py compares with the eager rule.
"""

import torch
import triton
import triton.language as tl

from bitexact import sampling
from bitexact.vllm_plugin import gpu_pcg
from bitexact.vllm_plugin.gpu_pcg import _mulhi
from bitexact.vllm_plugin.tri_mlp import FRC, _bitlen, _rsh, _y

NEG_INF = tl.constexpr(float("-inf"))
SHG = tl.constexpr(sampling.FL - 16)       # FL - FR
LN2Q = tl.constexpr(sampling.LN2Q)
UMIN = tl.constexpr(sampling.UMIN)
UMAX = tl.constexpr((1 << 53) - sampling.UMIN)
TWO53 = tl.constexpr(1 << 53)
C53LN2 = tl.constexpr(53 * sampling.LN2Q)
CFLLN2 = tl.constexpr(sampling.FL * sampling.LN2Q)
SMALL = tl.constexpr(1 << 45)
SERDIV = tl.constexpr(24 << 24)
BIGI = tl.constexpr(1 << 62)
IMINI = tl.constexpr(-(1 << 63))

M32 = tl.constexpr(0xFFFFFFFF)
SEEDB = tl.constexpr(sampling.SEED_BASE)
MULT_H = tl.constexpr(gpu_pcg.MULT >> 64)
MULT_L = tl.constexpr(gpu_pcg.MULT & ((1 << 64) - 1))


@triton.jit
def _hm(v, hc):
    """numpy SeedSequence hashmix (uint32 in int64): returns (value, next hash const)."""
    v = v ^ hc
    hc = (hc * 0x931E8875) & M32
    v = (v * hc) & M32
    return v ^ (v >> 16), hc


@triton.jit
def _mx(x, y):
    r = (0xCA01F9DD * x - 0x4973F715 * y) & M32
    return r ^ (r >> 16)


@triton.jit
def _seed_state(seed, t):
    """numpy PCG64(SeedSequence([SEED_BASE, seed, t])).state as uint64 (state hi, lo, inc hi, lo); seed, t < 2^32.
    SeedSequence.mix_entropy (pool of 4) -> generate_state(4, uint64) -> pcg64_set_seed."""
    p0, hc = _hm(seed * 0 + SEEDB, seed * 0 + 0x43B0D7E5)
    p1, hc = _hm(seed, hc)
    p2, hc = _hm(t, hc)
    p3, hc = _hm(seed * 0, hc)
    # mixing: for each source word, each other word is mixed with a fresh hashmix of the source (hash const advances
    # per call, in numpy's loop order)
    h, hc = _hm(p0, hc)
    p1 = _mx(p1, h)
    h, hc = _hm(p0, hc)
    p2 = _mx(p2, h)
    h, hc = _hm(p0, hc)
    p3 = _mx(p3, h)
    h, hc = _hm(p1, hc)
    p0 = _mx(p0, h)
    h, hc = _hm(p1, hc)
    p2 = _mx(p2, h)
    h, hc = _hm(p1, hc)
    p3 = _mx(p3, h)
    h, hc = _hm(p2, hc)
    p0 = _mx(p0, h)
    h, hc = _hm(p2, hc)
    p1 = _mx(p1, h)
    h, hc = _hm(p2, hc)
    p3 = _mx(p3, h)
    h, hc = _hm(p3, hc)
    p0 = _mx(p0, h)
    h, hc = _hm(p3, hc)
    p1 = _mx(p1, h)
    h, hc = _hm(p3, hc)
    p2 = _mx(p2, h)
    hb = seed * 0 + 0x8B51F9DD
    o0 = (p0 ^ hb)
    hb = (hb * 0x58F38DED) & M32
    o0 = (o0 * hb) & M32
    o0 = o0 ^ (o0 >> 16)
    o1 = (p1 ^ hb)
    hb = (hb * 0x58F38DED) & M32
    o1 = (o1 * hb) & M32
    o1 = o1 ^ (o1 >> 16)
    o2 = (p2 ^ hb)
    hb = (hb * 0x58F38DED) & M32
    o2 = (o2 * hb) & M32
    o2 = o2 ^ (o2 >> 16)
    o3 = (p3 ^ hb)
    hb = (hb * 0x58F38DED) & M32
    o3 = (o3 * hb) & M32
    o3 = o3 ^ (o3 >> 16)
    o4 = (p0 ^ hb)
    hb = (hb * 0x58F38DED) & M32
    o4 = (o4 * hb) & M32
    o4 = o4 ^ (o4 >> 16)
    o5 = (p1 ^ hb)
    hb = (hb * 0x58F38DED) & M32
    o5 = (o5 * hb) & M32
    o5 = o5 ^ (o5 >> 16)
    o6 = (p2 ^ hb)
    hb = (hb * 0x58F38DED) & M32
    o6 = (o6 * hb) & M32
    o6 = o6 ^ (o6 >> 16)
    o7 = (p3 ^ hb)
    hb = (hb * 0x58F38DED) & M32
    o7 = (o7 * hb) & M32
    o7 = o7 ^ (o7 >> 16)
    w0 = (o0 | (o1 << 32)).to(tl.uint64, bitcast=True)
    w1 = (o2 | (o3 << 32)).to(tl.uint64, bitcast=True)
    w2 = (o4 | (o5 << 32)).to(tl.uint64, bitcast=True)
    w3 = (o6 | (o7 << 32)).to(tl.uint64, bitcast=True)
    il = (w3 << 1) | 1                                    # inc = (initseq << 1) | 1
    ih = (w2 << 1) | (w3 >> 63)
    lo = il + w1                                          # state = inc + initstate
    hi = ih + w0 + (((il & w1) | ((il | w1) & ~lo)) >> 63)
    ml = w1 * 0 + MULT_L                                  # state = state * MULT + inc
    mh = w1 * 0 + MULT_H
    lo2 = lo * ml
    hi2 = _mulhi(lo, ml) + lo * mh + hi * ml
    sl = lo2 + il
    sh = hi2 + ih + (((lo2 & il) | ((lo2 | il) & ~sl)) >> 63)
    return sh, sl, ih, il


@triton.jit
def _ln_q(n, lnt_ptr):
    """sampling.ln_q for int64 n >= 1."""
    k = _bitlen(n) - 1
    m = tl.where(k <= 52, n << tl.maximum(52 - k, 0), n >> tl.maximum(k - 52, 0))
    idx = (m >> 36) - (1 << 16)
    r = (m >> 16) & ((1 << 20) - 1)
    a = tl.load(lnt_ptr + idx)
    return k * LN2Q + a + (((tl.load(lnt_ptr + idx + 1) - a) * r) >> 20)


@triton.jit
def _gumbel(raw, lnt_ptr):
    """sampling.gumbel(raw, lnt, FR). One ln_q call serves both branches (it takes Dd or A)."""
    U = tl.minimum(tl.maximum((raw >> 11) & (TWO53 - 1), UMIN), UMAX)
    Dd = TWO53 - U
    A = C53LN2 - _ln_q(U, lnt_ptr)
    small = Dd < SMALL
    ln = _ln_q(tl.where(small, Dd, tl.maximum(A, 1)), lnt_ptr)
    dq = Dd >> 13
    lna_ser = ln - C53LN2 + (dq >> 1) + (5 * ((dq >> 8) * (dq >> 8))) // SERDIV
    lna = tl.where(small, lna_ser, ln - CFLLN2)
    return (-lna + (1 << (SHG - 1))) >> SHG


@triton.jit
def _head_kernel(c_ptr, m_ptr, fmsh_ptr, mant_ptr, e_ptr, out_ptr, pmax, pidx, M, V, CH, NCH,
                 MODE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    ch = tl.program_id(1)
    fmsh = tl.load(fmsh_ptr)
    mant = tl.load(mant_ptr + row)
    e = tl.load(e_ptr + row)
    lane = tl.arange(0, BLOCK)
    vmax = tl.full([BLOCK], IMINI, tl.int64)
    vidx = tl.full([BLOCK], BIGI, tl.int64)
    base = row.to(tl.int64) * V
    for c0 in range(ch * CH, tl.minimum(ch * CH + CH, V), BLOCK):
        col = c0 + lane
        ok = col < V
        _, x = _y(c_ptr, m_ptr, col, ok, row, M, V, fmsh, mant, e)
        x = tl.where(ok, x, IMINI)
        if MODE == 1:
            tl.store(out_ptr + base + col, x.to(tl.float32) * 1.52587890625e-05, mask=ok)
        elif MODE == 2:
            tl.store(out_ptr + base + col, tl.full([BLOCK], NEG_INF, tl.float32), mask=ok)
        take = ok & ((x > vmax) | (vidx == BIGI))
        vmax = tl.where(take, x, vmax)
        vidx = tl.where(take, col.to(tl.int64), vidx)
    mx = tl.max(vmax, 0)
    i = tl.min(tl.where(vmax == mx, vidx, BIGI), 0)
    tl.store(pmax + row * NCH + ch, mx)
    tl.store(pidx + row * NCH + ch, i)


@triton.jit
def _noise_kernel(c_ptr, m_ptr, fmsh_ptr, mant_ptr, e_ptr, st_ptr, tab_ptr, lnt_ptr, pmax, pidx, M, V, CH, NCH,
                  out_ptr, BLOCK: tl.constexpr, SEEDED: tl.constexpr, FILL: tl.constexpr):
    row = tl.program_id(0)
    ch = tl.program_id(1)
    fmsh = tl.load(fmsh_ptr)
    mant = tl.load(mant_ptr + row)
    e = tl.load(e_ptr + row)
    if SEEDED:      # st [M, 4] = seed, t, tn, td: the PCG state is derived here (_seed_state)
        sp = st_ptr + row.to(tl.int64) * 4
        tn = tl.load(sp + 2)
        td = tl.load(sp + 3)
        sh, sl, ch_, cl = _seed_state(tl.load(sp), tl.load(sp + 1))
    else:           # st [M, 6] = PCG state hi, lo, inc hi, lo, tn, td (row_state)
        sp = st_ptr + row.to(tl.int64) * 6
        tn = tl.load(sp + 4)
        td = tl.load(sp + 5)
        sh = tl.load(sp).to(tl.uint64, bitcast=True)
        sl = tl.load(sp + 1).to(tl.uint64, bitcast=True)
        ch_ = tl.load(sp + 2).to(tl.uint64, bitcast=True)
        cl = tl.load(sp + 3).to(tl.uint64, bitcast=True)
    lane = tl.arange(0, BLOCK)
    vmax = tl.full([BLOCK], IMINI, tl.int64)
    vidx = tl.full([BLOCK], BIGI, tl.int64)
    for c0 in range(ch * CH, tl.minimum(ch * CH + CH, V), BLOCK):
        col = c0 + lane
        ok = col < V
        _, x = _y(c_ptr, m_ptr, col, ok, row, M, V, fmsh, mant, e)
        if FILL:         # vLLM's logits row: -inf here, 0 at the chosen token (_comb_kernel)
            tl.store(out_ptr + row.to(tl.int64) * V + col, tl.full([BLOCK], NEG_INF, tl.float32), mask=ok)
        if tn == 0:
            v = x
        else:
            ah = tl.load(tab_ptr + col, mask=ok, other=0).to(tl.uint64, bitcast=True)
            al = tl.load(tab_ptr + V + col, mask=ok, other=0).to(tl.uint64, bitcast=True)
            gh = tl.load(tab_ptr + 2 * V + col, mask=ok, other=0).to(tl.uint64, bitcast=True)
            gl = tl.load(tab_ptr + 3 * V + col, mask=ok, other=0).to(tl.uint64, bitcast=True)
            lo1 = al * sl
            hi1 = _mulhi(al, sl) + al * sh + ah * sl
            lo2 = gl * cl
            hi2 = _mulhi(gl, cl) + gl * ch_ + gh * cl
            lo = lo1 + lo2
            carry = ((lo1 & lo2) | ((lo1 | lo2) & ~lo)) >> 63
            hi = hi1 + hi2 + carry
            w = hi ^ lo
            rot = hi >> 58
            raw = ((w >> rot) | (w << ((64 - rot) & 63))).to(tl.int64, bitcast=True)
            g = _gumbel(raw, lnt_ptr)
            if tn == 1:
                v = x * td + g
            else:
                a = x * td
                n = 2 * a + tn
                d = 2 * tn
                q = n // d
                q = tl.where(n - q * d < 0, q - 1, q)        # floor, whatever the sign convention of //
                v = q + g
        v = tl.where(ok, v, IMINI)
        take = ok & ((v > vmax) | (vidx == BIGI))
        vmax = tl.where(take, v, vmax)
        vidx = tl.where(take, col.to(tl.int64), vidx)
    mx = tl.max(vmax, 0)
    i = tl.min(tl.where(vmax == mx, vidx, BIGI), 0)
    tl.store(pmax + row * NCH + ch, mx)
    tl.store(pidx + row * NCH + ch, i)


@triton.jit
def _comb_kernel(pmax, pidx, y, out_ptr, V, NCH, NB: tl.constexpr, ONEHOT: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, NB)
    ok = k < NCH
    m = tl.load(pmax + row * NCH + k, mask=ok, other=IMINI)
    i = tl.load(pidx + row * NCH + k, mask=ok, other=BIGI)
    mm = tl.max(m, 0)
    idx = tl.min(tl.where(ok & (m == mm), i, BIGI), 0)
    tl.store(y + row, idx)
    if ONEHOT:
        tl.store(out_ptr + row.to(tl.int64) * V + idx, 0.0)


class Head:
    """The LM-head GEMM output and the constants that turn it into logits (segs.post's arguments)."""

    def __init__(self, c, m, fmsh, mant, e):
        self.c, self.m, self.fmsh, self.mant, self.e = c, m, fmsh, mant, e
        self.M, self.V = mant.shape[0], c.shape[1]

    def logits(self):
        """The int64 logits [M, V] (eager segs.post; only for hash logging)."""
        from bitexact.int8.segs import post
        from bitexact.intops import join_limbs
        return post(join_limbs(self.c, self.M, 2), self.m, self.fmsh, self.mant, self.e, False)[0]


def _plan(M, V, block):
    nch = max(1, min(triton.cdiv(V, block), triton.cdiv(264, max(M, 1))))     # ~2 programs per SM over all rows
    ch = triton.cdiv(triton.cdiv(V, nch), block) * block
    return ch, triton.cdiv(V, ch)


def head(h, mode, block=1024):
    """mode 'float': (float32 logits * 2^-16, first argmax); 'onehot': (-inf row with 0 at the argmax, argmax);
    'none': (None, argmax)."""
    M, V = h.M, h.V
    dev = h.c.device
    ch, nch = _plan(M, V, block)
    out = torch.empty(M, V, dtype=torch.float32, device=dev) if mode != "none" else h.c
    pmax = torch.empty(M, nch, dtype=torch.long, device=dev)
    pidx = torch.empty_like(pmax)
    y = torch.empty(M, dtype=torch.long, device=dev)
    if M:
        _head_kernel[(M, nch)](h.c, h.m, h.fmsh, h.mant, h.e, out, pmax, pidx, M, V, ch, nch,
                               MODE={"none": 0, "float": 1, "onehot": 2}[mode], BLOCK=block, num_warps=8)
        _comb_kernel[(M,)](pmax, pidx, y, out, V, nch, NB=triton.next_power_of_2(nch), ONEHOT=mode == "onehot")
    return (None if mode == "none" else out), y


def row_state(keys, temps):
    """[(seed, t)], [(tn, td)] -> int64 [n, 6]: PCG state hi, lo, inc hi, lo, tn, td. Greedy rows (tn = 0)
    need no seed state."""
    st = torch.zeros(len(keys), 6, dtype=torch.long)
    live = [r for r, (tn, _) in enumerate(temps) if tn]
    if live:
        st[live, :4] = gpu_pcg.seed_state([keys[r] for r in live])
    st[:, 4] = torch.tensor([tn for tn, _ in temps], dtype=torch.long)
    st[:, 5] = torch.tensor([td for _, td in temps], dtype=torch.long)
    return st


class _Ring:
    """Pinned host staging for the per-step row state: a ring, so a buffer is rewritten only after its copy to the
    device has run (event); no pinned allocation per step."""
    N = 4

    def __init__(self):
        self.buf, self.ev, self.i = {}, {}, 0

    def put(self, a, dev):
        """numpy int64 [n, k] -> device int64 [n, k] (async copy)."""
        n, k = a.shape
        key = (k, dev)
        if key not in self.buf:
            self.buf[key] = [torch.empty(4096, k, dtype=torch.long).pin_memory() for _ in range(self.N)]
            self.ev[key] = [None] * self.N
        self.i = (self.i + 1) % self.N
        b, e = self.buf[key][self.i], self.ev[key][self.i]
        if e is not None:
            e.synchronize()
        b[:n].numpy()[:] = a
        d = b[:n].to(dev, non_blocking=True)
        e = self.ev[key][self.i] = torch.cuda.Event()
        e.record()
        return d


_RING = _Ring()


def noisy(h, st, lnt, block=1024, out=None):
    """Gumbel-max token per row: st from row_state (CPU int64 [n, 6]) or int64 [n, 4] seed, t, tn, td (numpy, seeds
    and t below 2^32: the PCG state is derived in the kernel). Same tokens as sampling.sample(li, tn, td,
    gumbel(raw, ...)). out: vLLM's float logits [M, V]; if given it is left as -inf rows with 0 at the token."""
    M, V = h.M, h.V
    dev = h.c.device
    ch, nch = _plan(M, V, block)
    st = _RING.put(st.numpy() if torch.is_tensor(st) else st, dev)
    pmax = torch.empty(M, nch, dtype=torch.long, device=dev)
    pidx = torch.empty_like(pmax)
    y = torch.empty(M, dtype=torch.long, device=dev)
    if M:
        fill = out is not None
        _noise_kernel[(M, nch)](h.c, h.m, h.fmsh, h.mant, h.e, st, gpu_pcg._tables(dev), lnt, pmax, pidx, M, V,
                                ch, nch, out if fill else h.c, BLOCK=block, SEEDED=st.shape[1] == 4, FILL=fill,
                                num_warps=8)
        _comb_kernel[(M,)](pmax, pidx, y, out if fill else h.c, V, nch, NB=triton.next_power_of_2(nch),
                           ONEHOT=fill)
    return y
