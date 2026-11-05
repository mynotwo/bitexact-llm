"""Integer log p of given tokens straight from the LM-head GEMM output (tri_sample.Head), without int64 logits.

rl.int_logp on the stored int64 logits [n, V] costs ~15 passes over n x V x 8 bytes (sub, clamp, shifts, two table
gathers, adds, sum). Here two Triton passes read the GEMM output (limbs, int32) and everything else is on [n]:

  _stat_kernel  per (row, column chunk): max of the row's logits and the target token's logit
  _sum_kernel   per (row, column chunk): sum of e^(l - max) from rl.lse_table (the same interpolation as int_logp)

The sums are integer (< 2^62 in all, so exact in any order); the tail (ln_q of the sum, target term) is int_logp's
code on [n]. Logits come from tri_mlp._y (segs.post), as in tri_sample. Same integers as rl.int_logp(head.logits(), ...).
"""

import torch
import triton
import triton.language as tl

from bitexact.vllm_plugin.tri_mlp import _y
from bitexact.vllm_plugin.tri_sample import IMINI, Head, _plan

FRL = tl.constexpr(16)
LSE_MIN = tl.constexpr(-40)
LSE_STEP = tl.constexpr(12)        # table step 2^-12


@triton.jit
def _stat_kernel(c_ptr, m_ptr, fmsh_ptr, mant_ptr, e_ptr, tok_ptr, pmax, ptg, M, V, CH, NCH, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    ch = tl.program_id(1)
    fmsh = tl.load(fmsh_ptr)
    mant = tl.load(mant_ptr + row)
    e = tl.load(e_ptr + row)
    tok = tl.load(tok_ptr + row)
    lane = tl.arange(0, BLOCK)
    vmax = tl.full([BLOCK], IMINI, tl.int64)
    tg = tl.zeros([BLOCK], tl.int64)
    for c0 in range(ch * CH, tl.minimum(ch * CH + CH, V), BLOCK):
        col = c0 + lane
        ok = col < V
        _, x = _y(c_ptr, m_ptr, col, ok, row, M, V, fmsh, mant, e)
        vmax = tl.maximum(vmax, tl.where(ok, x, IMINI))
        tg += tl.where(ok & (col == tok), x, 0)
    tl.store(pmax + row * NCH + ch, tl.max(vmax, 0))
    tl.store(ptg + row * NCH + ch, tl.sum(tg, 0))


@triton.jit
def _sum_kernel(c_ptr, m_ptr, fmsh_ptr, mant_ptr, e_ptr, mx_ptr, tab_ptr, psum, M, V, CH, NCH, LAST,
                BLOCK: tl.constexpr):
    row = tl.program_id(0)
    ch = tl.program_id(1)
    fmsh = tl.load(fmsh_ptr)
    mant = tl.load(mant_ptr + row)
    e = tl.load(e_ptr + row)
    mx = tl.load(mx_ptr + row)
    lane = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.int64)
    for c0 in range(ch * CH, tl.minimum(ch * CH + CH, V), BLOCK):
        col = c0 + lane
        ok = col < V
        _, x = _y(c_ptr, m_ptr, col, ok, row, M, V, fmsh, mant, e)
        zc = tl.maximum(x - mx, LSE_MIN << FRL)
        zi = zc >> (FRL - LSE_STEP)
        r = zc - (zi << (FRL - LSE_STEP))
        i0 = zi - (LSE_MIN << LSE_STEP)
        a = tl.load(tab_ptr + i0, mask=ok, other=0)
        b = tl.load(tab_ptr + tl.minimum(i0 + 1, LAST), mask=ok, other=0)
        acc += tl.where(ok, a + (((b - a) * r + (1 << (FRL - LSE_STEP - 1))) >> (FRL - LSE_STEP)), 0)
    tl.store(psum + row * NCH + ch, tl.sum(acc, 0))


def logp(h: Head, tok, lnt):
    """Integer log p of tok [M] (int64) under the logits rows of Head h; float64 at 2^-40, equal to
    rl.int_logp(h.logits(), tok, lnt)."""
    from bitexact.sampling import FL, LN2Q, ln_q
    from bitexact.vllm_plugin.rl import LSE_FE, lse_table
    M, V = h.M, h.V
    dev = h.c.device
    tab = lse_table(dev)
    block = 1024
    ch, nch = _plan(M, V, block)
    pmax = torch.empty(M, nch, dtype=torch.long, device=dev)
    ptg = torch.empty_like(pmax)
    psum = torch.empty_like(pmax)
    args = (h.c, h.m, h.fmsh, h.mant, h.e)
    _stat_kernel[(M, nch)](*args, tok, pmax, ptg, M, V, ch, nch, BLOCK=block, num_warps=8)
    mx = pmax.amax(1)
    _sum_kernel[(M, nch)](*args, mx, tab, psum, M, V, ch, nch, tab.shape[0] - 1, BLOCK=block, num_warps=8)
    S = psum.sum(1)
    lse = ln_q(S, lnt) - LSE_FE * LN2Q
    zt = (ptg.sum(1) - mx).clamp(min=-(1 << 36))
    return ((zt << (FL - 16)) - lse).double() / 2.0 ** FL
