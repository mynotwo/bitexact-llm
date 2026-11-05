"""Integer logits -> vLLM's float logits and the greedy token, in one pass over the logits.

compute_logits used to make the float copy (li.float() * 2^-16: two kernels) and the greedy sampler the
first argmax (intops.first_argmax: amax, compare, where over a full index tensor, amin: five kernels and
two extra int64 [M, V] tensors). Here one Triton kernel reads each int64 row once, writes the output row
and a per-chunk (max, first index); a second one combines the chunks of a row.

Same results:
  - float: int64 -> float32 rounds to nearest even in both (cvt.rn.f32.s64); * 2^-16 is exact (no
    underflow: |l| >= 1 or l = 0), so the bits equal li.float() * 2.0 ** -16.
  - token: the first index of the row maximum. Within a lane, a column is taken only when strictly larger
    (columns increase along the loop), so each lane keeps its first maximum; across lanes and then across
    chunks the smallest index among those holding the maximum wins. Integer max and min: exact, no order
    dependence. A lane that has seen no valid column is replaced by the first valid one (so a row of
    int64 minima still gives index 0, as first_argmax does).

Output mode (what vLLM's sampler sees):
  'float'   li * scale, the real float logits (requests that read them: logprobs, penalties, ...).
  'onehot'  -inf with 0 at the argmax: vLLM's greedy argmax returns the integer token. Used when no
            request reads the float logits, which includes the batch where the sampler's apply() is
            skipped altogether (see model.compute_logits); a float row would let int64 ties survive as
            float32 ties, and an uninitialised row would return whatever memory held.
  'none'    nothing is written (the caller only needs the token).
"""

import torch
import triton
import triton.language as tl

IMIN = -(1 << 63)
BIG = 1 << 62                   # > any column index


@triton.jit
def _part_kernel(li, out, pmax, pidx, V, CH, NCH, SCALE: tl.constexpr, MODE: tl.constexpr,
                 BLOCK: tl.constexpr):
    row = tl.program_id(0)
    ch = tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    vmax = tl.full([BLOCK], -9223372036854775808, tl.int64)
    vidx = tl.full([BLOCK], 4611686018427387904, tl.int64)
    base = row.to(tl.int64) * V
    for c0 in range(ch * CH, tl.minimum(ch * CH + CH, V), BLOCK):
        col = c0 + lane
        ok = col < V
        x = tl.load(li + base + col, mask=ok, other=-9223372036854775808)
        if MODE == 1:
            tl.store(out + base + col, x.to(tl.float32) * SCALE, mask=ok)
        elif MODE == 2:
            tl.store(out + base + col, tl.full([BLOCK], float("-inf"), tl.float32), mask=ok)
        take = ok & ((x > vmax) | (vidx == 4611686018427387904))
        vmax = tl.where(take, x, vmax)
        vidx = tl.where(take, col.to(tl.int64), vidx)
    m = tl.max(vmax, 0)
    i = tl.min(tl.where(vmax == m, vidx, 4611686018427387904), 0)
    tl.store(pmax + row * NCH + ch, m)
    tl.store(pidx + row * NCH + ch, i)


@triton.jit
def _comb_kernel(pmax, pidx, y, out, V, NCH, MODE: tl.constexpr, NB: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, NB)
    ok = k < NCH
    m = tl.load(pmax + row * NCH + k, mask=ok, other=-9223372036854775808)
    i = tl.load(pidx + row * NCH + k, mask=ok, other=4611686018427387904)
    mm = tl.max(m, 0)
    idx = tl.min(tl.where(ok & (m == mm), i, 4611686018427387904), 0)
    tl.store(y + row, idx)
    if MODE == 2:                                # the 0 of the one-hot row
        tl.store(out + row.to(tl.int64) * V + idx, 0.0)


MODES = {"none": 0, "float": 1, "onehot": 2}


def argmax_float(li, scale=2.0 ** -16, mode="float", nch=None, block=1024):
    """li int64 [M, V] (contiguous) -> (float32 row or None, int64 [M] first argmax), the row written in
    `mode` (see the module docstring). No host sync."""
    M, V = li.shape
    li = li.contiguous()
    if nch is None:                                  # ~2 programs per SM over all rows
        nch = max(1, min(triton.cdiv(V, block), triton.cdiv(264, max(M, 1))))
    ch = triton.cdiv(triton.cdiv(V, nch), block) * block
    nch = triton.cdiv(V, ch)
    md = MODES[mode]
    out = torch.empty(M, V, dtype=torch.float32, device=li.device) if md else li
    pmax = torch.empty(M, nch, dtype=torch.long, device=li.device)
    pidx = torch.empty_like(pmax)
    y = torch.empty(M, dtype=torch.long, device=li.device)
    if M:
        _part_kernel[(M, nch)](li, out, pmax, pidx, V, ch, nch, SCALE=scale, MODE=md, BLOCK=block,
                               num_warps=8)
        _comb_kernel[(M,)](pmax, pidx, y, out, V, nch, MODE=md, NB=triton.next_power_of_2(nch))
    return (out if md else None), y


def first_argmax(v):
    """intops.first_argmax for int64 [M, V] on CUDA (same indices)."""
    return argmax_float(v, mode="none")[1]
