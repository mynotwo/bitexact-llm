"""numpy's PCG64 raw stream on the GPU, bit for bit.

The Gumbel noise for (seed, t) (bitexact.sampling) is np.random.default_rng([SEED_BASE, seed, t]).bit_generator
.random_raw(V): V outputs of PCG64 (128-bit LCG, XSL-RR output). Output i (i = 1..V) is
  state_i = a^i * s0 + (a^(i-1) + ... + a + 1) * inc   (mod 2^128),   out_i = rotr64(hi ^ lo, state_i >> 122)
with s0, inc the generator's state after numpy's SeedSequence seeding (computed on the CPU, a few
microseconds per row). A_i = a^i and G_i = sum_{j<i} a^j are fixed tables (built once with Python
integers), so every output is independent: one Triton program per (row, block of i), 128-bit products
from 32-bit pieces in uint64. tests/test_vllm_kernels.py compares against numpy.
"""

import numpy as np
import torch
import triton
import triton.language as tl

from bitexact.llama import V
from bitexact.sampling import SEED_BASE

MULT = (2549297995355413924 << 64) + 4865540595714422341        # PCG_DEFAULT_MULTIPLIER_128
MASK128 = (1 << 128) - 1
_TABLES = {}


def _split64(x):
    """Python int < 2^64 -> int64 with the same bits."""
    return x - (1 << 64) if x >= 1 << 63 else x


def _tables(device):
    key = str(device)
    if key not in _TABLES:
        ah, al, gh, gl = (np.empty(V, dtype=np.int64) for _ in range(4))
        a, g = 1, 0
        for i in range(V):                       # entry i is for output i + 1
            g = (g + a) & MASK128
            a = (a * MULT) & MASK128
            ah[i], al[i] = _split64(a >> 64), _split64(a & (2**64 - 1))
            gh[i], gl[i] = _split64(g >> 64), _split64(g & (2**64 - 1))
        _TABLES[key] = torch.from_numpy(np.stack([ah, al, gh, gl])).to(device)
    return _TABLES[key]


def seed_state(keys):
    """[(seed, t)] -> int64 [n, 4]: state hi, lo, inc hi, lo after numpy's seeding."""
    out = np.empty((len(keys), 4), dtype=np.int64)
    for r, (s, t) in enumerate(keys):
        st = np.random.PCG64(np.random.SeedSequence([SEED_BASE, int(s), int(t)])).state["state"]
        x, inc = st["state"], st["inc"]
        out[r] = [_split64(x >> 64), _split64(x & (2**64 - 1)), _split64(inc >> 64), _split64(inc & (2**64 - 1))]
    return torch.from_numpy(out)


@triton.jit
def _mulhi(x, y):
    x1 = x >> 32
    y1 = y >> 32
    x0 = x - (x1 << 32)                          # low halves without a 32-bit mask constant
    y0 = y - (y1 << 32)
    p00 = x0 * y0
    p01 = x0 * y1
    p10 = x1 * y0
    p11 = x1 * y1
    mid = (p00 >> 32) + (p01 - ((p01 >> 32) << 32)) + (p10 - ((p10 >> 32) << 32))
    return p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32)


@triton.jit
def _pcg_kernel(st_ptr, tab_ptr, out_ptr, n_out, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = i < n_out
    sh = tl.load(st_ptr + r * 4).to(tl.uint64, bitcast=True)
    sl = tl.load(st_ptr + r * 4 + 1).to(tl.uint64, bitcast=True)
    ch = tl.load(st_ptr + r * 4 + 2).to(tl.uint64, bitcast=True)
    cl = tl.load(st_ptr + r * 4 + 3).to(tl.uint64, bitcast=True)
    ah = tl.load(tab_ptr + i, mask=ok, other=0).to(tl.uint64, bitcast=True)
    al = tl.load(tab_ptr + n_out + i, mask=ok, other=0).to(tl.uint64, bitcast=True)
    gh = tl.load(tab_ptr + 2 * n_out + i, mask=ok, other=0).to(tl.uint64, bitcast=True)
    gl = tl.load(tab_ptr + 3 * n_out + i, mask=ok, other=0).to(tl.uint64, bitcast=True)
    lo1 = al * sl
    hi1 = _mulhi(al, sl) + al * sh + ah * sl
    lo2 = gl * cl
    hi2 = _mulhi(gl, cl) + gl * ch + gh * cl
    lo = lo1 + lo2
    carry = ((lo1 & lo2) | ((lo1 | lo2) & ~lo)) >> 63              # unsigned carry, without a compare
    hi = hi1 + hi2 + carry
    v = hi ^ lo
    rot = hi >> 58
    outv = (v >> rot) | (v << ((64 - rot) & 63))
    tl.store(out_ptr + r * n_out + i, outv.to(tl.int64, bitcast=True), mask=ok)


def raw_from_state(st, device):
    """st int64 [n, 4] (seed_state) -> int64 [n, V] raw words on device."""
    tab = _tables(device)
    st = st.pin_memory().to(device, non_blocking=True)            # pinned: the copy never waits for the GPU
    out = torch.empty(st.shape[0], V, dtype=torch.long, device=device)
    if st.shape[0]:
        BLOCK = 1024
        with torch.cuda.device(out.device):
            _pcg_kernel[(st.shape[0], triton.cdiv(V, BLOCK))](st, tab, out, V, BLOCK=BLOCK)
    return out


def raw_rows(keys, device):
    """Same words as bitexact.sampling.raw_rows(keys), generated on the GPU."""
    return raw_from_state(seed_state(keys), device)
