"""The lm_head step of compute_logits (final RMSNorm + quantization, int8 GEMM, limb join + rescale) with static,
bucketed shapes, compiled by the plugin itself.

Why not IntLlama.logits: its epilogue (int8/segs.py Segs.post) is one torch.compile(dynamic=None) function that
serves every GEMM of the model, so after the first size change it is recompiled with symbolic row and width. Its
overflow flag is amax over the whole [M, N] output, a reduction to one element. Inductor picks the number of CTAs
of such a reduction at compile time from the size *hint* of the trace, not from the run-time size: when the hint is
at most 8192 elements (a recompile triggered by a small shape, or a kernel read back from the on-disk FX-graph cache
that was compiled that way) it emits one CTA that loops over the whole input. The same kernel then reduces the
lm_head output (M x 128,256 int64), 8 ms per step at M = 64 on an H100. Here the segments are compiled with
dynamic=False for a handful of row buckets, so the hint is the real size and the reduction is split across the GPU.

The integers are the same: same integer ops, rows are independent, and the overflow flag is a maximum. Rows added to
reach a bucket repeat the last real row, so they cannot raise the flag, and they are dropped from the result.
"""

import os

import torch

from bitexact.int8.segs import post, pre_norm
from bitexact.intops import join_limbs, stack_limbs

BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
_FN = {}


def _pre(x, gamma, nl: int):
    q, mant, e = pre_norm(x, gamma, nl)
    return stack_limbs(q, nl), mant, e


def _post(c, m, fmsh, mant, e, nl: int):
    return post(join_limbs(c, mant.shape[0], nl), m, fmsh, mant, e)


def _fns():
    if not _FN:
        on = os.environ.get("BITEXACT_COMPILE", "1") == "1"
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        wrap = (lambda f: torch.compile(f, dynamic=False)) if on else (lambda f: f)
        _FN["pre"], _FN["post"] = wrap(_pre), wrap(_post)
    return _FN["pre"], _FN["post"]


def lm_logits(m, x):
    """Final residual rows x int64 [M, D] -> logits int64 [M, V] at FR, equal to m.logits(x)."""
    M = x.shape[0]
    B = next((b for b in BUCKETS if b >= M), None)
    if B is None or not x.is_cuda:
        return m.logits(x)
    if B != M:
        x = torch.cat([x, x[-1:].expand(B - M, -1)])
    pre, post_ = _fns()
    w, mm, fmsh = m.lm
    q, mant, e = pre(x, m.gf, m.nl)
    y, flag = post_(torch._int_mm(q, w.t()), mm, fmsh, mant, e, m.nl)
    m._flag(flag)
    return y[:M]


class DeferredCheck:
    """IntLlama.check() without the host sync. check() reads the on-device overflow flag with bool(), which blocks the
    host until the whole step's kernels have finished, so vLLM's async scheduling cannot prepare the next step while
    the GPU runs. Here the flag is copied to pinned memory with an event and read at the next call, so an overflow
    raises one step later (the flag is sticky, the run is aborted either way). Nothing a token depends on changes."""

    def __init__(self, m):
        self.m = m
        self.on = os.environ.get("BITEXACT_DEFER_CHECK", "1") == "1"
        self.host = torch.zeros((), dtype=torch.bool, device="cpu").pin_memory() if self.on else None
        self.event = torch.cuda.Event() if self.on else None
        self.pending = False

    def __call__(self):
        if not self.on:
            return self.m.check()
        if self.pending:
            self.event.synchronize()
            if bool(self.host):
                raise OverflowError("integer bound violated (int8/model.py)")
        self.host.copy_(self.m.err, non_blocking=True)
        self.event.record()
        self.pending = True
