"""The lm_head step of compute_logits (final RMSNorm + quantization, int8 GEMM, limb join + rescale) with static,
bucketed shapes, compiled by the plugin itself.

Why not IntLlama.logits: its epilogue (int8/segs.py Segs.post) is one torch.compile(dynamic=None) function that
serves every GEMM of the model, so after the first size change (vLLM's warm-up and prefill steps change the row
count) it is recompiled with symbolic row count and width. Its overflow flag is amax over the whole [M, N] output,
a reduction to one element. In the static compile Inductor splits that reduction over many CTAs; in the symbolic
recompile it emits a kernel with grid (1, 1, 1) that loops over all M x N elements in one CTA (experiments/
speed_decode/rootcause.py: every recompile order and size we tried gives it, from 3 to 512 rows, so it does not
depend on the hint size). For the lm_head output (M x 128,256 int64) that is 8 ms per step at M = 64 on an H100.
Here the segments are compiled with dynamic=False for a few row buckets, so the reduction is split as in the
static compile.

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
    return post(join_limbs(c, mant.shape[0], nl), m, fmsh, mant, e, whole=True)


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


class OverflowCheck:
    """IntLlama.check() for the plugin, called once per compute_logits, before the step's tokens are sampled.

    Default ("assert"): a device-side assert on the overflow flag is enqueued on the stream behind the step's
    kernels and ahead of the sampler. If any bound of the step was violated the assert traps, every later kernel of
    the stream (the sampler, the copy of the sampled tokens to the host) never runs, and the process fails with a
    CUDA "device-side assert" error: no token of an overflowed step is ever returned, including on the last step of
    a run, and the host never waits for the GPU (vLLM's async scheduling keeps overlapping).
    BITEXACT_CHECK=sync: IntLlama.check(), which reads the flag with bool() and raises OverflowError; it blocks
    the host until the step's kernels have finished (slower; the same guarantee with a Python exception)."""

    def __init__(self, m):
        self.m = m
        self.mode = os.environ.get("BITEXACT_CHECK", "assert")
        assert self.mode in ("assert", "sync"), self.mode

    def __call__(self):
        if self.mode == "sync":
            return self.m.check()
        torch._assert_async(~self.m.err, "bitexact: integer bound violated (int8/model.py)")
