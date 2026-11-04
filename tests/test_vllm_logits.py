"""vllm_plugin.logits.lm_logits (static bucketed compile) equals IntLlama.logits bit for bit (skip without CUDA)."""

import subprocess
import sys

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


class _Fake:
    """The few IntLlama members logits() touches, with a random int8 lm_head of a small vocabulary."""

    def __init__(self, D=2048, V=4096, nl=2):
        from bitexact.int8.segs import Segs
        g = torch.Generator(device="cuda").manual_seed(0)
        self.nl, self.dev = nl, torch.device("cuda")
        self.sg = Segs(nl, True, True)
        self.gf = torch.randint(1, 1 << 13, (D,), device="cuda", generator=g)
        w = torch.randint(-127, 128, (V, D), device="cuda", generator=g, dtype=torch.int8)
        mm = torch.randint(1 << 20, 1 << 22, (V,), device="cuda", generator=g)
        self.lm = (w, mm, torch.tensor(30, device="cuda"))
        self.err = torch.zeros((), dtype=torch.bool, device="cuda")

    def _flag(self, cond):
        self.err |= cond

    def check(self):
        if bool(self.err):
            raise OverflowError("integer bound violated")

    def _gemm(self, a, w):
        return torch._int_mm(a, w.t())

    def logits(self, x):
        from bitexact.int8.model import IntLlama
        return IntLlama._linear(self, self.sg.pre_norm(x, self.gf), self.lm)


def test_lm_logits_equal_all_row_counts():
    from bitexact.vllm_plugin.logits import lm_logits
    m = _Fake()
    g = torch.Generator(device="cuda").manual_seed(1)
    for M in (1, 3, 8, 21, 64, 130, 255, 256, 257, 511, 512, 513, 700):   # bucket edges; > 512 takes m.logits
        x = torch.randint(-(1 << 24), 1 << 24, (M, 2048), device="cuda", generator=g)
        assert torch.equal(lm_logits(m, x), m.logits(x))
    assert not bool(m.err)


def test_lm_logits_flag_still_raised():
    from bitexact.vllm_plugin.logits import lm_logits
    m = _Fake()
    m.lm = (m.lm[0], m.lm[1], torch.tensor(0, device="cuda"))     # no down-shift: t >= 2^38, the flag must be set
    x = torch.randint(1 << 22, 1 << 24, (5, 2048), device="cuda")
    lm_logits(m, x)
    assert bool(m.err)


def test_overflow_check_passes_without_flag():
    from bitexact.vllm_plugin.logits import OverflowCheck
    m = _Fake()
    OverflowCheck(m)()
    torch.cuda.synchronize()


def test_overflow_check_sync_mode_raises(monkeypatch):
    from bitexact.vllm_plugin.logits import OverflowCheck
    monkeypatch.setenv("BITEXACT_CHECK", "sync")
    m = _Fake()
    chk = OverflowCheck(m)
    chk()
    m.err.fill_(True)
    with pytest.raises(OverflowError):
        chk()


_LAST_STEP = """
import torch
from tests.test_vllm_logits import _Fake
from bitexact.vllm_plugin.logits import OverflowCheck, lm_logits
m = _Fake()
chk = OverflowCheck(m)
x = torch.randint(-(1 << 24), 1 << 24, (4, 2048), device="cuda")
for step in range(3):
    if step == 2:                    # the last step of the run overflows
        m.err.fill_(True)
    y = lm_logits(m, x)
    chk()
    tokens = y.argmax(-1)            # the "sampler"
    print("sampled", step, tokens.cpu().tolist(), flush=True)     # the copy to the host
"""


def test_overflow_on_last_step_never_returns_tokens():
    """An overflow flagged on the last step must fail before that step's tokens reach the host."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([root, os.path.join(root, "src"), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-c", _LAST_STEP], capture_output=True, text=True, env=env, cwd=root)
    assert r.returncode != 0
    assert r.stdout.count("sampled") == 2, r.stdout + r.stderr          # steps 0 and 1 only
    assert "device-side assert" in r.stderr or "integer bound violated" in r.stderr, r.stderr[-500:]
