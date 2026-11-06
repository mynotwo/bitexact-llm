"""Fused exact-mode linear (exact_gemm.FusedLin) against exact.model.Lin + segs.lin_y on real constants
(BITEXACT_EXACT_CONST; skip without CUDA, vLLM or the constants)."""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]
pytest.importorskip("vllm")


def test_fused_lin_matches_lin_y(const):
    kind, c = const
    if kind != "exact":
        pytest.skip("exact constants only")
    from bitexact.exact.model import Lin
    from bitexact.exact.segs import lin_y
    from bitexact.vllm_plugin.exact_gemm import FusedLin
    g = torch.Generator(device="cuda").manual_seed(0)
    for name in ("0.wqkv", "0.wo", "0.wgu", "0.wd", "lm"):
        lin = Lin(c, name, "cuda", "triton", False)
        fl = FusedLin(lin)
        K = lin.limbs.shape[2]
        for M in (1, 3, 8, 9, 16, 17, 20, 21, 31, 32, 33, 40, 41, 64, 65, 72, 73, 100, 105, 120, 127, 128, 129, 200, 1000):
            for lim, shr in ((1 << 22, (-30, -20)), (1 << 12, (-45, -8)), (3, (-70, 10))):
                q = torch.randint(-lim, lim + 1, (M, K), device="cuda", generator=g).to(torch.int32)
                sh = torch.randint(shr[0], shr[1], (M,), device="cuda", generator=g)
                ref, rflag = lin_y(*lin(q), sh, True)
                flag = torch.zeros(1, dtype=torch.int32, device="cuda")
                y = fl(q, sh, flag)
                assert torch.equal(y, ref), (name, M, lim)
                assert bool(flag[0]) == bool(rflag), (name, M, lim)
