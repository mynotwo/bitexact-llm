"""vLLM plugin kernels (skip without CUDA or vLLM; written for the Triton that ships with vLLM 0.32): fused
paged attention against the unfused int8 softmax with exact int64 products, and GPU PCG64 words against
numpy."""

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]
pytest.importorskip("vllm")


def test_fused_paged_attention_exact():
    from bitexact.int8.consts import exp_table
    from bitexact.int8.segs import softmax
    from bitexact.intops import LIM
    from bitexact.vllm_plugin import tri_paged
    g = torch.Generator(device="cuda").manual_seed(0)
    lut = exp_table().cuda()
    nb, bs, HKV, HD = 96, 16, 8, 64
    for lim in (LIM[3], 1 << 13, 1 << 10):
        c = torch.randint(-lim, lim + 1, (nb, HKV, bs, 2 * HD), device="cuda", generator=g).to(torch.int32)
        for R, M, P in [(3, 4, 600), (5, 4, 3000), (1, 1024, 2000), (2, 64, 600), (4, 20, 100)]:
            bt = torch.randint(1, nb, (R, 200), device="cuda", generator=g).to(torch.int32)
            qpos = torch.randint(0, P, (R, M), device="cuda", generator=g)
            q = torch.randint(-lim, lim + 1, (R, HKV, M, HD), device="cuda", generator=g)
            S = int(qpos.max()) + 1
            s = torch.arange(S, device="cuda")
            kv = c[bt.long()[:, s // bs], :, (s % bs)[None].expand(R, -1)].long()     # [R, S, HKV, 2 HD]
            K, V = kv[..., :HD].permute(0, 2, 1, 3).cpu(), kv[..., HD:].permute(0, 2, 1, 3).cpu()
            raw = torch.einsum("rhmd,rhsd->rhms", q.cpu(), K)                     # exact int64 on CPU
            p, den = softmax(raw, qpos.cpu()[:, None, :, None], lut.cpu())
            ref_num = torch.einsum("rhms,rhsd->rhmd", p, V)
            for ns in (None, 1, 5, 64):
                num, den2 = tri_paged.attend(q, qpos, c, bt, tri_paged.weight_table(lut), nsplit=ns)
                assert torch.equal(num.cpu(), ref_num) and torch.equal(den2.cpu(), den), (R, M, ns)


def test_gpu_pcg_matches_numpy():
    from bitexact import sampling
    from bitexact.vllm_plugin import gpu_pcg
    rng = np.random.default_rng(5)
    keys = [(int(s), int(t)) for s, t in zip(rng.integers(0, 2**31, 64), rng.integers(0, 4096, 64))]
    keys += [(0, 0), (1, 0), (2**32 - 1, 7), (12, 47)]
    got = gpu_pcg.raw_rows(keys, "cuda").cpu()
    assert torch.equal(got, sampling.raw_rows(keys))
