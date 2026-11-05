"""Exact-mode paged attention kernel (tri_exact_attn) against exact/ops.softmax_w with exact int64 products
(skip without CUDA or vLLM)."""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]
pytest.importorskip("vllm")


def _exp2():
    f = torch.arange((1 << 14) + 1, dtype=torch.float64) / (1 << 14)
    return torch.round(torch.pow(2.0, -f) * 2.0 ** 60).long()


@pytest.mark.parametrize("HD", [64, 128])
def test_exact_paged_attention(HD):
    from bitexact.exact import ops
    from bitexact.exact.segs import FK, FP
    from bitexact.vllm_plugin import tri_exact_attn, tri_paged
    g = torch.Generator(device="cuda").manual_seed(0)
    lut = _exp2().cuda()
    nb, bs, HKV = 96, 16, 8
    l2e = ops.L2E28_HD[HD]
    for lim in (8355711, 1 << 13, 1 << 10):
        c = torch.randint(-lim, lim + 1, (nb, HKV, bs, 2 * HD), device="cuda", generator=g).to(torch.int32)
        for R, M, P in [(3, 4, 600), (5, 4, 3000), (6, 1, 900), (2, 3, 50), (7, 4, 5)]:
            bt = torch.randint(1, nb, (R, 200), device="cuda", generator=g).to(torch.int32)
            qpos = torch.randint(0, P, (R, M), device="cuda", generator=g)
            q = torch.randint(-(1 << 22), (1 << 22) + 1, (R, HKV, M, HD), device="cuda", generator=g)
            sh = torch.randint(-40, -10, (R, HKV, M), device="cuda", generator=g)
            S = int(qpos.max()) + 1
            s = torch.arange(S, device="cuda")
            kv = c[bt.long()[:, s // bs], :, (s % bs)[None].expand(R, -1)].long()
            K, V = kv[..., :HD].permute(0, 2, 1, 3).cpu(), kv[..., HD:].permute(0, 2, 1, 3).cpu()
            raw = torch.einsum("rhmd,rhsd->rhms", q.cpu(), K)
            ok = torch.arange(S)[None, None, None] <= qpos.cpu()[:, None, :, None]
            p, den = ops.softmax_w(raw, ok, sh.cpu()[..., None], FK, lut.cpu(), FP, l2e)
            ref_num = torch.einsum("rhms,rhsd->rhmd", p, V)
            rmap = torch.arange(R, device="cuda")
            for ns in (None, 1, 5, 64):
                num, den2 = tri_exact_attn.attend(q.to(torch.int32), sh, qpos, tri_paged.to_limbs(c), bt, rmap, lut,
                                                  l2e, nsplit=ns)
                assert torch.equal(num.cpu(), ref_num) and torch.equal(den2.cpu(), den), (R, M, ns, lim)
