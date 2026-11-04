"""tri_prefill.attend (token-major, tiles of BM // G tokens, both sweeps in one program) against
tri_paged.attend on the gathered g-major layout (itself compared with the exact int64 softmax in
test_vllm_kernels.py). Needs CUDA and vLLM's Triton."""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]
pytest.importorskip("triton")


@pytest.mark.parametrize("bm", [64, 128])
def test_prefill_attend_matches_paged(bm):
    from bitexact.int8.consts import exp_table
    from bitexact.intops import LIM
    from bitexact.vllm_plugin import tri_paged, tri_prefill
    HD, HKV, G, nb, bs = 128, 8, 4, 160, 16
    g = torch.Generator(device="cuda").manual_seed(3)
    ptab = tri_paged.weight_table(exp_table().cuda())
    for lim in (LIM[3], 1 << 13, 1 << 10):
        c = tri_paged.to_limbs(torch.randint(-lim, lim + 1, (nb, HKV, bs, 2 * HD), device="cuda", generator=g)
                               .to(torch.int32))
        # (qlen, seq_len): fresh prompts, a later chunk, one-token chunks, odd lengths
        reqs = [(37, 37), (64, 300), (1, 20), (130, 130), (5, 700), (16, 16), (200, 900)]
        bt = torch.randint(1, nb, (len(reqs), 64), device="cuda", generator=g).to(torch.int64)
        tot = sum(q for q, _ in reqs)
        N = tot + 3                           # trailing rows without attention
        q = torch.randint(-lim, lim + 1, (N, HKV, G, HD), device="cuda", generator=g)
        pos = torch.zeros(N, dtype=torch.long, device="cuda")
        pre, t0 = [], 0
        for r, (ql, sl) in enumerate(reqs):
            pos[t0:t0 + ql] = torch.arange(sl - ql, sl, device="cuda")
            pre.append((r, t0, ql, sl))
            t0 += ql
        tiles = tri_prefill.tile_table(pre, G, bm).cuda()
        num = torch.zeros(N, HKV, G, HD, dtype=torch.long, device="cuda")
        den = torch.ones(N, HKV, G, 1, dtype=torch.long, device="cuda")
        tri_prefill.attend(q, pos, c, bt, ptab, tiles, num, den, bm=bm)
        for r, t0, ql, sl in pre:
            qr = q[t0:t0 + ql].permute(1, 2, 0, 3).reshape(1, HKV, G * ql, HD)
            qp = pos[t0:t0 + ql].view(1, 1, ql).expand(1, G, ql).reshape(1, G * ql)
            rn, rd = tri_paged.attend(qr, qp, c, bt[r:r + 1], ptab)
            rn = rn.view(HKV, G, ql, HD).permute(2, 0, 1, 3)
            rd = rd.view(HKV, G, ql, 1).permute(2, 0, 1, 3)
            assert torch.equal(num[t0:t0 + ql], rn) and torch.equal(den[t0:t0 + ql], rd), (lim, r)
        assert not num[tot:].any() and bool((den[tot:] == 1).all())     # untouched rows
