"""Fused element-wise kernels of exact mode (vllm_plugin/exact_fuse.py) against the eager exact/segs.py stages
on real constants, and the fused ExactEngine forward against the unfused one (skip without CUDA, vLLM or
the exact constants)."""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]
pytest.importorskip("vllm")


@pytest.fixture(scope="module")
def mdl(const):
    kind, c = const
    if kind != "exact":
        pytest.skip("exact constants only")
    from bitexact.exact.model import ExactLlama
    m = ExactLlama(c, "cuda", n_layers=2, compiled=False)
    return m


def _rand(g, shape, bits_lo, bits_hi):
    """int64 values whose magnitude per row is 2^U(bits_lo, bits_hi), random sign (+ a few outliers)."""
    b = torch.randint(bits_lo, bits_hi, (shape[0],) + (1,) * (len(shape) - 1), device="cuda", generator=g)
    v = (torch.rand(shape, device="cuda", generator=g, dtype=torch.float64) * 2 - 1) * torch.pow(2.0, b.double())
    return v.round().long()


def test_norm(mdl):
    from bitexact.vllm_plugin import exact_fuse as ef
    m, g = mdl, torch.Generator(device="cuda").manual_seed(0)
    fl = torch.zeros(1, dtype=torch.int32, device="cuda")
    for M in (1, 5, 64):
        for lo, hi in ((10, 20), (20, 30), (26, 36), (0, 6)):
            x, y = _rand(g, (M, m.D), lo, hi), _rand(g, (M, m.D), lo, hi)
            for gam in (m.layers[0]["ga"], m.layers[1]["gm"], m.gf):
                rx, rq, rs, rerr = m.sg.add_norm_y(y, x, *gam, torch.zeros((), dtype=torch.bool, device="cuda"))
                fx, fq, fs = ef.add_norm(m, y, x, gam, fl)
                assert torch.equal(fx, rx) and torch.equal(fq, rq) and torch.equal(fs, rs), (M, lo, hi)
                pq, ps = ef.pre_norm(m, x, gam, fl)
                rq, rs = m.sg.pre_norm(x, *gam)
                assert torch.equal(pq, rq) and torch.equal(ps, rs)
            ids = torch.randint(0, 100000, (M,), device="cuda", generator=g)
            ex, eq, es = ef.embed_norm(m, ids, m.layers[0]["ga"], fl)
            rx = m.emb[ids].long()
            rq, rs = m.sg.pre_norm(rx, *m.layers[0]["ga"])
            assert torch.equal(ex, rx) and torch.equal(eq, rq) and torch.equal(es, rs)
    x = _rand(g, (3, m.D), 20, 24)
    x[1, 5] = 1 << 41                                           # bound flag
    ef.add_norm(m, x * 0, x, None, fl)
    assert int(fl[0]) == 1


@pytest.mark.parametrize("N", [1, 7, 96])
def test_qkv(mdl, N):
    from bitexact.vllm_plugin import exact_fuse as ef
    m, g = mdl, torch.Generator(device="cuda").manual_seed(1)
    H, HKV, HD = m.H, m.HKV, m.HD
    G = H // HKV
    nb, bs = 20, 16
    fl = torch.zeros(1, dtype=torch.int32, device="cuda")
    for lo, hi in ((10, 20), (16, 28), (4, 12)):
        y = _rand(g, (N, (H + 2 * HKV) * HD), lo, hi)
        pos = torch.randint(0, 1000, (N,), device="cuda", generator=g)
        slot = torch.randperm(nb * bs, device="cuda", generator=g)[:N]
        cache = torch.zeros(nb, HKV, bs, 8 * HD, dtype=torch.int8, device="cuda")
        err = torch.zeros((), dtype=torch.bool, device="cuda")
        qg, shg, k, v, err = m.sg.qkv_y(y, pos.view(N, 1), m.cos, m.sin, N, 1, err)
        ref = torch.zeros_like(cache)
        from bitexact.vllm_plugin import tri_paged
        tri_paged.store(ref, slot, k[:, 0], v[:, 0])
        q, sh = ef.qkv(m, y, pos, slot, cache, fl)
        assert torch.equal(q, qg.view(N, -1)) and torch.equal(sh, shg.view(N, H)), (N, lo, hi)
        assert torch.equal(cache, ref)
        q2, sh2 = ef.qkv(m, y, pos, slot, None, fl)
        assert torch.equal(q2, q) and torch.equal(sh2, sh)
        assert bool(fl[0]) == bool(err) or lo == 16, (lo, bool(err), int(fl[0]))
        fl.zero_()


def test_attn_out(mdl):
    from bitexact.vllm_plugin import exact_fuse as ef
    m, g = mdl, torch.Generator(device="cuda").manual_seed(2)
    H, HKV, HD = m.H, m.HKV, m.HD
    G = H // HKV
    for N in (1, 9, 64):
        for lo, hi in ((20, 40), (30, 46)):
            num = _rand(g, (N * HKV, G, HD), lo, hi)
            den = torch.randint(1 << 22, 1 << 33, (N * HKV, G, 1), device="cuda", generator=g)
            rq, rs = m.sg.attn_out(num, den, 1)
            q, sh = ef.attn_out(m, num.view(N, HKV, G, HD), den.view(N, HKV, G, 1))
            assert torch.equal(q, rq) and torch.equal(sh, rs), (N, lo, hi)


def test_silu(mdl):
    from bitexact.vllm_plugin import exact_fuse as ef
    m, g = mdl, torch.Generator(device="cuda").manual_seed(3)
    fl = torch.zeros(1, dtype=torch.int32, device="cuda")
    for N in (1, 6, 64):
        for lo, hi in ((20, 27), (24, 30), (8, 24)):
            y = _rand(g, (N, 2 * 8192), lo, hi)
            q, sh = ef.silu_quant(m, y, m.exp2, fl)
            rq, rs, rerr = m.sg.gu_silu_y(y, m.exp2, torch.zeros((), dtype=torch.bool, device="cuda"))
            assert torch.equal(q, rq) and torch.equal(sh, rs), (N, lo, hi)
            assert bool(fl[0]) == bool(rerr)
            fl.zero_()


def test_engine_fused_matches_unfused(const):
    kind, c = const
    if kind != "exact":
        pytest.skip("exact constants only")
    from bitexact.vllm_plugin import exact_run
    c = dict(c)
    c["n_layers"] = 2
    eng = exact_run.ExactEngine(c, torch.device("cuda"))
    N, bs = 40, 16
    g = torch.Generator(device="cuda").manual_seed(4)
    ids = torch.randint(1000, 100000, (N,), device="cuda", generator=g)
    pos = torch.randint(0, 6, (N,), device="cuda", generator=g)
    slot = torch.arange(N, device="cuda") * bs + pos
    bt = (torch.arange(N, device="cuda", dtype=torch.int32) + 1).view(N, 1)
    rmap = torch.arange(N, device="cuda")
    outs = []
    for ew in (True, False):
        exact_run.EW_FUSE = ew
        kvs = [torch.zeros(N + 2, eng.HKV, bs, 8 * eng.HD, dtype=torch.int8, device="cuda") for _ in range(eng.L)]
        kvs = [k.view(torch.bfloat16) for k in kvs]
        slot2 = slot + bs                                   # block 0 is the null block
        h = eng.forward(kvs, ids, pos, slot2, bt, rmap)
        outs.append((h, eng.logits(h[:8]), [k.view(torch.int8).clone() for k in kvs]))
        eng.check()
    exact_run.EW_FUSE = True
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
    assert all(torch.equal(a, b) for a, b in zip(outs[0][2], outs[1][2]))
