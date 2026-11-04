"""tri_fuse kernels against the eager int8 segments (needs CUDA and BITEXACT_INT8_CONST), bit for bit.

Run: BITEXACT_INT8_CONST=int8_const.pt pytest -q tests/test_tri_fuse.py
Random int32 GEMM outputs (limb products within the real bounds) go through the real weights' scales and gammas.
"""

import os

import pytest
import torch

CONST = os.environ.get("BITEXACT_INT8_CONST")
CONST8 = os.environ.get("BITEXACT_INT8_CONST_8B")           # optional: the Llama-3.1-8B constants (adds the 8B geometry)
pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not (torch.cuda.is_available() and CONST), reason="needs CUDA + BITEXACT_INT8_CONST")]
pytest.importorskip("triton")


@pytest.fixture(scope="module", params=["1b", "8b"])
def m(request):
    from bitexact.int8.model import IntLlama
    from bitexact.vllm_plugin import tri_fuse
    path = CONST if request.param == "1b" else CONST8
    if not path:
        pytest.skip("needs BITEXACT_INT8_CONST_8B")
    mm = IntLlama(torch.load(path, map_location="cpu", weights_only=False), "cuda", nl=2, compiled=False)
    tri_fuse.scratch(mm)
    yield mm
    del mm
    torch.cuda.empty_cache()


def _c(M, N, g, scale):
    """int32 limb-product rows [2M (+ pad), N]: two limbs of random sums, ranges as real GEMM outputs."""
    rows = 2 * M
    rows += max(24, (rows + 7) // 8 * 8) - rows
    return torch.randint(-scale, scale + 1, (rows, N), device="cuda", generator=g, dtype=torch.int32)


def _pre(m, M, g):
    """A (mant, e) pair as pre_norm would give."""
    x = torch.randint(-(1 << 22), 1 << 22, (M, m.D), device="cuda", generator=g)
    return m.sg.pre_norm(x, m.layers[0]["ga"])[1:]


@pytest.mark.parametrize("M", [1, 3, 24, 128, 130])
def test_add_norm_exact(m, M):
    from bitexact.vllm_plugin import tri_fuse
    g = torch.Generator(device="cuda").manual_seed(M)
    for name, nxt in (("wo", "gm"), ("wd", "ga")):
        lin = m.layers[3][name]
        N = lin[0].shape[0]
        for scale in (1 << 12, 1 << 20, 1 << 27):
            c = _c(M, N, g, scale)
            mant, e = _pre(m, M, g)
            x = torch.randint(-(1 << 30), 1 << 30, (M, N), device="cuda", generator=g)
            ref_x = m.sg.post_add(c, lin[1], lin[2], mant, e, x)[0] if m.post_check else None
            ref_x = x + m.sg.post(c, lin[1], lin[2], mant, e)[0]
            ref = m.sg.pre_norm(ref_x, m.layers[3][nxt])
            xo, a, mt, et = tri_fuse.add_norm(m, c, lin, mant, e, x, m.layers[3][nxt])
            assert torch.equal(xo, ref_x) and torch.equal(a, ref[0]) and torch.equal(mt, ref[1]) \
                and torch.equal(et, ref[2]), (M, name, scale)
    ids = torch.randint(0, m.emb.shape[0], (M,), device="cuda", generator=g)
    xo, a, mt, et = tri_fuse.embed_norm(m, ids, m.layers[0]["ga"])
    ref = m.sg.pre_norm(m.emb[ids].long(), m.layers[0]["ga"])
    assert torch.equal(xo, m.emb[ids].long()) and torch.equal(a, ref[0]) and torch.equal(mt, ref[1])


@pytest.mark.parametrize("M", [1, 5, 128])
def test_qkv_store_exact(m, M):
    from bitexact.vllm_plugin import tri_fuse, tri_paged
    HD, HKV = m.HD, m.HKV
    g = torch.Generator(device="cuda").manual_seed(M + 7)
    lin = m.layers[2]["wqkv"]
    bs, nb = 16, 40
    for scale in (1 << 12, 1 << 22):
        c = _c(M, lin[0].shape[0], g, scale)
        mant, e = _pre(m, M, g)
        pos = torch.randint(0, 2000, (M,), device="cuda", generator=g)
        slot = torch.randperm(nb * bs, device="cuda", generator=g)[:M]
        slot[::5] = -1                       # skipped rows (padding)
        y = m.sg.post(c, lin[1], lin[2], mant, e)[0].view(M, 1, -1)
        qg, k, v, flag = m.sg.qkv(y, m.cos[pos][:, None, None], m.sin[pos][:, None, None])
        ref_c = torch.zeros(nb, HKV, bs, 8 * HD, dtype=torch.int8, device="cuda")
        tri_paged.store(ref_c, slot, k[:, 0], v[:, 0])
        got_c = torch.zeros_like(ref_c)
        m.err.zero_()
        q = tri_fuse.qkv(m, c, lin, mant, e, pos, slot, got_c)
        assert torch.equal(q, qg.view(M, HKV, m.H // HKV, HD)) and torch.equal(got_c, ref_c), (M, scale)
        assert bool(m.err) == bool(flag), (M, scale)
    m.err.zero_()


@pytest.mark.parametrize("M", [1, 3, 128])
def test_attn_out_and_mlp_exact(m, M):
    from bitexact.vllm_plugin import paged, tri_fuse, tri_mlp
    g = torch.Generator(device="cuda").manual_seed(M + 11)
    for scale in (1 << 10, 1 << 26):
        num = torch.randint(-scale * 50, scale * 50 + 1, (M, m.HKV, m.H // m.HKV, m.HD), device="cuda", generator=g)
        den = torch.randint(1, scale + 2, (M, m.HKV, m.H // m.HKV, 1), device="cuda", generator=g)
        ref = m.sg.pre_plain(paged.attn_div(num, den).reshape(M, m.H * m.HD))
        got = tri_fuse.attn_out(m, num, den)
        assert all(torch.equal(a, b) for a, b in zip(got, ref)), (M, scale)
    lin = m.layers[1]["wgu"]
    for scale in (1 << 12, 1 << 22):
        c = _c(M, lin[0].shape[0], g, scale)
        mant, e = _pre(m, M, g)
        ref = m.sg.silu_mul(m.sg.post(c, lin[1], lin[2], mant, e)[0], m.silu_t)
        scr = tri_fuse.scratch(m)
        got = tri_fuse.mlp_mid(m, m.layers[1], c, mant, e, 1)
        assert all(torch.equal(a, b) for a, b in zip(got, ref[:3])), (M, scale)
        m.err.zero_()
        tri_fuse.flag_end(m)
        assert bool(m.err) == bool(ref[3]) and not scr.any()
    m.err.zero_()


@pytest.mark.parametrize("big", [False, True])
def test_final_bound_flag(m, big):
    from bitexact.vllm_plugin import tri_fuse
    lin = m.layers[0]["wd"]
    M = 4
    x = torch.zeros(M, m.D, dtype=torch.long, device="cuda")
    x[2, 5] = (1 << 42) if big else (1 << 42) - 1
    c = torch.zeros(2 * M + 16, m.D, dtype=torch.int32, device="cuda")
    mant = torch.ones(M, dtype=torch.long, device="cuda")
    e = torch.zeros(M, dtype=torch.long, device="cuda")
    m.err.zero_()
    out = tri_fuse.add_norm(m, c, lin, mant, e, x, None)[0]
    assert torch.equal(out, x) and bool(m.err) == big
    m.err.zero_()
