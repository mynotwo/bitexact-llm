"""vLLM plugin kernels (skip without CUDA or vLLM; written for the Triton that ships with vLLM 0.32): fused
paged attention against the unfused int8 softmax with exact int64 products, and GPU PCG64 words against
numpy."""

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]
pytest.importorskip("vllm")


@pytest.mark.parametrize("HD", [64, 128])
def test_fused_paged_attention_exact(HD):
    from bitexact.int8.consts import exp_table
    from bitexact.int8.segs import softmax
    from bitexact.intops import LIM
    from bitexact.vllm_plugin import tri_paged
    g = torch.Generator(device="cuda").manual_seed(0)
    lut = exp_table().cuda()
    nb, bs, HKV = 96, 16, 8
    for lim in (LIM[3], 1 << 13, 1 << 10):
        c = torch.randint(-lim, lim + 1, (nb, HKV, bs, 2 * HD), device="cuda", generator=g).to(torch.int32)
        for R, M, P in [(3, 4, 600), (5, 4, 3000), (6, 1, 900), (2, 3, 50), (1, 1024, 2000), (2, 64, 600),
                        (4, 20, 100)]:
            bt = torch.randint(1, nb, (R, 200), device="cuda", generator=g).to(torch.int32)
            qpos = torch.randint(0, P, (R, M), device="cuda", generator=g)
            q = torch.randint(-lim, lim + 1, (R, HKV, M, HD), device="cuda", generator=g)
            S = int(qpos.max()) + 1
            s = torch.arange(S, device="cuda")
            kv = c[bt.long()[:, s // bs], :, (s % bs)[None].expand(R, -1)].long()     # [R, S, HKV, 2 HD]
            K, V = kv[..., :HD].permute(0, 2, 1, 3).cpu(), kv[..., HD:].permute(0, 2, 1, 3).cpu()
            raw = torch.einsum("rhmd,rhsd->rhms", q.cpu(), K)                     # exact int64 on CPU
            p, den = softmax(raw, qpos.cpu()[:, None, :, None], lut.cpu(), HD)
            ref_num = torch.einsum("rhms,rhsd->rhmd", p, V)
            for ns in (None, 1, 5, 64):
                for bm in (None, 16):                   # M <= 4: limb-stacked decode kernels, then the general ones
                    num, den2 = tri_paged.attend(q, qpos, tri_paged.to_limbs(c), bt, tri_paged.weight_table(lut), nsplit=ns, bm=bm)
                    assert torch.equal(num.cpu(), ref_num) and torch.equal(den2.cpu(), den), (R, M, ns, bm)


def test_gpu_pcg_matches_numpy():
    from bitexact import sampling
    from bitexact.vllm_plugin import gpu_pcg
    rng = np.random.default_rng(5)
    keys = [(int(s), int(t)) for s, t in zip(rng.integers(0, 2**31, 64), rng.integers(0, 4096, 64))]
    keys += [(0, 0), (1, 0), (2**32 - 1, 7), (12, 47)]
    got = gpu_pcg.raw_rows(keys, "cuda").cpu()
    assert torch.equal(got, sampling.raw_rows(keys))


def test_integer_sample_matches_rule():
    """The plugin sampler (one temperature: whole batch; mixed: gathered groups) against bitexact.sampling per row."""
    from bitexact import sampling
    from bitexact.int8.consts import FR
    from bitexact.llama import V, ln_table
    from bitexact.vllm_plugin.sampler import integer_sample
    g = torch.Generator(device="cuda").manual_seed(1)
    li = torch.randint(-(1 << 20), 1 << 20, (6, V), device="cuda", generator=g)
    li[:, 77] = li.max() + 1                                     # ties: the first argmax must win
    li[:, 5] = li.max()
    lnt = ln_table().cuda()
    seeds, ts = [0, 3, 3, 1001, 7, 9], [0, 5, -2, 40, 1, 2]
    for temps in ([(0, 1)] * 6, [(1, 1)] * 6, [(0, 1), (1, 1), (7, 10), (1, 1), (0, 1), (7, 10)]):
        got = integer_sample(li, seeds, ts, temps, lnt).cpu()
        for r, (tn, td) in enumerate(temps):
            raw = sampling.raw_rows([(seeds[r], max(ts[r], 0))])
            ref = sampling.sample(li[r:r + 1].cpu(), tn, td, None if tn == 0 else sampling.gumbel(raw, lnt.cpu(), FR))
            assert int(got[r]) == int(ref[0]), (temps, r)


def test_logits_argmax_float_exact():
    from bitexact.intops import first_argmax
    from bitexact.vllm_plugin import tri_logits
    g = torch.Generator(device="cuda").manual_seed(0)
    for M, V, lim in [(1, 128256, 1 << 40), (128, 128256, 1 << 40), (7, 5000, 3), (33, 1023, 1 << 61), (0, 64, 9)]:
        li = torch.randint(-lim, lim, (M, V), device="cuda", generator=g)
        if M > 2:
            li[1] = 5                                          # all tied: index 0
            li[2] = li[2].amin()
            li[2, [V - 1, V // 2, 3]] = lim                    # tie between chunks: first one
            li[0] = -(1 << 63)                                 # int64 minimum everywhere
        ref = first_argmax(li) if M else torch.empty(0, dtype=torch.long, device="cuda")
        for nch in (None, 1, 3, 40):
            f, y = tri_logits.argmax_float(li, nch=nch)
            assert torch.equal(y, ref), (M, V, nch)
            assert torch.equal(f.view(torch.int32), (li.float() * (2.0 ** -16)).view(torch.int32)), (M, V, nch)
        assert torch.equal(tri_logits.first_argmax(li), ref)


@pytest.mark.parametrize("M", [1, 3, 64, 128, 300])
def test_tri_mlp_silu_quant_exact(M):
    from bitexact.int8 import segs
    from bitexact.int8.consts import FA, FR, SILU_LIM
    from bitexact.llama import FF
    from bitexact.vllm_plugin import tri_mlp
    g = torch.Generator(device="cuda").manual_seed(M)
    xs = torch.arange(-SILU_LIM << FA, (SILU_LIM << FA) + 1, dtype=torch.float64) / (1 << FA)
    lut = torch.round(xs / (1.0 + torch.exp(-xs)) * (1 << FR)).long().cuda()
    K = 256
    for mb, fm, check in [(20, 30, False), (20, 22, True), (35, 30, True), (12, 8, False)]:
        sg = segs.Segs(2, True, False, post_check=check)
        x = torch.randint(-(1 << 26), 1 << 26, (M, K), device="cuda", generator=g)
        x = x >> torch.randint(0, 20, (M, 1), device="cuda", generator=g)
        w = torch.randint(-128, 128, (2 * FF, K), device="cuda", generator=g).to(torch.int8)
        m = torch.randint(-(1 << mb), 1 << mb, (2 * FF,), device="cuda", generator=g)
        fmsh = torch.tensor(fm, device="cuda")
        a, mant, e = sg.pre_plain(x)
        c = torch._int_mm(a, w.t())
        gu, f1 = sg.post(c, m, fmsh, mant, e)
        ref = sg.silu_mul(gu, lut)
        got = tri_mlp.silu_quant(c, m, fmsh, mant, e, lut, check)
        for r, o in zip(ref[:3], got[:3]):
            assert r.dtype == o.dtype and torch.equal(r, o), (M, mb, fm)
        rf = ref[3] if f1 is None else ref[3] | f1
        assert bool(rf) == bool(got[3]), (M, mb, fm, bool(rf))


def _head_case(M, V, seed, scale):
    from bitexact.vllm_plugin import tri_sample
    g = torch.Generator(device="cuda").manual_seed(seed)
    c = torch.randint(-scale, scale, (2 * M + 24, V), device="cuda", generator=g).to(torch.int32)
    m = torch.randint(-(1 << 22), 1 << 22, (V,), device="cuda", generator=g)
    mant = torch.randint(1 << 23, (1 << 24) + 1, (M,), device="cuda", generator=g)
    e = torch.randint(-6, 6, (M,), device="cuda", generator=g)
    fmsh = torch.tensor(26, device="cuda")
    return tri_sample.Head(c, m, fmsh, mant, e)


def _head_ref(h):
    from bitexact.int8.segs import post
    from bitexact.intops import join_limbs
    return post(join_limbs(h.c, h.M, 2), h.m, h.fmsh, h.mant, h.e, False)[0]


@pytest.mark.parametrize("M", [1, 5, 128])
def test_tri_sample_head_exact(M):
    from bitexact.intops import first_argmax
    from bitexact.vllm_plugin import tri_sample
    for V, scale in [(128256, 1 << 18), (128256, 1 << 29), (5000, 3), (1023, 1 << 29)]:
        h = _head_case(M, V, M + V, scale)
        li = _head_ref(h)
        assert int(li.abs().max()) > 0
        ref = first_argmax(li)
        for mode in ("float", "onehot", "none"):
            f, y = tri_sample.head(h, mode)
            assert torch.equal(y, ref), (M, V, mode)
            if mode == "float":
                assert torch.equal(f.view(torch.int32), (li.float() * (2.0 ** -16)).view(torch.int32))
            if mode == "onehot":
                want = torch.full_like(f, float("-inf"))
                want.scatter_(1, ref[:, None], 0.0)
                assert torch.equal(f, want)


def test_tri_sample_ties_first_index():
    from bitexact.intops import first_argmax
    from bitexact.vllm_plugin import tri_sample
    h = _head_case(4, 128256, 9, 1 << 12)
    h.c[:] = 0
    h.c[0, [777, 100000, 5]] = 1000                        # row 0: tie, first is 5
    h.m[:] = 1
    li = _head_ref(h)
    assert torch.equal(tri_sample.head(h, "none")[1], first_argmax(li))
    assert int(tri_sample.head(h, "none")[1][1]) == 0      # all zeros: index 0


@pytest.mark.parametrize("M", [1, 7, 128])
def test_tri_sample_noisy_exact(M):
    from bitexact import sampling
    from bitexact.int8.consts import FR
    from bitexact.llama import V, ln_table
    from bitexact.vllm_plugin import tri_sample
    lnt = ln_table().cuda()
    rng = np.random.default_rng(M)
    for scale in (1 << 18, 1 << 29):
        h = _head_case(M, V, M + scale, scale)
        li = _head_ref(h)
        keys = [(int(s), int(t)) for s, t in zip(rng.integers(0, 2**31, M), rng.integers(0, 4096, M))]
        keys[:3] = [(0, 0), (2**32 - 1, 2**32 - 1), (1, 2**32 - 1)][:M]
        for temps in ([(1, 1)] * M, [(0, 1)] * M, [(2, 3), (7, 10), (1, 10), (0, 1), (3, 2), (1, 100), (100, 1)] * 20,
                      [(1, 1), (0, 1)] * 70):
            temps = temps[:M]
            got = tri_sample.noisy(h, tri_sample.row_state(keys, temps), lnt)
            st4 = np.array([[s, t, tn, td] for (s, t), (tn, td) in zip(keys, temps)], dtype=np.int64)
            out = torch.empty(M, V, dtype=torch.float32, device="cuda")
            got4 = tri_sample.noisy(h, st4, lnt, out=out)        # PCG state derived in the kernel, logits written
            want = torch.full_like(out, float("-inf"))
            want.scatter_(1, got[:, None], 0.0)
            assert torch.equal(got4, got) and torch.equal(out, want), (M, scale, temps)
            raw = sampling.raw_rows(keys).cuda()
            for r, (tn, td) in enumerate(temps):
                ref = sampling.sample(li[r:r + 1], tn, td, sampling.gumbel(raw[r:r + 1], lnt, FR) if tn else None)
                assert int(got[r]) == int(ref[0]), (M, scale, temps[r], r)


@pytest.mark.parametrize("M", [1, 5, 128, 300])
def test_tri_logp_matches_int_logp(M):
    from bitexact.llama import V, ln_table
    from bitexact.vllm_plugin import tri_logp
    from bitexact.vllm_plugin.rl import int_logp
    lnt = ln_table().cuda()
    for Vn, scale in [(V, 1 << 18), (V, 1 << 23), (V, 1 << 29), (5000, 3), (1023, 1 << 25)]:
        h = _head_case(M, Vn, M + Vn + scale, scale)
        li = _head_ref(h)
        g = torch.Generator(device="cuda").manual_seed(M)
        tok = torch.randint(0, Vn, (M,), device="cuda", generator=g)
        tok[: M // 3] = li[: M // 3].argmax(-1)                       # the max itself
        tok[M // 3: M // 2] = li[M // 3: M // 2].argmin(-1)           # far below the max (clamped terms)
        got = tri_logp.logp(h, tok, lnt)
        want = int_logp(li, tok, lnt)
        assert got.dtype == want.dtype and torch.equal(got.view(torch.int64), want.view(torch.int64)), (M, Vn, scale)
