"""cuBLASLt int8 GEMM (BITEXACT_GEMM=lt) equals torch._int_mm bit for bit (skip without CUDA)."""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


@pytest.mark.parametrize("M,N,K", [(24, 3072, 2048), (128, 2048, 8192), (1040, 16384, 2048), (4096, 2048, 8192),
                                   (56, 128256, 2048), (17 * 8, 520, 136)])
def test_lt_int_mm_equal(M, N, K):
    from bitexact.int8.gemm_lt import int_mm
    g = torch.Generator(device="cuda").manual_seed(M + N + K)
    a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="cuda", generator=g)
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda", generator=g)
    a[0] = -128                                             # extreme row x extreme column: K * 2^14
    w[0] = -128
    w[1] = 127
    a[1] = 127                                              # a large sum with arbitrary low bits, which a
    w[2] = torch.randint(100, 128, (K,), dtype=torch.int8, device="cuda", generator=g)  # float32 epilogue rounds
    ref = torch._int_mm(a, w.t())
    out = int_mm(a, w, "lt")
    assert out.dtype == torch.int32 and torch.equal(out, ref)
    assert int(out[0, 0]) == K << 14
    exact = sum(127 * int(x) for x in w[2].tolist())
    assert int(out[1, 2]) == exact and (K < 1024 or exact > 1 << 24)
    # row-strided a (a slice of a wider buffer) takes the same path
    wide = torch.zeros((M, K + 64), dtype=torch.int8, device="cuda")
    wide[:, :K] = a
    assert torch.equal(int_mm(wide[:, :K], w, "lt"), ref)


def test_lt_int_mm_compiled_and_graph():
    from bitexact.int8.gemm_lt import int_mm
    a = torch.randint(-128, 128, (64, 2048), dtype=torch.int8, device="cuda")
    w = torch.randint(-128, 128, (3072, 2048), dtype=torch.int8, device="cuda")
    ref = torch._int_mm(a, w.t())
    f = torch.compile(lambda x, y: int_mm(x, y, "lt"), fullgraph=True)
    assert torch.equal(f(a, w), ref)
    int_mm(a, w, "lt")                                      # plan + workspace exist before capture
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        out = int_mm(a, w, "lt")
    a.copy_(torch.randint(-128, 128, a.shape, dtype=torch.int8, device="cuda"))
    gr.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, torch._int_mm(a, w.t()))
