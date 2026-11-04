import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
def test_skinny_equals_int_mm():
    from bitexact.int8.gemm_skinny import TABLE, int_mm
    g = torch.Generator(device="cuda").manual_seed(0)
    for (N, K) in set(TABLE) | {(128256, 2048)}:
        w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda", generator=g)
        for R in (24, 32, 56, 64, 72, 100, 128, 200, 256, 264, 384):
            for kind in ("rand", "max", "min"):
                a = torch.randint(-128, 128, (R, K), dtype=torch.int8, device="cuda", generator=g)
                if kind == "max":
                    a.fill_(-128); w2 = torch.full_like(w, -128)
                elif kind == "min":
                    a.fill_(127); w2 = torch.full_like(w, -128)
                else:
                    w2 = w
                assert torch.equal(int_mm(a, w2), torch._int_mm(a, w2.t())), (N, K, R, kind)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
def test_large_equals_int_mm():
    from bitexact.int8.gemm_skinny import LARGE, LARGE_MIN, int_mm
    g = torch.Generator(device="cuda").manual_seed(1)
    for (N, K) in LARGE:
        w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda", generator=g)
        for R in (LARGE_MIN, 2056, 4096, 5000, 16384, 20000, 40000):
            for kind in ("rand", "max", "min"):
                a = torch.randint(-128, 128, (R, K), dtype=torch.int8, device="cuda", generator=g)
                w2 = w
                if kind == "max":
                    a.fill_(-128); w2 = torch.full_like(w, -128)
                elif kind == "min":
                    a.fill_(127); w2 = torch.full_like(w, -128)
                assert torch.equal(int_mm(a, w2), torch._int_mm(a, w2.t())), (N, K, R, kind)
