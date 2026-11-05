import pytest
import torch

pytestmark = pytest.mark.gpu


def _sm90():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def _cases(N, K):
    """Every dispatch bucket, its edges, and ragged M."""
    from bitexact.int8.gemm_sm90 import TABLE
    ms = {17, 33, 130, 257, 32769}   # torch._int_mm (the reference) needs M > 16
    for mx, _ in TABLE[(N, K)]:
        ms |= {mx, mx + 1} if mx < 1 << 30 else set()
    return sorted(ms)


@pytest.mark.skipif(not _sm90(), reason="sm90")
def test_sm90_equals_int_mm():
    """Every dispatch-table entry on random int8 and on all -128 / all 127."""
    from bitexact.int8.gemm_sm90 import TABLE, gemm, pick
    g = torch.Generator(device="cuda").manual_seed(0)
    for (N, K) in TABLE:
        ws = [torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda", generator=g),
              torch.full((N, K), -128, dtype=torch.int8, device="cuda"),
              torch.full((N, K), 127, dtype=torch.int8, device="cuda")]
        for M in _cases(N, K):
            cfg = pick(N, K, M)
            if cfg is None:
                continue
            As = [torch.randint(-128, 128, (M, K), dtype=torch.int8, device="cuda", generator=g),
                  torch.full((M, K), -128, dtype=torch.int8, device="cuda"),
                  torch.full((M, K), 127, dtype=torch.int8, device="cuda")]
            for a in As:
                for w in ws:
                    assert torch.equal(gemm(a, w, cfg), torch._int_mm(a, w.t())), (N, K, M, cfg)


@pytest.mark.skipif(not _sm90(), reason="sm90")
def test_dispatch_equals_int_mm():
    """What serve calls (gemm_skinny.int_mm) stays exact at every table entry's row bucket."""
    from bitexact.int8.gemm_skinny import int_mm
    from bitexact.int8.gemm_sm90 import TABLE
    g = torch.Generator(device="cuda").manual_seed(1)
    for (N, K) in TABLE:
        w = torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda", generator=g)
        for M in _cases(N, K):
            a = torch.randint(-128, 128, (M, K), dtype=torch.int8, device="cuda", generator=g)
            assert torch.equal(int_mm(a, w), torch._int_mm(a, w.t())), (N, K, M)