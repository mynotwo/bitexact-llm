"""bmm gives the same exact integers with 2 pipeline stages (used where 3 exceed shared memory, sm_89)."""
import pytest
import torch

from bitexact import bmm as B

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and B.HAVE_TRITON), reason="needs CUDA + Triton")


@pytest.mark.parametrize("M,K,N", [(4, 128, 300), (256, 128, 300), (256, 300, 128), (8, 64, 40)])
def test_two_stages_same_integers(M, K, N):
    g = torch.Generator().manual_seed(M * 7 + K + N)
    lim = 8355711
    a = torch.randint(-lim, lim + 1, (8, M, K), generator=g, dtype=torch.int32).cuda()
    b = torch.randint(-lim, lim + 1, (8, K, N), generator=g, dtype=torch.int32).cuda()
    ref = (a.cpu().long().unsqueeze(-1) * b.cpu().long().unsqueeze(1)).sum(2)   # exact int64
    B._STAGES.clear()
    c3 = B.bmm(a, b).cpu()
    BM, BN = (16 if M <= 16 else 64), (64 if N <= 64 else 128)
    B._STAGES[(a.device.index, 3, 3, BM, BN)] = 2
    try:
        c2 = B.bmm(a, b).cpu()
    finally:
        B._STAGES.clear()
    assert torch.equal(c3, ref) and torch.equal(c2, ref)
