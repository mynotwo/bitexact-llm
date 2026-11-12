"""Full exact-mode model path on the weight-free synthetic checkpoint, pinned to golden digests.

The goldens were recorded on Apple M4 Pro (arm64, CPU, torch 2.15). CI recomputes them on Linux x86_64, so a
green run there is a cross-platform bit-identity check of the whole forward pass, KV cache, batching and
integer sampler, not only of the arithmetic fixtures. A changed golden means the integer model changed: that
is a new model identity and needs a deliberate update here, never a tolerance.
"""

import hashlib
import json

import pytest
import torch

from bitexact import generate, load, synthetic
from bitexact.repro import _batch_verify

WEIGHTS_SHA = "e82e6815dbb0ab8d14f595d9598413065507fd0fec68a1c77b2956d52ab38397"
CONSTANTS_SHA = "963eaca6036acecc7a6846a3677ca82701f4d16b4cdf06bbf634344587f2f54c"
ROWS_SHA = "a6f44d1e6c6ec25b7eb520a0e2182a8e623df02ce0e058cfe4970c60693afbcf"   # 4 prompts x 16 tokens, T = 7/10
MAX_NEW = 16


@pytest.fixture(scope="module")
def consts():
    return synthetic.constants()


def _rows(model):
    rows, tokens = {}, []
    for i, p in enumerate(synthetic.PROMPTS):
        seq, hashes = generate(model, p, MAX_NEW, 7, 10, seed=i)
        tokens.append(seq)
        rows.update({f"{i}:{t}": [h, y] for t, (h, y) in enumerate(zip(hashes, seq))})
    return rows, tokens


def test_weights_are_platform_independent():
    h = hashlib.sha256()
    for k, v in sorted(synthetic.state_dict().items()):
        h.update(k.encode())
        h.update(v.view(torch.int16).numpy().tobytes())
    assert h.hexdigest() == WEIGHTS_SHA


def test_constants_identity(consts):
    from bitexact.exact.consts import summary
    s = summary(consts)
    assert s["all_same_bits"] and s["nnz_res"] > 0          # sparse residual path is exercised
    assert consts["sha256"] == CONSTANTS_SHA


@pytest.mark.parametrize("threads", [1, 4])
def test_generation_golden_and_schedule_invariant(consts, threads):
    torch.set_num_threads(threads)
    model = load(consts, "cpu")
    rows, tokens = _rows(model)
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    assert digest == ROWS_SHA
    for bs in (1, 3, 4):                                    # right-padded full forward, several batch splits
        _batch_verify(model, synthetic.PROMPTS, tokens, rows, bs)
