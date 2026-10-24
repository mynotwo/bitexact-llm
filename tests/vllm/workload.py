"""Shared workload for the vLLM tests: target and filler prompts as token ids (fixed RNG).

Targets (targets): 12 prompts, lengths 7..700, temperatures 0 / 0.7 / 1.0, seeds 1..12. Targets 4..9 share a
320-token prefix (prefix-cache hits when that is on). Fillers: other prompts and seeds >= 1000, used only
to make the batch busy. targets_long (--long): 3 prompts of 5k to 14k tokens.
"""

import random

V_LO, V_HI = 1000, 120000
TEMPS = [0.0, 0.7, 1.0]


def _ids(rng, n):
    return [rng.randrange(V_LO, V_HI) for _ in range(n)]


def targets(max_new=48):
    rng = random.Random(17)
    shared = _ids(rng, 320)
    lens = [7, 40, 129, 300, 5, 33, 64, 200, 380, 17, 511, 700]
    out = []
    for i, n in enumerate(lens):
        ids = shared + _ids(rng, n) if 3 <= i <= 8 else _ids(rng, n)
        out.append({"seed": i + 1, "ids": ids, "temp": TEMPS[i % 3], "max_new": max_new})
    return out


def targets_long(max_new=16):
    """3 long prompts (past the 2048-row RoPE table of the constants file): 5,000, 9,000 and 14,000 tokens; the last two
    share a 4,000-token prefix. Seeds 101..103."""
    rng = random.Random(23)
    shared = _ids(rng, 4000)
    out = []
    for i, (n, sh) in enumerate([(5000, False), (9000, True), (14000, True)]):
        ids = (shared + _ids(rng, n - 4000)) if sh else _ids(rng, n)
        out.append({"seed": 101 + i, "ids": ids, "temp": TEMPS[i % 3], "max_new": max_new})
    return out


def pick(long, max_new):
    return targets_long() if long else targets(max_new)


def fillers(n=40, seed=99, max_new_hi=96):
    rng = random.Random(seed)
    shared = _ids(rng, 200)
    out = []
    for i in range(n):
        ids = (shared if rng.random() < 0.3 else []) + _ids(rng, rng.randrange(3, 900))
        out.append({"seed": 1000 + i, "ids": ids, "temp": rng.choice(TEMPS),
                    "max_new": rng.randrange(1, max_new_hi)})
    return out


def params(w):
    from vllm import SamplingParams
    return SamplingParams(temperature=w["temp"], seed=w["seed"], max_tokens=w["max_new"], ignore_eos=True,
                          detokenize=False)
