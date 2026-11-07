"""p18: how many distinct block grid exponents does one output row need?

Per tensor, per output row, for block size B in {32, 16} and alignment m in {4, 8}: take the exact-rule grid
g = m * floor(bot / m) over the row's nonempty blocks (bot = min(ex + lowbit), as in analyze.py) and count the
distinct values of g per row. The free grid (m = 1) uses g = top - 23 per block, and is counted too.
This count is the number of shift classes an accumulator-per-class kernel would need per output row.
Reports, over all rows of a model: mean, 90th percentile and max distinct count, per tensor type and in total.

usage: distinct_grid.py MODEL_DIR MODEL_TAG OUT_JSON   (CPU only)
"""
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bitexact.exact.consts import _bitlen, _lowbit, bf16_parts  # noqa: E402

TYPES = [("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"),
         ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"), ("down", "mlp.down_proj")]
BLOCKS = [32, 16]
MS = [1, 4, 8]


def row_counts(W, B, m):
    """Distinct grid exponents per row (over nonempty blocks), for alignment m (m=1 means free grid top-23)."""
    N, K = W.shape
    nb = K // B
    sm, ex = bf16_parts(W)
    mag = sm.abs()
    nz = mag > 0
    low = torch.where(nz, ex + _lowbit(mag), torch.full_like(ex, 10 ** 6)).view(N, nb, B)
    top = torch.where(nz, ex + _bitlen(mag), torch.full_like(ex, -10 ** 6)).view(N, nb, B)
    nzv = nz.view(N, nb, B)
    tb = top.amax(2)
    lb = low.amin(2)
    has = nzv.any(2)
    if m == 1:
        g = tb - 23
    else:
        g = m * torch.div(lb, m, rounding_mode="floor")
    out = np.zeros(N, dtype=np.int64)
    g_np, has_np = g.numpy(), has.numpy()
    for r in range(N):
        vals = np.unique(g_np[r][has_np[r]])
        out[r] = len(vals)
    return out


def main():
    model_dir, tag, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    from pathlib import Path
    files = sorted(Path(model_dir).glob("*.safetensors"))
    res = {}
    t0 = time.time()
    with torch.no_grad():
        for f in files:
            with safe_open(str(f), framework="pt") as fh:
                keys = list(fh.keys())
                for k in keys:
                    if not k.startswith("model.layers."):
                        continue
                    typ = next((t for t, s in TYPES if s in k), None)
                    if typ is None or not k.endswith(".weight"):
                        continue
                    W = fh.get_tensor(k).to(torch.bfloat16)
                    for B in BLOCKS:
                        for m in MS:
                            c = row_counts(W, B, m)
                            res.setdefault(f"B{B}_m{m}", {}).setdefault(typ, []).append(c)
                    print("done", k, round(time.time() - t0, 1), flush=True)
    summary = {}
    for key, bytype in res.items():
        allc = np.concatenate([np.concatenate(v) for v in bytype.values()])
        summary[key] = {"all_mean": float(allc.mean()), "all_p90": float(np.percentile(allc, 90)),
                        "all_max": int(allc.max()), "n_rows": int(allc.size),
                        "by_type": {t: float(np.concatenate(v).mean()) for t, v in bytype.items()}}
    with open(out_json, "w") as fo:
        json.dump(summary, fo, indent=1)
    print("wrote", out_json)


if __name__ == "__main__":
    main()
