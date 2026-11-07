"""p18b: free per-block grid (g = top - 23), per output row: distinct g, distinct (g mod 8), accumulator classes.

Free grid: each block of B weights along K uses grid exponent g = top - 23, top = max(ex + bitlen(mag)) over the
block's nonzero entries (23 bits = 3 signed base-256 limbs). Per output row this script records:
- distinct g over the row's nonempty blocks (number of grid exponents the row needs);
- distinct (g mod 8) (number of limb-diagonal offsets; exponents differing by a multiple of 8 share a chain);
- per class r = (g mod 8), the number of K positions in blocks of that class, max over r (accumulator depth).
Histograms are kept so that percentiles and tails can be recomputed exactly on the host.

usage: free_rows.py MODEL_DIR TAG OUT_JSONL [--layers all|0,1,..] [--workers N] [--check]
CPU only. Shipped bf16_parts/_bitlen/_lowbit from src/bitexact/exact/consts.py.
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bitexact.exact.consts import _bitlen, _lowbit, bf16_parts  # noqa: E402

BLOCKS = [256, 128, 64, 32, 16, 8]
CAP = 23
SENT = 10 ** 9
TYPES = [("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"),
         ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"), ("down", "mlp.down_proj")]
DG_CLIP = 64
DM_CLIP = 8


def distinct(vals, valid):
    """Number of distinct valid values per row. vals, valid: (N, nb)."""
    s = torch.where(valid, vals, torch.full_like(vals, SENT)).sort(dim=1).values
    v = s != SENT
    newv = torch.cat([v[:, :1], v[:, 1:] & (s[:, 1:] != s[:, :-1])], 1)
    return newv.sum(1)


def tensor_stats(W):
    N, K = W.shape
    sm, ex = bf16_parts(W)
    mag = sm.abs()
    nz = mag > 0
    top = torch.where(nz, ex + _bitlen(mag), torch.full_like(ex, -10 ** 6))
    recs = {}
    for B in BLOCKS:
        if K % B:
            continue
        nb = K // B
        tb = top.view(N, nb, B).amax(2)
        has = nz.view(N, nb, B).any(2)
        g = tb - CAP
        gm8 = torch.remainder(g, 8)
        dg = distinct(g, has)
        dm = distinct(gm8, has)
        onehot = torch.nn.functional.one_hot(gm8.clamp(0, 7), 8).to(torch.int64) * has.unsqueeze(-1)
        cls_k = onehot.sum(1) * B                      # (N, 8): K positions per class, per row
        row_max_k = cls_k.max(1).values                # (N,)
        recs[f"B{B}"] = {
            "N": N, "K": K, "sum_dg": int(dg.sum()), "max_dg": int(dg.max()),
            "hist_dg": torch.bincount(dg.clamp(max=DG_CLIP), minlength=DG_CLIP + 1).tolist(),
            "sum_dm": int(dm.sum()), "max_dm": int(dm.max()),
            "hist_dm": torch.bincount(dm.clamp(max=DM_CLIP), minlength=DM_CLIP + 1).tolist(),
            "max_class_k": int(row_max_k.max()), "sum_max_class_k": int(row_max_k.sum()),
        }
    return recs


def brute(W):
    """Plain-Python reference for tensor_stats on a small matrix: returns the same fields for B in (32, 8)."""
    sm, ex = bf16_parts(W)
    sm = sm.numpy()
    ex = ex.numpy()
    N, K = sm.shape
    out = {}
    for B in (32, 8):
        nb = K // B
        dg_all, dm_all, mk_all = [], [], []
        for r in range(N):
            gs = []
            for b in range(nb):
                nzs = [(ex[r, c] + int(abs(sm[r, c])).bit_length()) for c in range(b * B, (b + 1) * B) if sm[r, c] != 0]
                if nzs:
                    gs.append((max(nzs) - CAP, b))
            dg = len({g for g, _ in gs})
            dm = len({g % 8 for g, _ in gs})
            cls = {}
            for g, _ in gs:
                cls[g % 8] = cls.get(g % 8, 0) + B
            dg_all.append(dg)
            dm_all.append(dm)
            mk_all.append(max(cls.values()) if cls else 0)
        out[f"B{B}"] = {"sum_dg": sum(dg_all), "max_dg": max(dg_all), "sum_dm": sum(dm_all),
                        "max_dm": max(dm_all), "max_class_k": max(mk_all), "sum_max_class_k": sum(mk_all)}
    return out


def self_check():
    torch.manual_seed(1)
    N, K = 12, 512
    W = (torch.randn(N, K) * torch.exp2(torch.randint(-10, 4, (N, K)).float())).to(torch.bfloat16)
    W[0, :20] = 0
    fast = tensor_stats(W)
    ref = brute(W)
    bad = 0
    for key, r in ref.items():
        f = fast[key]
        for k in ("sum_dg", "max_dg", "sum_dm", "max_dm", "max_class_k", "sum_max_class_k"):
            if f[k] != r[k]:
                bad += 1
                print("SELFCHECK MISMATCH", key, k, f[k], r[k], flush=True)
    print("selfcheck mismatches:", bad, flush=True)
    if bad:
        sys.exit(1)


def work(args):
    key, path, layer, typ = args
    torch.set_num_threads(1)
    with safe_open(path, framework="pt") as fh:
        W = fh.get_tensor(key).to(torch.bfloat16)
    return {"key": key, "layer": layer, "type": typ, "recs": tensor_stats(W)}


def tasks(model_dir, layers):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        pairs = [(k, os.path.join(model_dir, f)) for k, f in wm.items()]
    else:
        f = os.path.join(model_dir, "model.safetensors")
        with safe_open(f, framework="pt") as fh:
            pairs = [(k, f) for k in fh.keys()]
    out = []
    keys = {k for k, _ in pairs}
    head = "lm_head.weight" if "lm_head.weight" in keys else "model.embed_tokens.weight"   # tied head as analyze.py
    for key, path in pairs:
        if key == head:
            out.append((key, path, -1, "lm_head"))
            continue
        if not key.startswith("model.layers."):
            continue
        layer = int(key.split(".")[2])
        if layers is not None and layer not in layers:
            continue
        typ = next((t for t, s in TYPES if s in key), None)
        if typ is None or not key.endswith(".weight"):
            continue
        out.append((key, path, layer, typ))
    return sorted(out, key=lambda x: (x[2], x[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("tag")
    ap.add_argument("out")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    self_check()
    if a.check:
        return
    layers = None if a.layers == "all" else {int(x) for x in a.layers.split(",")}
    ts = tasks(a.model_dir, layers)
    print("tensors", len(ts), flush=True)
    t0 = time.time()
    with open(a.out, "w") as fo, cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, res in enumerate(ex.map(work, ts)):
            rec = {"model": a.tag, "key": res["key"], "layer": res["layer"], "type": res["type"], **res["recs"]}
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            print(f"[{i + 1}/{len(ts)}] {res['key']} {round(time.time() - t0, 1)}s", flush=True)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
