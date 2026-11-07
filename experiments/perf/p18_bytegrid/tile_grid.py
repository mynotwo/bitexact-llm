"""p18c: free grid chosen per 2D tile (Nb output rows x Kb weights along K), literal rule tau = 1e-4.

Per tile: top = max(ex + bitlen(mag)), bot = min(ex + lowbit(mag)) over nonzero entries, grid g = top - C.
Residual at capacity C = entries with ex + lowbit < top - C. Chosen L = smallest of {1,2,3} whose residual count
is <= tau x nnz(tile) (same rule as p17 analyze.py). Mean limbs weight each tile by all Nb x Kb elements.

Per tile-row-group (the Nb rows of one row-block, across all K-tiles): distinct g, distinct (g mod 8), and the
largest K-span in one (g mod 8) class. The 1D case Nb = 1 reproduces p18b's per-row counts (cross-check).

usage: tile_grid.py MODEL_DIR TAG OUT_JSONL [--workers N] [--check]
"""
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bitexact.exact.consts import _bitlen, _lowbit, bf16_parts  # noqa: E402

TILES = [(1, 32), (1, 64), (1, 128),
         (128, 32), (128, 64), (128, 128), (64, 32), (64, 64), (64, 128), (32, 32), (32, 64), (16, 32)]
TAU = 1e-4
CAP = 23
SENT = 10 ** 9
DG_CLIP = 64
DM_CLIP = 8
TYPES = [("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"),
         ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"), ("down", "mlp.down_proj")]


def distinct(vals, valid):
    s = torch.where(valid, vals, torch.full_like(vals, SENT)).sort(dim=1).values
    v = s != SENT
    newv = torch.cat([v[:, :1], v[:, 1:] & (s[:, 1:] != s[:, :-1])], 1)
    return newv.sum(1)


def new_rec(Nb, Kb):
    return {"Nb": Nb, "Kb": Kb, "n_entries": 0, "n_tiles": 0, "nnz": 0, "lit_sum": 0, "cnt": [0, 0, 0],
            "over23": 0, "res": 0, "n_rowgroups": 0, "hist_dg": [0] * (DG_CLIP + 1), "sum_dg": 0,
            "hist_dm": [0] * (DM_CLIP + 1), "sum_dm": 0, "max_classK": 0}


def tensor_stats(W):
    R_all, K = W.shape
    sm, ex = bf16_parts(W)
    mag = sm.abs()
    nz = mag > 0
    top_e = torch.where(nz, ex + _bitlen(mag), torch.full_like(ex, -10 ** 6))
    low_e = torch.where(nz, ex + _lowbit(mag), torch.full_like(ex, 10 ** 6))
    recs = {}
    CH = 256 if R_all % 256 == 0 else 128
    for Nb, Kb in TILES:
        if K % Kb or R_all % Nb:
            continue
        rec = new_rec(Nb, Kb)
        for r0 in range(0, R_all, CH):
            sl = slice(r0, r0 + CH)
            tp, lw, z = top_e[sl], low_e[sl], nz[sl]
            R = tp.shape[0]
            if R % Nb:
                raise ValueError("row chunk not divisible by Nb")
            nrb, nkb = R // Nb, K // Kb
            tv = tp.view(nrb, Nb, nkb, Kb)
            lv = lw.view(nrb, Nb, nkb, Kb)
            zv = z.view(nrb, Nb, nkb, Kb)
            tops = tv.amax(dim=(1, 3))                    # (nrb, nkb)
            nnz = zv.sum(dim=(1, 3))
            has = nnz > 0
            r = {}
            for C in (7, 15, 23):
                gc = tops - C
                r[C] = ((lv < gc[:, None, :, None]) & zv).sum(dim=(1, 3))
            allow = TAU * nnz.double()
            L = torch.where(r[7].double() <= allow, 1, torch.where(r[15].double() <= allow, 2, 3))
            elems = Nb * Kb
            rec["n_entries"] += R * K
            rec["n_tiles"] += nrb * nkb
            rec["nnz"] += int(nnz.sum())
            rec["lit_sum"] += int(L.sum()) * elems
            for k in range(3):
                rec["cnt"][k] += int((L == k + 1).sum())
            rec["over23"] += int((r[23].double() > allow).sum())
            res_ch = torch.where(L == 1, r[7], torch.where(L == 2, r[15], r[23]))
            rec["res"] += int(res_ch.sum())
            # per tile-row-group (row-block) statistics over K-tiles
            g = tops - CAP
            dg = distinct(g, has)
            dm = distinct(torch.remainder(g, 8), has)
            rec["n_rowgroups"] += nrb
            rec["hist_dg"] = [a + b for a, b in zip(rec["hist_dg"], torch.bincount(dg.clamp(max=DG_CLIP), minlength=DG_CLIP + 1).tolist())]
            rec["sum_dg"] += int(dg.sum())
            rec["hist_dm"] = [a + b for a, b in zip(rec["hist_dm"], torch.bincount(dm.clamp(max=DM_CLIP), minlength=DM_CLIP + 1).tolist())]
            rec["sum_dm"] += int(dm.sum())
            oh = torch.nn.functional.one_hot(torch.remainder(g, 8).clamp(0, 7), 8).to(torch.int64)
            oh = oh * has.unsqueeze(-1)
            classK = oh.sum(1) * Kb                        # (nrb, 8): K positions per (g mod 8) class
            rec["max_classK"] = max(rec["max_classK"], int(classK.max()))
        recs[f"T{Nb}x{Kb}"] = rec
    return recs


def brute(W, Nb, Kb):
    """Plain-Python reference for one tile shape: returns the counted fields."""
    sm, ex = bf16_parts(W)
    sm, ex = sm.numpy(), ex.numpy()
    R, K = sm.shape
    res = {"n_tiles": 0, "lit_sum": 0, "cnt": [0, 0, 0], "over23": 0, "res": 0, "dg": [], "dm": [], "maxK": 0}
    bl = lambda x: int(x).bit_length()
    lb = lambda x: (int(x) & -int(x)).bit_length() - 1
    for rb in range(R // Nb):
        gs = []
        cls = {}
        for kb in range(K // Kb):
            entries = []
            for i in range(rb * Nb, (rb + 1) * Nb):
                for c in range(kb * Kb, (kb + 1) * Kb):
                    if sm[i, c] != 0:
                        entries.append((ex[i, c] + bl(abs(sm[i, c])), ex[i, c] + lb(abs(sm[i, c]))))
            res["n_tiles"] += 1
            nnz = len(entries)
            if nnz == 0:                       # empty tile: L = 1 and zero residual, as in the vectorized path
                res["lit_sum"] += Nb * Kb
                res["cnt"][0] += 1
                continue
            top = max(t for t, _ in entries)
            rc = {C: sum(1 for _, l in entries if l < top - C) for C in (7, 15, 23)}
            allow = TAU * nnz
            L = 1 if rc[7] <= allow else (2 if rc[15] <= allow else 3)
            res["lit_sum"] += L * Nb * Kb
            res["cnt"][L - 1] += 1
            res["over23"] += 1 if rc[23] > allow else 0
            res["res"] += rc[{1: 7, 2: 15, 3: 23}[L]]
            gs.append(top - CAP)
            cls[(top - CAP) % 8] = cls.get((top - CAP) % 8, 0) + Kb
        res["dg"].append(len(set(gs)))
        res["dm"].append(len({g % 8 for g in gs}))
        res["maxK"] = max(res["maxK"], max(cls.values()) if cls else 0)
    return res


def self_check():
    torch.manual_seed(2)
    R, K = 128, 128
    W = (torch.randn(R, K) * torch.exp2(torch.randint(-9, 5, (R, K)).float())).to(torch.bfloat16)
    W[3, :40] = 0
    W[5:9, 64:96] = 0
    W[32:64, 0:32] = 0
    fast = tensor_stats(W)
    bad = 0
    for Nb, Kb in [(1, 32), (16, 32), (64, 32), (32, 64), (128, 128)]:
        f = fast[f"T{Nb}x{Kb}"]
        ref = brute(W, Nb, Kb)
        pairs = [("n_tiles", f["n_tiles"], ref["n_tiles"]), ("lit_sum", f["lit_sum"], ref["lit_sum"]),
                 ("cnt1", f["cnt"][0], ref["cnt"][0]), ("cnt2", f["cnt"][1], ref["cnt"][1]),
                 ("cnt3", f["cnt"][2], ref["cnt"][2]), ("over23", f["over23"], ref["over23"]),
                 ("res", f["res"], ref["res"]), ("sum_dg", f["sum_dg"], sum(ref["dg"])),
                 ("sum_dm", f["sum_dm"], sum(ref["dm"])), ("maxK", f["max_classK"], ref["maxK"])]
        for name, a, b in pairs:
            if a != b:
                bad += 1
                print("SELFCHECK MISMATCH", Nb, Kb, name, a, b, flush=True)
    print("selfcheck mismatches:", bad, flush=True)
    if bad:
        sys.exit(1)


def work(args):
    key, path, layer, typ = args
    torch.set_num_threads(1)
    with safe_open(path, framework="pt") as fh:
        W = fh.get_tensor(key).to(torch.bfloat16)
    return {"key": key, "layer": layer, "type": typ, "recs": tensor_stats(W)}


def tasks(model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        pairs = [(k, os.path.join(model_dir, f)) for k, f in wm.items()]
    else:
        f = os.path.join(model_dir, "model.safetensors")
        with safe_open(f, framework="pt") as fh:
            pairs = [(k, f) for k in fh.keys()]
    keys = {k for k, _ in pairs}
    head = "lm_head.weight" if "lm_head.weight" in keys else "model.embed_tokens.weight"
    out = []
    for key, path in pairs:
        if key == head:
            out.append((key, path, -1, "lm_head"))
        elif key.startswith("model.layers.") and key.endswith(".weight"):
            typ = next((t for t, s in TYPES if s in key), None)
            if typ:
                out.append((key, path, int(key.split(".")[2]), typ))
    return sorted(out, key=lambda x: (x[2], x[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("tag")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    self_check()
    if a.check:
        return
    ts = tasks(a.model_dir)
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
