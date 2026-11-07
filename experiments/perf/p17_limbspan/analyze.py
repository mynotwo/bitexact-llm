"""p17: per-block weight limb spans for exact mode (CPU only, no GPU).

Measures how many signed int8 limbs per bf16 weight a K-block needs when the grid exponent is chosen per
block instead of per output row. Uses the shipped bf16_parts/_bitlen/_lowbit/decompose from
src/bitexact/exact/consts.py, so the per-row C=23 cross-check runs the shipped code path.

Definitions (see RESULTS.md): per block, top = max(ex + bitlen(mag)), bot = min(ex + lowbit(mag)) over
nonzero entries, span = top - bot. For cap C in {7,15,23} an entry needs a residual iff ex + lowbit < top - C.
Limb choice per block (literal rule): smallest L in {1,2,3} whose residual count r_L <= tau * nnz_block.
Pooled rule: per tensor, minimize sum(L * n_block) subject to sum(r_L) <= tau * nnz_tensor (Lagrangian bisection).
"""
import argparse
import json
import concurrent.futures as cf
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

from bitexact.exact.consts import _bitlen, _lowbit, bf16_parts, decompose

BLOCKS = [0, 256, 128, 64, 32, 16, 8]   # 0 = whole output row (shipped scheme)
CAPS = {1: 7, 2: 15, 3: 23}             # limb count -> capacity in bits
TAUS = [1e-5, 1e-4, 1e-3]               # residual budgets: 1e-4 is the main threshold
HB = 128                                # span histogram bins
TYPES = [("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"),
         ("o", "self_attn.o_proj"), ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"), ("down", "mlp.down_proj")]


def load_tensor(model_dir, name):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        fname = json.load(open(idx))["weight_map"][name]
    else:
        fname = "model.safetensors"
    with safe_open(os.path.join(model_dir, fname), framework="pt") as f:
        return f.get_tensor(name)


def tensor_names(model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    keys = json.load(open(idx))["weight_map"] if os.path.exists(idx) else None
    if keys is None:
        with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
            keys = {k: 1 for k in f.keys()}
    return keys


def block_arrays(W, B):
    """Per-block arrays for block size B (0 = whole row): nnz, span (-1 if empty), residual counts r1/r2/r3."""
    N, K = W.shape
    bs = K if B == 0 else B
    assert K % bs == 0, (K, bs)
    nb = K // bs
    acc = {k: [] for k in ("nnz", "span", "r1", "r2", "r3")}
    CH = max(1, (1 << 22) // K)
    for r0 in range(0, N, CH):
        w = W[r0:r0 + CH].contiguous()
        n_ = w.shape[0]
        sm, ex = bf16_parts(w)
        mag = sm.abs()
        nz = mag > 0
        top = torch.where(nz, ex + _bitlen(mag), torch.full_like(ex, -10 ** 6)).view(n_, nb, bs)
        low = torch.where(nz, ex + _lowbit(mag), torch.full_like(ex, 10 ** 6)).view(n_, nb, bs)
        nzv = nz.view(n_, nb, bs)
        tb = top.amax(2)
        lb = low.amin(2)
        nnz = nzv.sum(2)
        span = torch.where(nnz > 0, tb - lb, torch.full_like(tb, -1))
        acc["nnz"].append(nnz.flatten().numpy())
        acc["span"].append(span.flatten().numpy())
        for c, cap in CAPS.items():
            res = (low < (tb - cap)[..., None]) & nzv
            acc[f"r{c}"].append(res.sum(2).flatten().numpy())
    return {k: np.concatenate(v).astype(np.int64) for k, v in acc.items()}, bs


def literal_choice(A, tau):
    allow = tau * A["nnz"].astype(np.float64)
    return np.where(A["r1"] <= allow, 1, np.where(A["r2"] <= allow, 2, 3))


def pooled_choice(A, tau, bs):
    """Minimize sum(L*bs) s.t. sum(r_L) <= tau*nnz_total. Returns (L array, feasible flag)."""
    R = [A["r1"].astype(np.float64), A["r2"].astype(np.float64), A["r3"].astype(np.float64)]
    budget = tau * float(A["nnz"].sum())

    def choose(lam):
        vals = np.stack([bs * (i + 1) + lam * R[i] for i in range(3)])
        idx = vals.argmin(0)
        res = float(np.where(idx == 0, R[0], np.where(idx == 1, R[1], R[2])).sum())
        return idx, res

    idx, res = choose(1e-6)                     # cost-dominated: all L=1
    if res <= budget:
        return idx + 1, True
    lo, hi = np.log(1e-6), np.log(1e12)
    idx_hi, res_hi = choose(np.exp(hi))
    if res_hi > budget:                         # infeasible even at 3 limbs everywhere
        return np.full(len(A["nnz"]), 3), False
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        idx_m, res_m = choose(np.exp(mid))
        if res_m <= budget:
            hi, idx_hi = mid, idx_m
        else:
            lo = mid
    return idx_hi + 1, True


def task(args):
    model_dir, layer, typ, name = args
    torch.set_num_threads(2)
    t0 = time.time()
    out = {"layer": layer, "type": typ, "name": name}
    W = load_tensor(model_dir, name)
    assert W.dtype == torch.bfloat16 and W.dim() == 2, (name, W.dtype, W.shape)
    N, K = W.shape
    out["shape"] = [N, K]
    out["n_entries"] = N * K
    out["nnz"] = int((W != 0).sum())
    dec = decompose(W)["stats"]               # shipped code path, whole-row grid
    out["shipped"] = {"nnz_res": dec["nnz_res"], "same_val": dec["same_val"], "same_bits": dec["same_bits"],
                      "max_d": dec["max_d"], "rows_res": dec["rows_res"]}
    for B in BLOCKS:
        A, bs = block_arrays(W, B)
        rec = {"n_blocks": int(len(A["nnz"]))}
        ne = A["span"] >= 0
        span = A["span"][ne]
        rec["span_hist"] = np.bincount(np.clip(span, 0, HB - 1), minlength=HB).tolist()
        rec["r"] = {str(c): int(A[f"r{c}"].sum()) for c in CAPS}
        rec["nnz"] = int(A["nnz"].sum())
        for tau in TAUS:
            key = f"{tau:g}"
            L = literal_choice(A, tau)
            rec[f"lit_sum_{key}"] = int((L * bs).sum())
            rec[f"lit_count_{key}"] = [int((L == k).sum()) for k in (1, 2, 3)]
            rec[f"lit_over_{key}"] = int((A["r3"] > tau * A["nnz"]).sum())
            Lp, feas = pooled_choice(A, tau, bs)
            rec[f"pool_sum_{key}"] = int((Lp * bs).sum())
            rec[f"pool_feasible_{key}"] = bool(feas)
        out[f"B{bs}"] = rec
    out["sec"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model name tag, e.g. llama32_1b")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-head", action="store_true", help="skip lm_head")
    ap.add_argument("--head-only", action="store_true", help="only lm_head")
    a = ap.parse_args()
    keys = tensor_names(a.model_dir)
    n_layers = 1 + max(int(k.split(".")[2]) for k in keys if k.startswith("model.layers."))
    layers = list(range(n_layers)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    tasks = []
    for L in layers:
        for typ, suf in TYPES:
            tasks.append((a.model_dir, L, typ, f"model.layers.{L}.{suf}.weight"))
    head = "lm_head.weight" if "lm_head.weight" in keys else "model.embed_tokens.weight"
    if a.head_only:
        tasks = []
    if not a.no_head:
        tasks.append((a.model_dir, -1, "lm_head", head))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.model}.jsonl")
    done = 0
    mode = "a" if a.head_only else "w"
    with open(path, mode) as fo, cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(task, tasks):
            r["model"] = a.model
            r["tied_lm_head"] = head != "lm_head.weight"
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            done += 1
            print(f"[{done}/{len(tasks)}] L{r['layer']} {r['type']} {r['shape']} {r['sec']}s "
                  f"shipped_nnz_res={r['shipped']['nnz_res']} same_val={r['shipped']['same_val']}", flush=True)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
