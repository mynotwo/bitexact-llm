"""p17: aggregate analyze.py JSON-lines output into CSV tables and print the headline numbers.

usage: summarize.py OUT_DIR MODEL [MODEL ...]   (reads OUT_DIR/MODEL.jsonl, writes OUT_DIR/MODEL_*.csv)
All counts are additive across tensors, so aggregation sums counts and recomputes ratios; span quantiles come
from the summed span histograms (integer bins, exact).
"""
import csv
import json
import os
import sys

TAUS = ["1e-05", "0.0001", "0.001"]
LABELS = ["row", "256", "128", "64", "32", "16", "8"]
TYPES = ["q", "k", "v", "o", "gate", "up", "down", "lm_head"]
HB = 128


def block_key(rec, label):
    K = rec["shape"][1]
    return f"B{K}" if label == "row" else f"B{label}"


def empty_acc():
    return {"n_blocks": 0, "nonempty": 0, "hist": [0] * HB, "nnz": 0, "n_entries": 0,
            "r7": 0, "r15": 0, "r23": 0, "lit": {t: 0 for t in TAUS}, "pool": {t: 0 for t in TAUS},
            "lit_count": {t: [0, 0, 0] for t in TAUS}, "lit_over": {t: 0 for t in TAUS},
            "pool_feasible": {t: True for t in TAUS}, "n_tensors": 0, "shipped_nnz_res": 0,
            "shipped_nnz": 0}


def add(acc, rec, label):
    b = rec[block_key(rec, label)]
    acc["n_tensors"] += 1
    acc["n_blocks"] += b["n_blocks"]
    h = b["span_hist"]
    acc["hist"] = [x + y for x, y in zip(acc["hist"], h)]
    acc["nonempty"] += sum(h)
    acc["nnz"] += b["nnz"]
    acc["n_entries"] += rec["n_entries"]
    # analyze.py keys r by limb count: "1" = cap 7 bits, "2" = cap 15, "3" = cap 23
    acc["r7"] += b["r"]["1"]
    acc["r15"] += b["r"]["2"]
    acc["r23"] += b["r"]["3"]
    for t in TAUS:
        acc["lit"][t] += b[f"lit_sum_{t}"]
        acc["pool"][t] += b[f"pool_sum_{t}"]
        acc["lit_over"][t] += b[f"lit_over_{t}"]
        acc["lit_count"][t] = [x + y for x, y in zip(acc["lit_count"][t], b[f"lit_count_{t}"])]
        acc["pool_feasible"][t] = acc["pool_feasible"][t] and b[f"pool_feasible_{t}"]
    if label == "row":
        acc["shipped_nnz_res"] += rec["shipped"]["nnz_res"]
        acc["shipped_nnz"] += rec["nnz"]


def quantile(hist, q):
    total = sum(hist)
    if total == 0:
        return None
    target = q * total
    c = 0
    for v, n in enumerate(hist):
        c += n
        if c >= target:
            return v
    return len(hist) - 1


def metrics(acc):
    h, ne = acc["hist"], acc["nonempty"]
    cum = lambda x: sum(h[: x + 1]) / ne if ne else None
    maxspan = max((v for v, n in enumerate(h) if n), default=None)
    out = {"n_blocks": acc["n_blocks"], "n_entries": acc["n_entries"], "nnz": acc["nnz"],
           "span_p50": quantile(h, 0.5), "span_p90": quantile(h, 0.9), "span_p99": quantile(h, 0.99),
           "span_max": maxspan, "frac_span_le7": cum(7), "frac_span_le15": cum(15), "frac_span_le23": cum(23),
           "dens_C7": acc["r7"] / acc["nnz"], "dens_C15": acc["r15"] / acc["nnz"],
           "dens_C23": acc["r23"] / acc["nnz"]}
    for t in TAUS:
        lb = acc["lit"][t] / acc["n_entries"]
        pb = acc["pool"][t] / acc["n_entries"]
        out[f"lit_Lbar_{t}"] = lb
        out[f"lit_prod_{t}"] = 3 * lb
        out[f"lit_ratio_{t}"] = lb / 3
        cnt = acc["lit_count"][t]
        out[f"lit_frac_L1_{t}"] = cnt[0] / acc["n_blocks"]
        out[f"lit_frac_L2_{t}"] = cnt[1] / acc["n_blocks"]
        out[f"lit_frac_L3_{t}"] = cnt[2] / acc["n_blocks"]
        out[f"lit_over_{t}"] = acc["lit_over"][t] / acc["n_blocks"]
        out[f"pool_Lbar_{t}"] = pb
        out[f"pool_prod_{t}"] = 3 * pb
        out[f"pool_ratio_{t}"] = pb / 3
        out[f"pool_feasible_{t}"] = acc["pool_feasible"][t]
    if acc["shipped_nnz"]:
        out["shipped_dens_C23"] = acc["shipped_nnz_res"] / acc["shipped_nnz"]
    return out


def main():
    out_dir = sys.argv[1]
    for model in sys.argv[2:]:
        recs = [json.loads(l) for l in open(os.path.join(out_dir, f"{model}.jsonl"))]
        layers = sorted({r["layer"] for r in recs if r["layer"] >= 0})
        nl = len(layers)
        sel = {"first": 0, "middle": nl // 2, "last": nl - 1}
        rows = []
        # per layer x type x label
        for label in LABELS:
            for typ in TYPES:
                for lay in layers:
                    for r in recs:
                        if r["type"] == typ and r["layer"] == lay:
                            acc = empty_acc()
                            add(acc, r, label)
                            rows.append({"model": model, "layer": lay, "type": typ, "block": label, **metrics(acc)})
        # aggregates: all layers by type, and all types/all layers
        for label in LABELS:
            groups = {t: empty_acc() for t in TYPES}
            allacc = empty_acc()
            for r in recs:
                if r["layer"] < 0:
                    tgt = groups["lm_head"]
                    add(tgt, r, label)
                    continue
                add(groups[r["type"]], r, label)
                add(allacc, r, label)
            # headline excludes lm_head
            for typ in TYPES:
                rows.append({"model": model, "layer": "all", "type": typ, "block": label, **metrics(groups[typ])})
            # all linear layers of the transformer (lm_head reported separately)
            rows.append({"model": model, "layer": "all", "type": "all_layers_no_head", "block": label,
                         **metrics(allacc)})
            total = empty_acc()
            for typ in TYPES:
                for k in ("n_blocks", "nonempty", "nnz", "n_entries", "r7", "r15", "r23"):
                    total[k] += groups[typ][k]
                total["hist"] = [x + y for x, y in zip(total["hist"], groups[typ]["hist"])]
                for t in TAUS:
                    total["lit"][t] += groups[typ]["lit"][t]
                    total["pool"][t] += groups[typ]["pool"][t]
                    total["lit_over"][t] += groups[typ]["lit_over"][t]
                    total["lit_count"][t] = [x + y for x, y in zip(total["lit_count"][t], groups[typ]["lit_count"][t])]
                    total["pool_feasible"][t] = total["pool_feasible"][t] and groups[typ]["pool_feasible"][t]
                total["shipped_nnz_res"] += groups[typ]["shipped_nnz_res"]
                total["shipped_nnz"] += groups[typ]["shipped_nnz"]
            rows.append({"model": model, "layer": "all", "type": "all_types", "block": label, **metrics(total)})
        # layer subset (first / middle / last) aggregated over all types
        for label in LABELS:
            for pos, lay in sel.items():
                acc = empty_acc()
                for r in recs:
                    if r["layer"] == lay:
                        add(acc, r, label)
                rows.append({"model": model, "layer": f"{pos}:{lay}", "type": "all_types", "block": label,
                             **metrics(acc)})
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        path = os.path.join(out_dir, f"{model}_summary.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("wrote", path, len(rows), "rows")


if __name__ == "__main__":
    main()
