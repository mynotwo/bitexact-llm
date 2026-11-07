"""p18: aggregate analyze.py JSON-lines (free grid m=1, 4-aligned g4_, 8-aligned g8_) into one CSV.

usage: summarize_grid.py OUT_DIR MODEL [MODEL ...]   (reads OUT_DIR/MODEL.jsonl, writes OUT_DIR/MODEL_grid.csv)
Rules per grid: "exact" = tau 0 (zero residual; smallest L with r_L == 0, else 3 with residual reported),
"lit" = p17 literal rule at tau 1e-4. Counts are additive across tensors; ratios are recomputed from sums.
"""
import csv
import json
import os
import sys

GRIDS = [("free", ""), ("g4", "g4_"), ("g8", "g8_")]
TAUS = ["0", "1e-05", "0.0001", "0.001"]
LABELS = ["row", "256", "128", "64", "32", "16", "8"]
TYPES = ["q", "k", "v", "o", "gate", "up", "down", "lm_head"]


def bkey(rec, label):
    return f"B{rec['shape'][1]}" if label == "row" else f"B{label}"


def empty():
    d = {"n_blocks": 0, "nnz": 0, "n_entries": 0, "botmod8": [0] * 8, "nonempty": 0, "n_tensors": 0}
    for g, _ in GRIDS:
        for t in TAUS:
            d[f"{g}_sum_{t}"] = 0
            d[f"{g}_res_{t}"] = 0
            d[f"{g}_over_{t}"] = 0
            d[f"{g}_cnt_{t}"] = [0, 0, 0]
    return d


def add(acc, rec, label):
    b = rec[bkey(rec, label)]
    acc["n_tensors"] += 1
    acc["n_blocks"] += b["n_blocks"]
    acc["nnz"] += b["nnz"]
    acc["n_entries"] += rec["n_entries"]
    acc["botmod8"] = [x + y for x, y in zip(acc["botmod8"], b["botmod8"])]
    acc["nonempty"] += sum(b["botmod8"])
    for g, pre in GRIDS:
        for t in TAUS:
            acc[f"{g}_sum_{t}"] += b[f"{pre}lit_sum_{t}"]
            acc[f"{g}_res_{t}"] += b[f"{pre}lit_res_{t}"]
            acc[f"{g}_over_{t}"] += b[f"{pre}lit_over_{t}"]
            acc[f"{g}_cnt_{t}"] = [x + y for x, y in zip(acc[f"{g}_cnt_{t}"], b[f"{pre}lit_count_{t}"])]


def metrics(acc):
    out = {"n_blocks": acc["n_blocks"], "n_entries": acc["n_entries"], "nnz": acc["nnz"]}
    nb = acc["n_blocks"]
    ne = acc["nonempty"]
    for k in range(8):
        out[f"botmod8_{k}"] = acc["botmod8"][k] / ne if ne else None
    out["botmod8_mean"] = sum(k * acc["botmod8"][k] for k in range(8)) / ne if ne else None
    for g, _ in GRIDS:
        for t in TAUS:
            cnt = acc[f"{g}_cnt_{t}"]
            out[f"{g}_t{t}_Lbar"] = acc[f"{g}_sum_{t}"] / acc["n_entries"]
            out[f"{g}_t{t}_prod"] = 3 * out[f"{g}_t{t}_Lbar"]
            out[f"{g}_t{t}_fL1"] = cnt[0] / nb
            out[f"{g}_t{t}_fL2"] = cnt[1] / nb
            out[f"{g}_t{t}_fL3"] = cnt[2] / nb
            out[f"{g}_t{t}_fover23"] = acc[f"{g}_over_{t}"] / nb
            out[f"{g}_t{t}_dens"] = acc[f"{g}_res_{t}"] / acc["nnz"]
    return out


def main():
    out_dir = sys.argv[1]
    for model in sys.argv[2:]:
        recs = [json.loads(l) for l in open(os.path.join(out_dir, f"{model}.jsonl"))]
        rows = []
        for label in LABELS:
            for typ in TYPES:
                acc = empty()
                for r in recs:
                    if r["type"] == typ:
                        add(acc, r, label)
                rows.append({"model": model, "group": typ, "block": label, **metrics(acc)})
            acc = empty()
            for r in recs:
                if r["type"] != "lm_head":
                    add(acc, r, label)
            rows.append({"model": model, "group": "all_no_head", "block": label, **metrics(acc)})
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        path = os.path.join(out_dir, f"{model}_grid.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print("wrote", path, len(rows))


if __name__ == "__main__":
    main()
