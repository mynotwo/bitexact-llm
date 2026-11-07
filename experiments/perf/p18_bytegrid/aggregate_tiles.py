"""p18c: aggregate tile_grid.py JSON-lines into per-model, per-tile-shape tables.

usage: aggregate_tiles.py IN_JSONL OUT_JSON
Linear tensors only (lm_head reported separately). Fields per tile shape: mean limbs and products (x3), fraction of
tiles at L=2 and L=3, fraction over 23 bits, residual density at the chosen L, side data 8/(Nb*Kb) bits per weight,
distinct g and distinct (g mod 8) per tile-row-group (mean, p50, p90, p99, max), accumulators 5 x distinct(g mod 8)
(mean, p99, max), largest K-span in one (g mod 8) class, and int32 headroom 2^31 / (maxK x 3 x 128 x 128).
"""
import json
import sys
from collections import defaultdict

INT32 = 2 ** 31


def pct(hist, q):
    total = sum(hist)
    target = q * total
    c = 0
    for v, n in enumerate(hist):
        c += n
        if c >= target:
            return v
    return len(hist) - 1


def new_acc():
    return defaultdict(int, {"cnt": [0, 0, 0], "hist_dg": None, "hist_dm": None, "maxK": 0})


def add(acc, b):
    for k in ("n_entries", "n_tiles", "nnz", "lit_sum", "over23", "res", "n_rowgroups", "sum_dg", "sum_dm"):
        acc[k] += b[k]
    for i in range(3):
        acc["cnt"][i] += b["cnt"][i]
    acc["hist_dg"] = [x + y for x, y in zip(acc["hist_dg"] or [0] * len(b["hist_dg"]), b["hist_dg"])]
    acc["hist_dm"] = [x + y for x, y in zip(acc["hist_dm"] or [0] * len(b["hist_dm"]), b["hist_dm"])]
    acc["maxK"] = max(acc["maxK"], b["max_classK"])


def stats(acc, Nb, Kb):
    ne, nt, nrg = acc["n_entries"], acc["n_tiles"], acc["n_rowgroups"]
    mean_limbs = acc["lit_sum"] / ne
    hdm, hdg = acc["hist_dm"], acc["hist_dg"]
    p99dm = pct(hdm, 0.99)
    out = {"Nb": Nb, "Kb": Kb, "mean_limbs": mean_limbs, "products": 3 * mean_limbs,
           "frac_L2": acc["cnt"][1] / nt, "frac_L3": acc["cnt"][2] / nt, "frac_over23": acc["over23"] / nt,
           "resid_density": acc["res"] / acc["nnz"], "side_bits_per_weight": 8 / (Nb * Kb),
           "rowgroups": nrg,
           "distinct_g_mean": acc["sum_dg"] / nrg, "distinct_g_p50": pct(hdg, 0.5), "distinct_g_p90": pct(hdg, 0.9),
           "distinct_g_p99": pct(hdg, 0.99), "distinct_g_max": max(v for v, n in enumerate(hdg) if n),
           "distinct_gm8_mean": acc["sum_dm"] / nrg, "distinct_gm8_p50": pct(hdm, 0.5),
           "distinct_gm8_p90": pct(hdm, 0.9), "distinct_gm8_p99": p99dm,
           "distinct_gm8_max": max(v for v, n in enumerate(hdm) if n),
           "acc_mean": 5 * acc["sum_dm"] / nrg, "acc_p99": 5 * p99dm,
           "acc_max": 5 * max(v for v, n in enumerate(hdm) if n),
           "max_classK": acc["maxK"],
           "int32_headroom": INT32 / (acc["maxK"] * 3 * 128 * 128) if acc["maxK"] else None}
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    recs = [json.loads(l) for l in open(src)]
    result = {}
    for m in sorted({r["model"] for r in recs}):
        mr = [r for r in recs if r["model"] == m]
        result[m] = {}
        keys = [k for k in mr[0] if k.startswith("T")]
        for key in keys:
            lin = defaultdict(new_acc)
            head = new_acc()
            for r in mr:
                b = r[key]
                acc = head if r["type"] == "lm_head" else lin["all"]
                add(acc, b)
            Nb, Kb = mr[0][key]["Nb"], mr[0][key]["Kb"]
            result[m][key] = {"linear": stats(lin["all"], Nb, Kb), "lm_head": stats(head, Nb, Kb)}
    json.dump(result, open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
