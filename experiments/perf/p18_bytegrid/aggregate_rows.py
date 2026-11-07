"""p18b: aggregate free_rows.py JSON-lines into per-model tables.

usage: aggregate_rows.py IN_JSONL OUT_JSON
Per model and block size: over all linear tensors (lm_head separate): distinct g per row (mean, p50, p90, p99,
max, distribution 0..12 and 13+), distinct (g mod 8) per row, implied accumulator count 5 x distinct(g mod 8),
max K-positions in one class (int32 bound), by type, and by layer.
"""
import json
import sys
from collections import defaultdict

DG_CLIP = 64
DM_CLIP = 8
BLOCKS = ["B256", "B128", "B64", "B32", "B16", "B8"]
INT32 = 2 ** 31
PROD_MAX = 127 * 127
DIAG_MULT = 3          # products landing on the same limb diagonal from one k (max, diagonal i+j = 2)


def empty():
    return {"N": 0, "sum_dg": 0, "max_dg": 0, "hist_dg": [0] * (DG_CLIP + 1), "sum_dm": 0, "max_dm": 0,
            "hist_dm": [0] * (DM_CLIP + 1), "max_class_k": 0, "sum_max_class_k": 0}


def add(acc, b):
    acc["N"] += b["N"]
    acc["sum_dg"] += b["sum_dg"]
    acc["max_dg"] = max(acc["max_dg"], b["max_dg"])
    acc["hist_dg"] = [x + y for x, y in zip(acc["hist_dg"], b["hist_dg"])]
    acc["sum_dm"] += b["sum_dm"]
    acc["max_dm"] = max(acc["max_dm"], b["max_dm"])
    acc["hist_dm"] = [x + y for x, y in zip(acc["hist_dm"], b["hist_dm"])]
    acc["max_class_k"] = max(acc["max_class_k"], b["max_class_k"])
    acc["sum_max_class_k"] += b["sum_max_class_k"]


def pct(hist, q):
    total = sum(hist)
    target = q * total
    c = 0
    for v, n in enumerate(hist):
        c += n
        if c >= target:
            return v
    return len(hist) - 1


def summarize(acc, mult=1, dm=False):
    h = acc["hist_dm"] if dm else acc["hist_dg"]
    N = acc["N"]
    out = {"rows": N, "mean": (acc["sum_dm"] if dm else acc["sum_dg"]) / N * mult,
           "p50": pct(h, 0.5) * mult, "p90": pct(h, 0.9) * mult, "p99": pct(h, 0.99) * mult,
           "max": (acc["max_dm"] if dm else acc["max_dg"]) * mult}
    if not dm:
        out["dist_0_12"] = [round(x / N, 5) for x in h[:13]]
        out["frac_13plus"] = round(sum(h[13:]) / N, 5)
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    recs = [json.loads(l) for l in open(src)]
    models = sorted({r["model"] for r in recs})
    result = {}
    for m in models:
        mr = [r for r in recs if r["model"] == m]
        result[m] = {}
        for B in BLOCKS:
            allacc = empty()
            head = empty()
            bytype = defaultdict(empty)
            bylayer = defaultdict(empty)
            kmax = 0
            for r in mr:
                b = r[B]
                if r["type"] == "lm_head":
                    add(head, b)
                    continue
                add(allacc, b)
                add(bytype[r["type"]], b)
                add(bylayer[r["layer"]], b)
                kmax = max(kmax, b["max_class_k"])
            K_ = max(r[B]["K"] for r in mr)
            d = {"all_linear": {"distinct_g": summarize(allacc), "distinct_gmod8": summarize(allacc, dm=True),
                                "acc_count_5x": summarize(allacc, mult=5, dm=True),
                                "max_class_k": allacc["max_class_k"], "sum_max_class_k": allacc["sum_max_class_k"]},
                 "lm_head": {"distinct_g": summarize(head), "distinct_gmod8": summarize(head, dm=True),
                             "acc_count_5x": summarize(head, mult=5, dm=True), "max_class_k": head["max_class_k"]},
                 "by_type": {t: {"distinct_g_mean": round(summarize(a)["mean"], 3),
                                 "distinct_gmod8_mean": round(summarize(a, dm=True)["mean"], 3)}
                             for t, a in sorted(bytype.items())},
                 "by_layer": {str(l): {"distinct_g_mean": round(summarize(a)["mean"], 3),
                                       "distinct_gmod8_mean": round(summarize(a, dm=True)["mean"], 3)}
                              for l, a in sorted(bylayer.items())}}
            # int32 headroom: worst class sums k * 3 diagonal products * 127^2 (coordinator form: k * 127^2)
            d["int32"] = {"K_max": K_, "max_class_k": kmax, "worst_partial_coord_form": kmax * PROD_MAX,
                          "worst_partial_with_diag3": kmax * DIAG_MULT * PROD_MAX, "limit_2_31": INT32,
                          "headroom_x": INT32 / (kmax * DIAG_MULT * PROD_MAX) if kmax else None}
            result[m][B] = d
    json.dump(result, open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
