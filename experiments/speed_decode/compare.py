"""Bitwise comparison of two rollout/score files (rl_mismatch/common.py format) over their common seeds.

  python compare.py REF.json X.json [--name NAME] [--out results/NAME.json] [--seeds-from FILE]

Per sequence: tokens identical? first divergence. Per token, over the positions where both files score the same
token history (the common prefix up to and including the first differing token's position is excluded after it):
lp bitwise equal?, |d| = |lp_x - lp_ref|, sha16 equal (when both have it). For score files (same input tokens)
the prefix is the whole sequence.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rl_mismatch"))
import common  # noqa: E402


def compare(ref, x, keep=None):
    a, b = common.load(ref)["seqs"], common.load(x)["seqs"]
    ks = sorted(k for k in set(a) & set(b) if keep is None or k in keep)
    seq_same, n_tok, n_lp_diff, n_sha, n_sha_diff, sum_d, max_d, n_tok_ref, first = 0, 0, 0, 0, 0, 0.0, 0.0, 0, []
    for k in ks:
        ra, rb = a[k], b[k]
        ta, tb = ra.get("tok"), rb.get("tok")
        if ta is None or tb is None:          # score file without tokens: same inputs by construction
            n = min(len(ra["lp"]), len(rb["lp"]))
            fd = None
        else:
            fd = common.first_div(ta, tb)
            n = len(ta) if fd is None else fd + 1   # position fd: same history, different token chosen
        seq_same += fd is None and len(ra["lp"]) == len(rb["lp"])
        if fd is not None:
            first.append(fd)
        n_tok_ref += len(ra["lp"])
        for i in range(n):
            if ta is not None and tb is not None and i == fd:
                continue                       # different tokens: lp of different events, not compared
            d = abs(rb["lp"][i] - ra["lp"][i])
            n_tok += 1
            n_lp_diff += rb["lp"][i] != ra["lp"][i]
            sum_d += d
            max_d = max(max_d, d)
            if "sha" in ra and "sha" in rb:
                n_sha += 1
                n_sha_diff += ra["sha"][i] != rb["sha"][i]
    first.sort()
    return {"ref": ref, "x": x, "seqs": len(ks), "seqs_identical": seq_same, "tokens_compared": n_tok,
            "tokens_ref": n_tok_ref, "lp_differ": n_lp_diff, "lp_differ_frac": n_lp_diff / max(n_tok, 1),
            "mean_abs_d": sum_d / max(n_tok, 1), "max_abs_d": max_d, "sha_compared": n_sha, "sha_differ": n_sha_diff,
            "median_first_div": first[len(first) // 2] if first else None, "n_diverged": len(first)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("x")
    ap.add_argument("--name")
    ap.add_argument("--out")
    ap.add_argument("--seeds-from")
    a = ap.parse_args()
    keep = set(int(s) for s in open(a.seeds_from).read().split()) if a.seeds_from else None
    r = compare(a.ref, a.x, keep)
    r["name"] = a.name
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
    print(f"CMP {a.name}: seqs {r['seqs_identical']}/{r['seqs']} identical, tokens {r['tokens_compared']}, "
          f"lp differ {r['lp_differ']} ({100 * r['lp_differ_frac']:.2f}%), mean|d| {r['mean_abs_d']:.3g}, "
          f"max|d| {r['max_abs_d']:.3g}, sha differ {r['sha_differ']}/{r['sha_compared']}")


if __name__ == "__main__":
    main()
