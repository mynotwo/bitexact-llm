"""Diff run_ident.py outputs against a reference: positions in common, logits rows and tokens that differ.

  python experiments/xhw_8b/compare.py ref.json other.json [more.json ...] [--out summary.json]

Runs must share the constants and workload SHA-256 (else the comparison is refused). A subset run is
compared on the positions it has; every one must also be in the reference. Also accepts the A18.2 research
files ({"rows": {"sid:t": [sha256[:24], token]}}): hashes are then compared on their common prefix (24 hex)
and tokens are not compared (A18.2 drew its Gumbel noise from another sampler).
"""

import argparse
import json
import os
import sys


def load(path):
    r = json.load(open(path))
    r.setdefault("schema", "a18")
    return r


def compare(ref, run):
    if run["schema"] != "a18" and ref["schema"] != "a18":
        for k in ("constants_sha256", "workload_sha256"):
            if ref[k] != run[k]:
                raise SystemExit(f"{k} differs: {ref[k][:16]} vs {run[k][:16]}")
    a, b = ref["rows"], run["rows"]
    common = sorted(set(a) & set(b), key=lambda k: tuple(map(int, k.split(":"))))
    missing = sorted(set(b) - set(a))
    legacy = (ref["schema"] == "a18") + (run["schema"] == "a18")
    ld = td = gd = 0
    first = None
    for k in common:
        x, y = a[k], b[k]
        n = min(len(x[0]), len(y[0]))
        bad_l = x[0][:n] != y[0][:n]
        bad_t = legacy == 0 and x[-1] != y[-1]    # A18.2 used another sampler: logits only
        bad_g = len(x) == 3 and len(y) == 3 and x[1] != y[1]
        ld, td, gd = ld + bad_l, td + bad_t, gd + bad_g
        if (bad_l or bad_t or bad_g) and first is None:
            first = k
    return {"positions": len(common), "not_in_ref": len(missing), "logits_differ": ld, "sampled_differ": td,
            "greedy_differ": gd, "first_diff": first, "mode": run.get("mode"), "device": run.get("device"),
            "env": run.get("env"), "sec": run.get("sec"), "sec_per_position": run.get("sec_per_position")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    ref = load(a.ref)
    res = {"ref": os.path.basename(a.ref), "ref_env": ref.get("env"), "runs": {}}
    for p in a.runs:
        res["runs"][os.path.basename(p)] = compare(ref, load(p))
    for name, r in res["runs"].items():
        print(f"{name:40s} {r['device'] or '':10s} {r['mode'] or '':6s} positions {r['positions']:4d}  "
              f"logits_differ {r['logits_differ']}  sampled_differ {r['sampled_differ']}  "
              f"greedy_differ {r['greedy_differ']}  not_in_ref {r['not_in_ref']}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    bad = any(r["logits_differ"] or r["sampled_differ"] or r["greedy_differ"] or r["not_in_ref"] or not r["positions"]
              for r in res["runs"].values())
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
