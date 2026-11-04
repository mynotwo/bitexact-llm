"""Bit-exactness gate: tokens, per-token log p and logits-row sha16 must be identical (dict equality, no tolerance).

  python gate.py BASE_TAG NEW_TAG [DIR]
compares NEW against BASE for {t1, t0} x {ms1, ms64}, and ms1 vs ms64 inside NEW.
Exit code 0 only if every comparison has the same seeds, tokens, lp and sha. Prints one line per comparison.
"""
import json
import os
import sys


def load(p):
    return json.load(open(p))["seqs"]


def same(a, b):
    ka, kb = set(a), set(b)
    if ka != kb:
        return False, f"seed sets differ ({len(ka)} vs {len(kb)})"
    bad = [k for k in ka if any(a[k][f] != b[k][f] for f in ("tok", "lp", "sha", "am"))]
    n = sum(len(a[k]["tok"]) for k in ka)
    return not bad, f"{len(ka)} seqs, {n} tokens, {len(bad)} seqs differ"


def main():
    base, new = sys.argv[1], sys.argv[2]
    d = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/speed_decode/out")
    ok = True
    for t in ("t1", "t0"):
        f = {(tag, ms): load(f"{d}/{tag}_{t}_ms{ms}.json") for tag in (base, new) for ms in (1, 64)}
        pairs = ((f"{t} ms1  new vs base", f[new, 1], f[base, 1]),
                 (f"{t} ms64 new vs base", f[new, 64], f[base, 64]),
                 (f"{t} new ms1 vs ms64", f[new, 1], f[new, 64]))
        for name, a, b in pairs:
            r, msg = same(a, b)
            ok &= r
            print(("PASS " if r else "FAIL ") + name + ": " + msg)
    print("GATE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
