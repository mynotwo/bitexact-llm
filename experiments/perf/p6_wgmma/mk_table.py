"""Fold the sweep4 logs into the dispatch table of src/bitexact/int8/gemm_sm90.py.

  python mk_table.py sw4_*.log

Each log line is one (shape, M) with its top-3 deadline cfgs and the graph-timed ms of cutlass / the old path /
those cfgs. A cfg is only usable in a row bucket if it was measured at every M of the bucket; the pick is the
one with the best worst-case ratio to the old path. A bucket enters TABLE only when that ratio is <= 0.95 (the
kernel has to be >= 5% faster than what the repo does today to be worth the branch), and the buckets are the
row ranges delimited by the largest gap in the measured M values.
"""
import json
import sys


def load(paths):
    rows = {}
    for p in paths:
        for line in open(p):
            line = line.strip()
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            key = (r["shape"], r["M"])
            cand = {tuple(b["cfg"]): b["ms"] for b in r["best"]}
            if key in rows:                               # same (shape, M) measured twice with another grid
                old = rows[key]
                rows[key] = {"cut": min(old["cut"], r["cutlass_ms"]), "old": min(old["old"], r["old_ms"]),
                             "cand": {c: min(v, cand[c]) for c, v in cand.items()} | {
                                 c: min(cand[c], old["cand"][c]) for c in cand if c in old["cand"]}}
            else:
                rows[key] = {"cut": r["cutlass_ms"], "old": r["old_ms"], "cand": cand}
    return rows


def table(rows, need=0.95, gap=1.6):
    shapes = sorted({s for s, _ in rows})
    out = {}
    for sh in shapes:
        ms = sorted(m for s, m in rows if s == sh)
        buckets, cur = [], [ms[0]]
        for m in ms[1:]:
            if m > cur[-1] * gap:
                buckets.append(cur)
                cur = [m]
            else:
                cur.append(m)
        buckets.append(cur)
        picks = []
        for b in buckets:
            cfgs = set(rows[(sh, b[0])]["cand"])
            for m in b[1:]:
                cfgs &= set(rows[(sh, m)]["cand"])
            if not cfgs:
                continue
            best = min(cfgs, key=lambda c: max(rows[(sh, m)]["cand"][c] / rows[(sh, m)]["old"] for m in b))
            worst = max(rows[(sh, m)]["cand"][best] / rows[(sh, m)]["old"] for m in b)
            if worst <= need:
                picks.append((b[-1] if b[-1] != ms[-1] else 1 << 30, list(best), round(worst, 3), b))
        if picks:
            out[sh] = picks
    return out


if __name__ == "__main__":
    for sh, picks in table(load(sys.argv[1:])).items():
        print(f"    ({sh.replace('x', ', ')}): [", end="")
        print(", ".join(f"({mx}, {tuple(cfg)})" for mx, cfg, worst, _ in picks) + "],   #" +
              " /".join(f" {w}" for _, _, w, _ in picks))