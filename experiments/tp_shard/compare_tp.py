"""Compare TP-shard runs against the TP=1 reference: logits-row hash, greedy tokens, log p."""
import json, sys

def load(p):
    d = json.load(open(p))
    return d, p

def rows(ref, seqs, name, out):
    for k, a in ref.items():
        if not k.endswith("_tp1"):
            continue
        b = seqs.get(k.replace("_tp1", name))
        if b is None:
            continue
        n = len(a["tok"])
        tok = sum(x == y for x, y in zip(a["tok"], b["tok"]))
        sha = sum(x == y for x, y in zip(a["sha"], b["sha"]))
        lp = sum(x == y for x, y in zip(a["lp"], b["lp"]))
        out.append((k[:-4], name, tok, sha, lp, n))

def main():
    files = sys.argv[1:]
    base, bp = load(files[0])
    res = []
    for f in files[1:]:
        d, _ = load(f)
        rows(base["seqs"], d["seqs"], f"_tp{d['world']}", res)
    # the shipped IntLlama reference recorded inside the TP1 file
    ref = {k.replace("_ref", "_tp1"): v for k, v in base["seqs"].items() if k.endswith("_ref")}
    if ref:
        rows(ref, base["seqs"], "_tp1", res)
    print(f"const_sha {base.get('const_sha')}  max_new {base.get('max_new')}")
    print(f"{'prompt':8} {'arm':6} {'tok':10} {'sha16':10} {'lp':10}")
    for p, arm, tok, sha, lp, n in res:
        ok = tok == sha == lp == n
        print(f"{p:8} {arm:6} {tok}/{n:<7} {sha}/{n:<7} {lp}/{n:<7} {'identical' if ok else 'DIFFERS'}")
    print("ALL IDENTICAL" if all(r[2] == r[3] == r[4] == r[5] for r in res) else "MISMATCH")

main()
