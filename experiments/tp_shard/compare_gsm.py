"""Compare TP scorings over the shared seeds: logits-row sha16 and int_logp, same token histories."""
import json, sys
sys.path.insert(0, ".")
import common

def main():
    ref = common.load(sys.argv[1])["seqs"]
    print(f"{'file':28} {'tp':4} {'seqs':5} {'tokens':8} {'sha16':14} {'lp':14} {'max|d|':8}")
    for f in sys.argv[2:]:
        d = common.load(f)
        x = d["seqs"]
        ks = sorted(set(ref) & set(x))
        n = sha_bad = lp_bad = 0
        mx = 0.0
        for k in ks:
            a, b = ref[k], x[k]
            for i in range(min(len(a["sha"]), len(b["sha"]))):
                n += 1
                sha_bad += a["sha"][i] != b["sha"][i]
                lp_bad += a["lp"][i] != b["lp"][i]
                mx = max(mx, abs(a["lp"][i] - b["lp"][i]))
        tp = d["meta"].get("tp")
        print(f"{f.split('/')[-1]:28} {tp:<4} {len(ks):<5} {n:<8} {n - sha_bad}/{n:<10} {n - lp_bad}/{n:<10} {mx:<8.3g}")
    print("done")

main()
