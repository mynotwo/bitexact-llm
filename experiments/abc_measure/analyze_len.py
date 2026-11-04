"""Length-dependent mismatch metrics: sampler file vs scorer file over the sampler's tokens.

  python analyze_len.py --pair NAME=SAMPLER.json,SCORER.json ... --out metrics_len.json

d = log p_scorer - log p_sampler per token, r = exp(d), clip = [0.8, 1.2].
Per position bucket [0,256) [256,1k) [1k,4k) [4k,16k): token count, mean |d|, token r outside clip.
Per checkpoint T in 256 / 1024 / 4096 / 16384: over sequences with >= T tokens, cumulative sequence log-ratio
S_T = sum of d over the first T tokens: mean |S_T|, mean S_T, and fraction of sequences with exp(S_T) outside clip.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rl_mismatch"))
import common  # noqa: E402

BUCKETS = [(0, 256), (256, 1024), (1024, 4096), (4096, 16384)]
CHECK = [256, 1024, 4096, 16384]


def out_clip(x):
    return (x < 0.8) | (x > 1.2)


def metrics(sa, sc):
    seeds = sorted(set(sa) & set(sc))
    ds = []
    for s in seeds:
        assert len(sc[s]["lp"]) == len(sa[s]["tok"]), s
        ds.append(np.array(sc[s]["lp"], dtype=np.float64) - np.array(sa[s]["lp"], dtype=np.float64))
    res = {"seqs": len(seeds), "tokens": int(sum(map(len, ds))), "buckets": [], "checkpoints": []}
    for lo, hi in BUCKETS:
        v = np.concatenate([d[lo:hi] for d in ds if len(d) > lo] or [np.zeros(0)])
        if v.size:
            res["buckets"].append({"range": [lo, hi], "tokens": int(v.size), "mean_abs_d": float(np.abs(v).mean()),
                                   "max_abs_d": float(np.abs(v).max()),
                                   "frac_token_r_outside_clip": float(out_clip(np.exp(v)).mean())})
    for T in CHECK:
        S = np.array([d[:T].sum() for d in ds if len(d) >= T])
        if S.size:
            res["checkpoints"].append({"T": T, "seqs": int(S.size), "mean_abs_S": float(np.abs(S).mean()),
                                       "mean_S": float(S.mean()), "p90_abs_S": float(np.quantile(np.abs(S), .9)),
                                       "frac_seq_r_outside_clip": float(out_clip(np.exp(S)).mean())})
    res["frac_tokens_d_nonzero"] = float(np.mean(np.concatenate(ds) != 0))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    for spec in a.pair:
        name, files = spec.split("=")
        f1, f2 = files.split(",")
        if not (os.path.exists(f1) and os.path.exists(f2)):
            print("missing", name)
            continue
        out[name] = metrics(common.load(f1)["seqs"], common.load(f2)["seqs"])
    json.dump(out, open(a.out, "w"), indent=1)
    for n, r in out.items():
        print(n, "seqs", r["seqs"], "nonzero", round(r["frac_tokens_d_nonzero"], 4))
        for b in r["buckets"]:
            print("   bucket", b["range"], "mean|d| %.3g" % b["mean_abs_d"], "tokclip %.4f" % b["frac_token_r_outside_clip"])
        for c in r["checkpoints"]:
            print("   T", c["T"], "n", c["seqs"], "mean|S| %.3g" % c["mean_abs_S"], "seqclip %.3f" % c["frac_seq_r_outside_clip"])


if __name__ == "__main__":
    main()
