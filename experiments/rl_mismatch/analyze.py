"""Metrics for the RL mismatch experiment (definitions fixed in the PLAN of the report).

  python analyze.py --tokenizer DIR --data GSM_DIR --out metrics.json \
      --mismatch NAME=SAMPLER.json,SCORER.json[,ROLLOUT.json] ... --replay NAME=REF.json,RUN.json ...

mismatch: per token d = logp_scorer - logp_sampler over the sampler's tokens (seeds present in both files).
replay: token-for-token comparison of two rollouts of the same (prompt, seed) requests; flagged = reward 0
in REF (rule-based grader, regraded here from the tokens with one tokenizer for every engine).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

CLIP = (0.8, 1.2)


def _q(x, q):
    return float(np.quantile(x, q)) if len(x) else None


def mismatch(sa, sc):
    seeds = sorted(set(sa) & set(sc))
    d, absd, amd, shad, nsha = [], [], 0, 0, 0
    seq_lr, k1, k3 = [], [], []
    for s in seeds:
        a, b = sa[s], sc[s]
        n = len(a["tok"])
        assert len(b["lp"]) == n, f"seed {s}: scorer has {len(b['lp'])} positions, sampler {n}"
        x = np.array(b["lp"], dtype=np.float64) - np.array(a["lp"], dtype=np.float64)
        d.append(x)
        amd += sum(p != q for p, q in zip(a["am"], b["am"]))
        if "sha" in a and "sha" in b:
            nsha += n
            shad += sum(p != q for p, q in zip(a["sha"], b["sha"]))
        seq_lr.append(x.sum())
        k1.append(float(np.mean(-x)))
        r = np.exp(x)
        k3.append(float(np.mean(r - 1 - x)))
    d = np.concatenate(d)
    absd, r = np.abs(d), np.exp(d)
    seq_lr = np.array(seq_lr)
    sr = np.exp(seq_lr)
    out = {
        "prompts": len({sa[s]["p"] for s in seeds}), "seqs": len(seeds), "tokens": int(d.size),
        "absd_mean": float(absd.mean()), "absd_p50": _q(absd, .5), "absd_p99": _q(absd, .99),
        "absd_max": float(absd.max()), "frac_tokens_d_nonzero": float((d != 0).mean()),
        "ratio_mean": float(r.mean()), "ratio_p1": _q(r, .01), "ratio_p99": _q(r, .99),
        "ratio_min": float(r.min()), "ratio_max": float(r.max()),
        "frac_tokens_outside_clip": float(((r < CLIP[0]) | (r > CLIP[1])).mean()),
        "frac_argmax_differs": amd / d.size,
        "seq_abs_logratio_mean": float(np.abs(seq_lr).mean()), "seq_abs_logratio_p99": _q(np.abs(seq_lr), .99),
        "seq_abs_logratio_max": float(np.abs(seq_lr).max()),
        "frac_seqs_ratio_outside_clip": float(((sr < CLIP[0]) | (sr > CLIP[1])).mean()),
        "frac_seqs_any_token_differs": float(np.mean([np.any(x != 0) for x in np.split(d, np.cumsum([len(sa[s]["tok"]) for s in seeds])[:-1])])),
        "kl_k1_mean": float(np.mean(k1)), "kl_k3_mean": float(np.mean(k3)), "kl_k3_max": float(np.max(k3)),
    }
    if nsha:
        out["frac_logits_sha_differs"] = shad / nsha
        out["positions_sha_compared"] = nsha
    return out


def replay(ref, run, reward):
    seeds = sorted(set(ref) & set(run))
    fd = {s: common.first_div(ref[s]["tok"], run[s]["tok"]) for s in seeds}
    flagged = [s for s in seeds if reward[s] == 0]
    div = [fd[s] for s in seeds if fd[s] is not None]
    fdiv = [fd[s] for s in flagged if fd[s] is not None]
    verdict = sum(reward[s] != run[s]["_reward"] for s in seeds)
    return {
        "seqs": len(seeds), "flagged": len(flagged),
        "flagged_by_kind": {k: sum(ref[s]["_flag"] == k for s in flagged) for k in ("wrong", "no_answer")},
        "replay_identical_all": sum(fd[s] is None for s in seeds),
        "replay_identical_flagged": sum(fd[s] is None for s in flagged),
        "first_div_p10": _q(div, .1), "first_div_p50": _q(div, .5), "first_div_p90": _q(div, .9),
        "first_div_min": min(div) if div else None, "first_div_max": max(div) if div else None,
        "flagged_div_within_16": sum(x < 16 for x in fdiv),
        "verdict_changed": verdict,
        "verdict_changed_flagged": sum(run[s]["_reward"] != 0 for s in flagged),
        "first_div_hist": np.histogram(div, bins=[0, 1, 4, 16, 64, 128, 257])[0].tolist() if div else [],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--mismatch", nargs="*", default=[])
    ap.add_argument("--replay", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok)
    cache = {}

    def get(path):
        if path not in cache:
            d = common.load(path)["seqs"]
            for v in d.values():
                if "tok" in v:
                    v["_reward"], v["_flag"] = common.grade(tok.decode(v["tok"], skip_special_tokens=True),
                                                            prompts[v["p"]][1])
            cache[path] = d
        return cache[path]

    res = {"mismatch": {}, "replay": {}}
    for spec in a.mismatch:
        name, files = spec.split("=")
        f1, f2, *base = files.split(",")        # optional 3rd file: rollout giving tok / p for scorer-vs-scorer
        sa, sc = get(f1), get(f2)
        if base:
            b = get(base[0])
            sa = {s: dict(v, tok=b[s]["tok"], p=b[s]["p"]) for s, v in sa.items()}
        sc = {s: dict(v, tok=sa[s]["tok"]) for s, v in sc.items() if s in sa}
        res["mismatch"][name] = mismatch(sa, sc)
        print(name, json.dumps(res["mismatch"][name]))
    for spec in a.replay:
        name, files = spec.split("=")
        f1, f2 = files.split(",")
        ref, run = get(f1), get(f2)
        res["replay"][name] = replay(ref, run, {s: v["_reward"] for s, v in ref.items()})
        print(name, json.dumps(res["replay"][name]))
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
