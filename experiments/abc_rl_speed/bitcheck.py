"""Bit check of arm C's integer path across the two code trees: old = exp/abc-rl (intq.generate + intq.score), new =
exp/abc-rl-speed (vLLM plugin sampler + packed Scorer).

The RL trainer itself is not bitwise reproducible from run to run (two runs of the same code differ in the gradient
at step 0, hence in all later weights), so two RL trajectories cannot be compared bit for bit. Both implementations get
the same weights instead: step k uses the fp32 master weights times (1 + 2e-3 * N(0,1)_k) (step 0: unchanged),
quantised by intq.quantize and loaded in place, as the RL loop does every step. Prompts and seeds are train.py's
(seed 1, 64 prompts x G = 8, 256 new tokens). Each mode writes tokens, sampler log p and scorer log p per step;
`compare` checks them.
  python bitcheck.py run --mode old|new --hf HF --const CONST --data DATA --work WORK --steps 10 --out OUT.pt
  python bitcheck.py compare OLD.pt NEW.pt
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "abc_rl"))
sys.path.insert(0, os.path.join(HERE, "..", "rl_mismatch"))
import common  # noqa: E402
import intq  # noqa: E402
import train  # noqa: E402


def compare(fo, fn):
    o, n = torch.load(fo), torch.load(fn)
    ok = True
    for a, b in zip(o, n):
        tokens = a["tok"] == b["tok"]
        flat = lambda L, ix: [x for i in ix for x in L[i]]
        r = {"step": a["step"], "rewards_equal": a["rew"] == b["rew"], "tokens_equal": tokens, "sampler_lp_equal": a["slp"] == b["slp"],
             "scorer_lp_equal": a["ilp"] == b["ilp"], "n_tok": len(flat(a["tok"], range(len(a["tok"])))),
             "delta_nonzero_old": a["delta_nonzero"], "delta_nonzero_new": b["delta_nonzero"]}
        ok &= all(v for k, v in r.items() if k.endswith("equal")) and not r["delta_nonzero_old"] and not r["delta_nonzero_new"]
        print(json.dumps(r))
    print("ALL EQUAL" if ok else "MISMATCH")


def main():
    if sys.argv[1] == "compare":
        return compare(sys.argv[2], sys.argv[3])
    sys.argv.pop(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["old", "new"], required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--const", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf)
    data = train.train_prompts(a.data, tok)
    sd = load_file(os.path.join(a.hf, "model.safetensors"))
    P0 = {k: v.float().to(dev) for k, v in sd.items() if k != "lm_head.weight"}
    del sd
    c0, _, _ = intq.quantize(P0)
    new = a.mode == "new"
    if new:
        pl = train.PluginSampler(a.hf, a.const, a.work, 0.2, 512)
    eng = intq.IntEngine(a.const, c0, dev)
    if new:
        from bitexact.int8 import scorer as scorer_mod
        sc = scorer_mod.Scorer(eng.m)
        scorer_mod.logp_rows = lambda lg, tgt: intq.int_logp(lg, tgt, eng.m.ln)
    gen = train.batches(len(data), 64, a.seed)
    res = []
    for step in range(a.steps):
        g = torch.Generator(device=dev).manual_seed(1000 + step)
        P = P0 if step == 0 else {k: v * (1 + 2e-3 * torch.randn(v.shape, generator=g, device=dev)) for k, v in P0.items()}
        c, _, _ = intq.quantize(P)
        changed = eng.load(c)
        if new:
            pl.load(c)
        pidx = next(gen)
        prompts = [data[i][0] for i in pidx for _ in range(8)]
        golds = [data[i][1] for i in pidx for _ in range(8)]
        seeds = [1 + (a.seed * 1000 + step) * 10000 + j for j in range(len(prompts))]
        toks, slp = pl.generate(prompts, seeds, 256) if new else intq.generate(eng, prompts, seeds, 256, bs=256)
        keep = [i for i, r in enumerate(toks) if len(r) > 0]
        kp, kr = [prompts[i] for i in keep], [toks[i] for i in keep]
        ilp = train.packed_score(eng, sc, kp, kr, 64) if new else intq.score(eng, kp, kr, 16)
        texts = tok.batch_decode([[x for x in r if x != intq.EOS] for r in toks])
        r = {"step": step, "tok": toks, "slp": slp, "ilp": ilp, "rew": [float(common.grade(t, gd)[0]) for t, gd in zip(texts, golds)],
             "delta_nonzero": sum(x != y for i, p in zip(keep, ilp) for x, y in zip(p, slp[i])), "int8_changed_frac": changed}
        print(step, "n_tok", sum(map(len, kr)), "delta_nonzero", r["delta_nonzero"], "changed", changed, flush=True)
        res.append(r)
    torch.save(res, a.out)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
