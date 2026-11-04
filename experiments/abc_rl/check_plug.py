"""C_plugin identity check: plugin rollouts vs intq.generate, plugin rescoring vs intq.score, sampler log p vs
rescoring log p, at step 0 and after a few weight updates (random perturbations of the fp32 master)."""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "rl_mismatch"))
import intq  # noqa: E402
import plug  # noqa: E402
import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--const", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--updates", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--noise", type=float, default=2e-3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(a.hf)
    train_p = train.train_prompts(a.data, tok)
    sd = load_file(os.path.join(a.hf, "model.safetensors"))
    P = {k: v.float().to(dev) for k, v in sd.items() if k != "lm_head.weight"}
    c, fw, qst = intq.quantize(P)
    pe = plug.PlugEngine(a.hf, a.const, a.work, 0.2)
    eng = intq.IntEngine(a.const, c, dev)
    g = torch.Generator(device=dev).manual_seed(0)
    res = []
    for upd in range(a.updates + 1):
        if upd > 0:
            for k in P:
                P[k] = P[k] * (1 + a.noise * torch.randn(P[k].shape, device=dev, generator=g))
            c, fw, qst = intq.quantize(P)
        ch_e, ch_p = eng.load(c), pe.load(c)
        idx = list(range(upd * a.n, (upd + 1) * a.n))
        prompts = [train_p[i][0] for i in idx]
        seeds = [1 + (7 * 1000 + upd) * 10000 + j for j in range(len(prompts))]
        t0 = time.time()
        it, ilp = intq.generate(eng, prompts, seeds, a.max_new, bs=256)
        t1 = time.time()
        pt, plp = pe.generate(prompts, seeds, a.max_new)
        t2 = time.time()
        tok_diff = sum(x != y for x, y in zip(it, pt))
        gen_lp_diff = sum(x != y for p, q in zip(ilp, plp) for x, y in zip(p, q)) + \
            sum(len(p) != len(q) for p, q in zip(ilp, plp))
        keep = [i for i, r in enumerate(it) if len(r) > 0]
        ps, rs = [prompts[i] for i in keep], [it[i] for i in keep]
        isc = intq.score(eng, ps, rs, 16)
        t3 = time.time()
        psc = pe.score(ps, rs)
        t4 = time.time()
        sc_diff = sum(x != y for p, q in zip(isc, psc) for x, y in zip(p, q))
        samp_vs_resc = sum(x != y for p, q in zip(plp, psc) for x, y in zip(p, q))
        samp_vs_resc_int = sum(x != y for i, q in zip(keep, isc) for x, y in zip(ilp[i], q))
        ntok = sum(map(len, rs))
        r = dict(update=upd, int8_changed_int=ch_e, int8_changed_plugin=ch_p, n_seq=len(prompts), n_tok=ntok,
                 seq_tokens_differ=tok_diff, sampler_lp_differ=gen_lp_diff, rescore_lp_differ=sc_diff,
                 plugin_sampler_vs_rescore_differ=samp_vs_resc, intq_sampler_vs_rescore_differ=samp_vs_resc_int,
                 sec=dict(intq_gen=t1 - t0, plug_gen=t2 - t1, intq_score=t3 - t2, plug_score=t4 - t3))
        print(json.dumps(r), flush=True)
        res.append(r)
    # bound check: a limit below the true bound must raise and write nothing
    from bitexact.vllm_plugin.rl import update_weights
    before = pe._rpc(lambda model: int(model.m.layers[0]["wqkv"][0].long().sum()))
    c2 = dict(c)
    c2["0.wqkv.m"] = c["0.wqkv.m"] * (1 << 20)         # same weights, multipliers 2^20 larger: |t| > 2^38
    try:
        pe._rpc(lambda model: update_weights(model.m, c2))
        raised = False
    except OverflowError as e:
        raised = str(e)
    after = pe._rpc(lambda model: int(model.m.layers[0]["wqkv"][0].long().sum()))
    print("BOUND", raised, before == after, flush=True)
    res.append({"bound_raised": raised, "unchanged": before == after})
    json.dump(res, open(a.out, "w"), indent=1)
    ok = all(r.get("seq_tokens_differ", 0) == r.get("sampler_lp_differ", 0) == r.get("rescore_lp_differ", 0) ==
             r.get("plugin_sampler_vs_rescore_differ", 0) == 0 for r in res) and bool(raised) and before == after
    print("CHECK", "PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
