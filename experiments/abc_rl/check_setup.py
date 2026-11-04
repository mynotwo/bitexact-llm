"""Setup checks on the GPU host before training.

  python check_setup.py --hf HF_DIR --const INT8_CONST --data GSM_DIR --rollout B0_BX_JSON

1. intq.quantize of the bf16 checkpoint (as fp32 master) reproduces the original int8 constants bit for bit.
2. intq.int_logp matches a float64 log-softmax of the same integer rows (|diff| ~1e-8) and is batch-invariant.
3. intq.generate (standalone engine, GPU PCG Gumbel) reproduces the vLLM-plugin rollout tokens of
   rl_mismatch (same prompts and seeds) and intq.score gives the same log p as the sampler.
4. A requantization after a small random update passes the engine's bound checks.
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "rl_mismatch"))
import common  # noqa: E402
import intq  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--const", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--n", type=int, default=16)
    a = ap.parse_args()
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    dev = torch.device("cuda")
    sd = load_file(os.path.join(a.hf, "model.safetensors"))
    P = {k: v.float().to(dev) for k, v in sd.items() if k != "lm_head.weight"}
    c, fw, st = intq.quantize(P)
    ref = torch.load(a.const, map_location="cpu", weights_only=False)
    bad = [k for k in c if k != "n_layers" and not (torch.equal(c[k].cpu(), ref[k]) if torch.is_tensor(c[k])
                                                    else c[k] == ref[k])]
    print("CHECK1 quantize == original constants:", "PASS" if not bad else f"FAIL {bad[:5]}")
    del ref
    eng = intq.IntEngine(a.const, c, dev)
    eng.load(c)

    tok = AutoTokenizer.from_pretrained(a.hf)
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok)
    rs = common.load(a.rollout)["seqs"]
    seeds = sorted(rs)[:a.n]
    ps = [prompts[rs[s]["p"]][0] for s in seeds]
    toks, lps = intq.generate(eng, ps, seeds, common.MAX_NEW, bs=a.n)
    same = sum(t == rs[s]["tok"] for t, s in zip(toks, seeds))
    dl = max(max(abs(x - y) for x, y in zip(lp, rs[s]["lp"])) for lp, s in zip(lps, seeds))
    print(f"CHECK3 standalone generate == plugin rollout: {same}/{len(seeds)} token-identical; "
          f"max |int_logp - common.int_logp| = {dl:.2e}")
    sc = intq.score(eng, ps, toks, bs=4)
    d = max(max(abs(x - y) for x, y in zip(u, v)) for u, v in zip(sc, lps))
    print("CHECK3b score == sampler log p:", "PASS" if d == 0 else f"FAIL max {d}")

    m = eng.m
    ids = torch.tensor([ps[0] + toks[0]], device=dev)
    h = m.forward(ids, torch.arange(ids.shape[1], device=dev)[None], m.new_cache(1, ids.shape[1]))[0]
    lg = m.logits(h)
    t = ids[0, 1:].clone()
    t = torch.cat([t, t[:1]])
    li = intq.int_logp(lg, t, m.ln)
    lf = (lg.double() / 2 ** 16).log_softmax(-1).gather(1, t[:, None])[:, 0]
    li3 = intq.int_logp(torch.cat([lg, lg, lg]), torch.cat([t, t, t]), m.ln)[:len(t)]
    print(f"CHECK2 int_logp vs float64: max {float((li - lf).abs().max()):.2e}; batch-invariant: "
          f"{bool(torch.equal(li, li3))}")

    g = torch.Generator(device=dev).manual_seed(0)
    P2 = {k: v + 1e-4 * torch.randn(v.shape, generator=g, device=dev) for k, v in P.items()}
    c2, _, st2 = intq.quantize(P2)
    frac = eng.load(c2)
    try:
        intq.generate(eng, ps[:4], seeds[:4], 64, bs=4)
        intq.score(eng, ps[:4], toks[:4], bs=4)
        ok = "PASS"
    except OverflowError as e:
        ok = f"FAIL {e}"
    print(f"CHECK4 requantized (1e-4 noise, {frac:.3%} int8 changed, fm {min(st2['fm'])}..{max(st2['fm'])} "
          f"vs {min(st['fm'])}..{max(st['fm'])}): bound checks {ok}")


if __name__ == "__main__":
    main()
