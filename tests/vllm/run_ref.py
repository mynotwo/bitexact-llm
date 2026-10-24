"""Reference for the invariance tests, without vLLM: bitexact.int8 IntLlama runs each target prompt on its own.

Per target: prefill of the prompt into its own Cache (1024-token chunks), then one token per step. Token choice
is greedy (first_argmax) or Gumbel-max with the same integer noise the plugin uses (sampling.raw_rows keyed by
(seed, t)). Writes OUT.json and OUT.hash in run_serve's format.

  python tests/vllm/run_ref.py --const int8_const.pt --out /path/ref [--long --model-dir DIR]
"""

import argparse
import hashlib
import json

import torch

PCHUNK = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--const", required=True)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--model-dir", help="needed with --long (RoPE table extension)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import workload

    from bitexact import sampling
    from bitexact.int8.consts import FR
    from bitexact.int8.model import IntLlama
    from bitexact.intops import first_argmax
    from bitexact.llama import Cache
    from bitexact.sampling import temp_ratio

    c = torch.load(a.const, map_location="cpu", weights_only=False)
    if a.long:
        from bitexact.vllm_plugin.model import _extend_rope
        _extend_rope(c, a.model_dir, 16384)
    m = IntLlama(c, "cuda", nl=2)
    res, lines = {}, []
    with torch.no_grad():
        for w in workload.pick(a.long, a.max_new):
            tn, td = temp_ratio(w["temp"])
            ids = list(w["ids"])
            P = len(ids)
            cache = Cache(m.L, 1, P + w["max_new"], m.dev)
            for c0 in range(0, P, PCHUNK):        # a12 prefill in chunks (exact in any split)
                c1 = min(P, c0 + PCHUNK)
                x = torch.tensor([ids[c0:c1]], device=m.dev)
                h = m.forward(x, torch.arange(c0, c1, device=m.dev)[None], cache)[:, -1]
            toks = []
            for t in range(w["max_new"]):
                li = m.logits(h)
                if tn == 0:
                    y = first_argmax(li)
                else:
                    g = sampling.gumbel(sampling.raw_rows([(w["seed"], t)]).to(m.dev), m.ln, FR)
                    y = sampling.sample(li, tn, td, g)
                yi = int(y[0])
                lines.append(f"{w['seed']} {t} {hashlib.sha256(li[0].cpu().numpy().tobytes()).hexdigest()[:16]} {yi}")
                toks.append(yi)
                if t + 1 < w["max_new"]:
                    h = m.forward(y.view(1, 1), torch.tensor([[P + t]], device=m.dev), cache)[:, -1]
            res[w["seed"]] = toks
    m.check()
    json.dump({str(k): v for k, v in sorted(res.items())}, open(a.out + ".json", "w"))
    open(a.out + ".hash", "w").write("\n".join(lines) + "\n")
    print("targets", len(res))


if __name__ == "__main__":
    main()
