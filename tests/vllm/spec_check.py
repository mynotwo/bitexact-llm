"""Speculative decoding (ngram) vs plain decoding: token identity and throughput.

  python tests/vllm/spec_check.py --model DIR --work suite|rep|bench --batch B --temp T --k K --out PREFIX
--k 0 = no speculative decoding. Writes PREFIX.json {idx: tokens}; prints one RESULT line (second, timed generate).
Workloads: suite (the 12 targets of workload.py, each own temp/seed, max_new 96), rep (copy-heavy: a random chunk
of 64..128 tokens repeated 3x, then the chunk start; model continues the copy), bench (random 128-token prompts).
"""
import argparse
import json
import os
import random
import time


def prompts(a):
    import workload
    rng = random.Random(5)
    if a.work == "suite":
        return [(w["ids"], w["temp"], w["seed"]) for w in workload.targets(96)]
    out = []
    for i in range(a.batch):
        if a.work == "rep":
            ch = [rng.randrange(1000, 120000) for _ in range(rng.randrange(64, 128))]
            out.append((ch * 3 + ch[:8], a.temp, i))
        else:
            out.append(([rng.randrange(1000, 120000) for _ in range(128)], a.temp, i))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--work", default="suite")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--out-len", type=int, default=128)
    ap.add_argument("--eager", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    P = prompts(a)
    kw = {}
    if a.k:
        kw["speculative_config"] = {"method": "ngram", "num_speculative_tokens": a.k,
                                    "prompt_lookup_max": 5, "prompt_lookup_min": 2}
    llm = LLM(model=a.model, enforce_eager=bool(a.eager), max_num_seqs=max(len(P), 1), max_model_len=2048,
              max_num_batched_tokens=max(2048, len(P) * 520), gpu_memory_utilization=0.5, seed=0, disable_log_stats=False, **kw)
    n_out = 96 if a.work == "suite" else a.out_len
    sp = [SamplingParams(temperature=t, seed=s, max_tokens=n_out, ignore_eos=True, detokenize=False) for _, t, s in P]
    pr = [TokensPrompt(prompt_token_ids=p) for p, _, _ in P]
    llm.generate(pr, sp, use_tqdm=False)
    t0 = time.time()
    outs = llm.generate(pr, sp, use_tqdm=False)
    dt = time.time() - t0
    toks = [list(o.outputs[0].token_ids) for o in outs]
    json.dump({str(i): t for i, t in enumerate(toks)}, open(a.out + ".json", "w"))
    n = sum(map(len, toks))
    acc = None
    try:
        m = {x.name: getattr(x, "value", None) for x in llm.get_metrics()}
        acc = {k: v for k, v in m.items() if "spec" in k}
    except Exception as e:
        acc = str(e)
    print("RESULT " + json.dumps({"work": a.work, "batch": len(P), "temp": a.temp, "k": a.k, "tokens": n,
                                  "sec": round(dt, 3), "tok_per_s": round(n / dt, 1), "spec": acc}), flush=True)


if __name__ == "__main__":
    main()
