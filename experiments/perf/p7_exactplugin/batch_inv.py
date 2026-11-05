"""8B plugin self-consistency: the same prompts alone (B=1), inside a batch of 32 and of 128 (greedy), plus the
first 16 steps' logits hashes (BITEXACT_HASH_LOG). All three must match token for token and hash for hash.

  CUDA_VISIBLE_DEVICES=0 python experiments/perf/p3_8b/batch_inv.py --model DIR --out OUT.json [--eager 0]
"""
import argparse
import json
import os
import random
import sys

NT, IN, OUT = 4, 128, 48


def hashes(path):
    d = {}
    for ln in open(path):
        s, t, h, y = ln.split()
        d[(int(s), int(t))] = (h, int(y))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eager", type=int, default=0)
    a = ap.parse_args()
    log = a.out + ".hash"
    os.environ["BITEXACT_HASH_LOG"] = log
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=a.model, enforce_eager=bool(a.eager), max_num_seqs=128, max_model_len=2048,
              max_num_batched_tokens=2048, gpu_memory_utilization=0.4, seed=0)
    rng = random.Random(1)
    ids = [[rng.randrange(1000, 120000) for _ in range(rng.choice([IN, 40, 300, 7]))] for _ in range(128)]
    sp = lambda i: SamplingParams(temperature=0.0, seed=i, max_tokens=OUT, ignore_eos=True, detokenize=False)
    res = {}
    for B in (1, 32, 128):
        open(log, "w").close()
        if B == 1:
            outs = [llm.generate([TokensPrompt(prompt_token_ids=ids[i])], sp(i), use_tqdm=False)[0] for i in range(NT)]
        else:
            outs = llm.generate([TokensPrompt(prompt_token_ids=ids[i]) for i in range(B)], [sp(i) for i in range(B)],
                                use_tqdm=False)[:NT]
        h = hashes(log)
        res[B] = ([list(o.outputs[0].token_ids) for o in outs], {k: v for k, v in h.items() if k[0] < NT})
    ok = all(res[B] == res[1] for B in (32, 128))
    json.dump({"tokens": res[1][0], "n_hash_rows": len(res[1][1]), "invariant_32_128": ok}, open(a.out, "w"))
    print("BATCH_INV", "PASS" if ok else "FAIL", "rows", len(res[1][1]), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
