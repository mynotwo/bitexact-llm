"""Prefill throughput: B random prompts of L tokens, max_tokens=1, one step (B * L <= max_num_batched_tokens).

The generate runs once to warm up (compile, graph capture), then --reps timed times; prints one RESULT json line
with prompt tokens/s per rep (median and min/max). --attn picks vLLM's attention backend (stock comparison).

  python bench_prefill.py --model DIR --batch 64 --in-len 512 [--attn TRITON_ATTN] [--reps 7]
"""

import argparse
import json
import os
import random
import statistics
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--in-len", type=int, required=True)
    ap.add_argument("--attn", default="")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--tag", default="")
    ap.add_argument("--temp", type=float, default=0.0)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    kw = {"attention_backend": a.attn} if a.attn else {}
    llm = LLM(model=a.model, max_num_seqs=max(a.batch, 1), max_model_len=2048, enable_prefix_caching=False,
              max_num_batched_tokens=max(2048, a.batch * a.in_len), gpu_memory_utilization=0.5, seed=0, **kw)
    rng = random.Random(0)
    prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 120000) for _ in range(a.in_len)])
               for _ in range(a.batch)]
    sp = [SamplingParams(temperature=a.temp, seed=i, max_tokens=1, ignore_eos=True, detokenize=False)
          for i in range(a.batch)]
    llm.generate(prompts, sp, use_tqdm=False)
    llm.generate(prompts, sp, use_tqdm=False)
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        llm.generate(prompts, sp, use_tqdm=False)
        ts.append(time.perf_counter() - t0)
    n = a.batch * a.in_len
    tp = [n / t for t in ts]
    print("RESULT " + json.dumps({"tag": a.tag, "model": os.path.basename(a.model.rstrip("/")), "batch": a.batch,
          "in_len": a.in_len, "attn": a.attn, "bi": os.environ.get("VLLM_BATCH_INVARIANT", "0"),
          "ms_median": round(statistics.median(ts) * 1e3, 2), "tok_per_s_median": round(statistics.median(tp)),
          "tok_per_s_min": round(min(tp)), "tok_per_s_max": round(max(tp))}), flush=True)


if __name__ == "__main__":
    main()
