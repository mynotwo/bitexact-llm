"""Throughput: output tokens/s for B prompts served together.

B random prompts of --in-len tokens, --out-len new tokens each (ignore_eos), max_num_seqs = B. The
same generate runs twice; the second is timed (the first pays torch.compile and CUDA graph capture).
Prints one JSON line.

  python tests/vllm/bench.py --model DIR --batch 32 [--eager] [--temp 0]
Stock batch-invariant mode: run with VLLM_BATCH_INVARIANT=1 in the environment.
"""

import argparse
import json
import os
import random
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--in-len", type=int, default=128)
    ap.add_argument("--out-len", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=a.model, enforce_eager=a.eager, max_num_seqs=max(a.batch, 1), max_model_len=2048,
              max_num_batched_tokens=max(2048, a.batch * a.in_len), gpu_memory_utilization=0.5, seed=0)
    rng = random.Random(0)
    prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 120000) for _ in range(a.in_len)])
               for _ in range(a.batch)]
    sp = [SamplingParams(temperature=a.temp, seed=i, max_tokens=a.out_len, ignore_eos=True, detokenize=False)
          for i in range(a.batch)]
    llm.generate(prompts, sp, use_tqdm=False)
    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.time() - t0
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    print("RESULT " + json.dumps({"tag": a.tag, "model": os.path.basename(a.model.rstrip("/")),
                                  "batch": a.batch, "eager": a.eager, "temp": a.temp,
                                  "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT", "0"),
                                  "in_len": a.in_len, "out_len": a.out_len, "out_tokens": n,
                                  "sec": round(dt, 3), "tok_per_s": round(n / dt, 1)}), flush=True)


if __name__ == "__main__":
    main()
