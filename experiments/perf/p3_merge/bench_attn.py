"""tests/vllm/bench.py with an attention-backend choice: output tokens/s for B prompts served together.

Same prompts, sampling and timing as bench.py (second of two identical generate calls is timed, tok/s counts
output tokens only but the time includes prefill). Only addition: --attn picks vLLM's attention backend.

  VLLM_BATCH_INVARIANT=1 python bench_attn.py --model DIR --batch 32 --attn TRITON_ATTN
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
    ap.add_argument("--attn", default="")
    ap.add_argument("--in-len", type=int, default=128)
    ap.add_argument("--out-len", type=int, default=128)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    kw = {"attention_backend": a.attn} if a.attn else {}
    llm = LLM(model=a.model, max_num_seqs=max(a.batch, 1), max_model_len=2048,
              max_num_batched_tokens=max(2048, a.batch * a.in_len), gpu_memory_utilization=0.5, seed=0, **kw)
    rng = random.Random(0)
    prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 120000) for _ in range(a.in_len)])
               for _ in range(a.batch)]
    sp = [SamplingParams(temperature=0.0, seed=i, max_tokens=a.out_len, ignore_eos=True, detokenize=False)
          for i in range(a.batch)]
    llm.generate(prompts, sp, use_tqdm=False)
    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.time() - t0
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    print("RESULT " + json.dumps({"model": os.path.basename(a.model.rstrip("/")), "batch": a.batch,
                                  "attn": a.attn or "auto",
                                  "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT", "0"),
                                  "out_tokens": n, "sec": round(dt, 3), "tok_per_s": round(n / dt, 1)}), flush=True)


if __name__ == "__main__":
    main()
