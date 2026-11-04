"""Token streams of one vLLM run, for base-vs-branch comparison (production path: no BITEXACT_HASH_LOG).
python tokens.py --model DIR --out FILE.json [--logprobs]
128 requests, 64 in / 48 out: temperature cycles 0, 1, 0.7, 0.5, 1.5, 2 over requests (seed = index)."""
import argparse
import json
import random

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--logprobs", action="store_true")
ap.add_argument("--temps", default="0,1,0.7,0.5,1.5,2")
a = ap.parse_args()
llm = LLM(model=a.model, max_num_seqs=a.n, max_model_len=2048, max_num_batched_tokens=4096,
          gpu_memory_utilization=0.5, seed=0)
rng = random.Random(1)
temps = [float(x) for x in a.temps.split(",")]
prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 120000) for _ in range(64)]) for _ in range(a.n)]
sp = [SamplingParams(temperature=temps[i % len(temps)], seed=i, max_tokens=48, ignore_eos=True, detokenize=False,
                     logprobs=1 if a.logprobs else None) for i in range(a.n)]
outs = llm.generate(prompts, sp, use_tqdm=False)
json.dump([list(o.outputs[0].token_ids) for o in outs], open(a.out, "w"))
print("RESULT tokens", a.out, sum(len(o.outputs[0].token_ids) for o in outs))
