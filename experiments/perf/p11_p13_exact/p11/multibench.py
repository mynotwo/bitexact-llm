"""Measure three exact-mode batch sizes in one engine with shared fixed settings."""
import argparse
import json
import random
import time

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--tag', required=True)
a = p.parse_args()
llm = LLM(model=a.model, max_num_seqs=128, max_model_len=2048,
          max_num_batched_tokens=16384, gpu_memory_utilization=0.5, seed=0)
rng = random.Random(0)
prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000,120000) for _ in range(128)]) for _ in range(128)]
params = [SamplingParams(temperature=0.0, seed=i, max_tokens=128, ignore_eos=True, detokenize=False)
          for i in range(128)]
for batch in (32,128,1):
    for _ in range(2):
        llm.generate(prompts[:batch], params[:batch], use_tqdm=False)
    t0 = time.perf_counter()
    out = llm.generate(prompts[:batch], params[:batch], use_tqdm=False)
    dt = time.perf_counter() - t0
    tokens = sum(len(o.outputs[0].token_ids) for o in out)
    assert tokens == batch*128
    print('RESULT '+json.dumps({'tag':a.tag,'batch':batch,'temp':0,'in_len':128,'out_len':128,
          'out_tokens':tokens,'sec':dt,'tok_per_s':tokens/dt,
          'max_num_seqs':128,'max_num_batched_tokens':16384,'warmups':2}),flush=True)
