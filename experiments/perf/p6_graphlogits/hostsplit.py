"""Unprofiled host-time split of one decode step (time.perf_counter, no torch.profiler).

Wraps, for one bench-like generate: Worker.execute_model / Worker.sample_tokens (GPUModelRunner level
prepare_inputs and sample), the plugin's compute_logits and each of its pieces (pre_norm, LM-head GEMM,
tri_sample pass, _check_late), and IntSampler.apply. Prints median us per step per piece.
Nothing is committed; this is the temporary instrumentation for step 1 of the graphlogits lane.

  python hostsplit.py --model DIR --batch 128 --temp 0 --out OUT
"""

import argparse
import collections
import json
import os
import random
import statistics
import time

T = collections.defaultdict(list)


def wrap(obj, name, label, attr=None):
    fn = getattr(obj, attr or name)

    def g(*a, **kw):
        t = time.perf_counter_ns()
        try:
            return fn(*a, **kw)
        finally:
            T[label].append((time.perf_counter_ns() - t) / 1e3)
    setattr(obj, attr or name, g)
    return g


def wrap_fn(fn, label):
    def g(*a, **kw):
        t = time.perf_counter_ns()
        try:
            return fn(*a, **kw)
        finally:
            T[label].append((time.perf_counter_ns() - t) / 1e3)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--in-len", type=int, default=128)
    ap.add_argument("--out-len", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0"

    from bitexact.vllm_plugin import tri_sample
    from bitexact.vllm_plugin.model import BitExactLlamaForCausalLM
    from bitexact.vllm_plugin.sampler import IntSampler
    from vllm.v1.worker import gpu_worker
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    wrap(gpu_worker.Worker, "execute_model", "worker.execute_model")
    wrap(gpu_worker.Worker, "sample_tokens", "worker.sample_tokens")
    wrap(GPUModelRunner, "prepare_inputs", "runner.prepare_inputs")
    wrap(GPUModelRunner, "sample", "runner.sample")
    wrap(BitExactLlamaForCausalLM, "compute_logits", "compute_logits")
    wrap(IntSampler, "apply", "IntSampler.apply")
    wrap(tri_sample, "head", "  tri_sample.head")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=a.model, max_num_seqs=max(a.batch, 1), max_model_len=2048, enable_prefix_caching=False,
              max_num_batched_tokens=max(2048, a.batch * a.in_len), gpu_memory_utilization=0.5, seed=0)
    rng = random.Random(0)
    prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 120000) for _ in range(a.in_len)])
               for _ in range(a.batch)]
    sp = [SamplingParams(temperature=a.temp, seed=i, max_tokens=a.out_len, ignore_eos=True, detokenize=False)
          for i in range(a.batch)]
    # pieces of compute_logits: instance attributes, so wrap them on the loaded model
    from bitexact.vllm_plugin.model import MODEL
    mm, sg = MODEL.m, MODEL.m.sg
    pn, gemm = sg.pre_norm, mm._gemm
    sg.pre_norm = wrap_fn(pn, "  sg.pre_norm")
    mm._gemm = wrap_fn(gemm, "  IntLlama._gemm")
    llm.generate(prompts, sp, use_tqdm=False)
    T.clear()
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    res = {"batch": a.batch, "temp": a.temp, "sec": round(dt, 3), "tok_per_s": round(n / dt, 1),
           "steps": len(T["worker.execute_model"]),
           "median_us": {k: round(statistics.median(v), 1) for k, v in sorted(T.items()) if v},
           "mean_us": {k: round(statistics.fmean(v), 1) for k, v in sorted(T.items()) if v},
           "calls": {k: len(v) for k, v in sorted(T.items()) if v}}
    print("HOSTSPLIT " + json.dumps(res), flush=True)
    json.dump(res, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()