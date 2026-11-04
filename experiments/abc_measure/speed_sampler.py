"""Sampler throughput on one GPU: vLLM bf16 vs BitExact plugin, batch 1 / 8 / 64 x 256 / 4096 new tokens.

  python speed_sampler.py --engine bf16|bitexact --model DIR --data GSM_DIR --out S.json [--dtype bfloat16]

One LLM instance (max_num_seqs 64), same GSM8K 2-shot prompts, T = 1, ignore_eos, fixed seeds. One untimed warm-up
run per configuration, then 3 timed reps (new seeds each rep); tok/s = generated tokens / wall time of llm.generate
(prefill included; prefix cache on, so repeated prompts hit the cache after the warm-up, same for both engines).
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rl_mismatch"))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["bf16", "bitexact"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-mem", type=float, default=0.6)
    ap.add_argument("--lens", type=int, nargs="+", default=[256, 4096])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 64])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.engine == "bitexact":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok, 64)
    llm = LLM(model=a.model, dtype=a.dtype, max_model_len=max(a.lens) + 600, max_num_seqs=max(a.batches),
              gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=0)
    res = []
    for L in a.lens:
        for B in a.batches:
            def run(rep):
                ps = [TokensPrompt(prompt_token_ids=prompts[i][0]) for i in range(B)]
                sp = [SamplingParams(temperature=1.0, top_p=1.0, seed=1000 * rep + i + 1, max_tokens=L, ignore_eos=True)
                      for i in range(B)]
                torch.cuda.synchronize()
                t0 = time.time()
                o = llm.generate(ps, sp, use_tqdm=False)
                torch.cuda.synchronize()
                dt = time.time() - t0
                n = sum(len(x.outputs[0].token_ids) for x in o)
                return n, dt
            run(0)
            reps = [run(r + 1) for r in range(a.reps)]
            tps = [n / dt for n, dt in reps]
            row = {"engine": a.engine, "gen_len": L, "batch": B, "tok_per_s_median": round(statistics.median(tps), 1),
                   "tok_per_s_reps": [round(x, 1) for x in tps], "tokens_per_rep": reps[0][0]}
            print(json.dumps(row), flush=True)
            res.append(row)
            json.dump({"gpu": torch.cuda.get_device_name(0), "rows": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
