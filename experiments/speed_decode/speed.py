"""Plugin decode speed inside vLLM (vLLM CUDA graphs, T = 1, ignore_eos), as prof_vllm_decode.py without profiler.

  python speed.py --model INT_DIR --data GSM_DIR --B 64 --gen 512 --out OUT.json

One LLM with max_num_seqs = B, B requests of 178-260-token GSM8K prompts, one untimed warm-up generate,
3 timed generates;
ms_per_step = wall / gen (prefill included), tok/s = generated tokens / wall; both medians of 3.
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--B", type=int, required=True)
    ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok, 64)
    from bitexact.vllm_plugin import register
    from bitexact.vllm_plugin.sampler import IntSampler
    register()
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.gen + 600, max_num_seqs=a.B,
              gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=True, seed=0, logits_processors=[IntSampler])

    def run(rep):
        ps = [TokensPrompt(prompt_token_ids=prompts[i % 64][0]) for i in range(a.B)]
        sp = [SamplingParams(temperature=1.0, top_p=1.0, seed=1000 * rep + i + 1, max_tokens=a.gen, ignore_eos=True)
              for i in range(a.B)]
        torch.cuda.synchronize()
        t0 = time.time()
        o = llm.generate(ps, sp, use_tqdm=False)
        torch.cuda.synchronize()
        return sum(len(x.outputs[0].token_ids) for x in o), time.time() - t0

    run(0)
    reps = [run(1), run(2), run(3)]
    row = {"B": a.B, "gen": a.gen, "tokens": reps[0][0], "sec": [round(d, 3) for _, d in reps],
           "tok_per_s_median": round(statistics.median(n / d for n, d in reps), 1),
           "ms_per_step_median": round(statistics.median(d for _, d in reps) / a.gen * 1e3, 3),
           "gpu": torch.cuda.get_device_name(0)}
    print("SPEED " + json.dumps(row), flush=True)
    json.dump(row, open(a.out, "w"), indent=1)
    import psutil
    for ch in psutil.Process().children(recursive=True):
        ch.kill()
    os._exit(0)


if __name__ == "__main__":
    main()
