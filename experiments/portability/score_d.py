"""Portability scorer for arm D (copy of abc_measure/score_vllm.py plus --tp): the same vLLM engine (VLLM_BATCH_INVARIANT=1, bf16) scores prompt + response via prompt_logprobs.

  VLLM_BATCH_INVARIANT=1 python score_vllm.py --rollout R.json --model DIR --data GSM_DIR --out S.json
         [--max-seqs 64] [--order shuffle|canon] [--max-len 20000]

Each sequence is submitted as a prompt of prompt+response tokens (max_tokens = 1, prompt_logprobs = 1: log p of every
prompt token under the raw logits, plus the rank-1 token). Batch size (--max-seqs) and request order (shuffle) differ
from the sampler's (256, canonical); prefix caching is off (vLLM skips it for prompt_logprobs). Response log p are
the entries at the response positions.
"""
import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rl_mismatch"))
import common  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-seqs", type=int, default=64)
    ap.add_argument("--order", default="shuffle")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    ap.add_argument("--attn", default=None)
    ap.add_argument("--prefix-cache", action="store_true")
    ap.add_argument("--batched-tokens", type=int, default=32768)
    ap.add_argument("--no-save", action="store_true", help="speed runs: skip building the output")
    ap.add_argument("--reps", type=int, default=1, help="timed passes; with reps > 1 the first is a warm-up and excluded")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok)
    rs = common.load(a.rollout)["seqs"]
    items = [(s, prompts[v["p"]][0], v["tok"]) for s, v in sorted(rs.items())]
    if a.order == "shuffle":
        random.Random(5).shuffle(items)
    llm = LLM(model=a.model, dtype=a.dtype, max_model_len=max(len(p) + len(r) for _, p, r in items) + 8,
              max_num_seqs=a.max_seqs, gpu_memory_utilization=a.gpu_mem, max_num_batched_tokens=a.batched_tokens,
              enable_prefix_caching=a.prefix_cache, seed=0, tensor_parallel_size=a.tp,
              **({"attention_backend": a.attn} if a.attn else {}))
    sp = SamplingParams(temperature=1.0, max_tokens=1, prompt_logprobs=1)
    secs = []
    for _ in range(a.reps):
        t0 = time.time()
        outs = llm.generate([TokensPrompt(prompt_token_ids=p + r) for _, p, r in items], sp, use_tqdm=False)
        secs.append(time.time() - t0)
    out = {}
    if not a.no_save:
        for (s, p, r), o in zip(items, outs):
            pl = o.prompt_logprobs
            lp, am = [], []
            for i in range(len(p), len(p) + len(r)):
                d = pl[i]
                lp.append(d[r[i - len(p)]].logprob)
                am.append(next(k for k, v in d.items() if v.rank == 1))
            out[s] = {"lp": lp, "am": am}
    n_seq = sum(len(p) + len(r) for _, p, r in items)
    import statistics
    import torch, vllm
    meta = dict(vars(a), vllm=vllm.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(0), secs=[round(x, 2) for x in secs], tok_per_s_median=round(n_seq / statistics.median(secs[1:] if len(secs) > 1 else secs), 1),
                seq_tokens=n_seq)
    common.save(a.out, "vllm_bi_score", meta, out)
    print(f"RESULT {a.out} seqs {len(items)} secs {secs} tok/s {meta['tok_per_s_median']}", flush=True)
    import psutil  # vLLM engine-core / TP worker children would outlive os._exit and hold the GPU
    for ch in psutil.Process().children(recursive=True):
        ch.kill()
    os._exit(0)


if __name__ == "__main__":
    main()
