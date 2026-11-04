"""Portability sampler: vLLM in batch-invariant bf16 mode (arm D) or the BitExact int8 plugin (arm C).

  [VLLM_BATCH_INVARIANT=1] python sample.py --engine bf16|bitexact --model DIR --data GSM_DIR --out R.json
         [--temp 1|0] [--n-prompts 32] [--g 4] [--max-new 256] [--max-seqs 256] [--tp 1] [--order canon|shuffle]
         [--attn TRITON_ATTN]

Workload: first --n-prompts GSM8K test questions, 2-shot prefix, G samples per prompt, request seed 1 + 8p + g,
T = --temp (0 = greedy), no top-k/top-p, stop at EOS. Output file format as rl_mismatch/common.py; bf16 records
vLLM's raw log p (logprobs=1), the plugin records sha16 / argmax / float64 log p of the int64 logits row.
"""
import argparse
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rl_mismatch"))
import common  # noqa: E402
from rollout_vllm import patch_plugin_log, read_plugin_log  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["bf16", "bitexact"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--g", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-seqs", type=int, default=256)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--order", choices=["canon", "shuffle"], default="canon")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    ap.add_argument("--attn", default=None)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    log = a.out + ".rows"
    if a.engine == "bitexact":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["BITEXACT_HASH_LOG"] = log
        if os.path.exists(log):
            os.remove(log)
        patch_plugin_log(log)
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok, a.n_prompts)
    reqs = [(common.seed_of(p, g), p, ids) for p, (ids, _) in enumerate(prompts) for g in range(a.g)]
    if a.order == "shuffle":
        random.Random(7).shuffle(reqs)
    kw = {"attention_backend": a.attn} if a.attn else {}
    if a.engine == "bitexact":
        # Without an installed bitexact package (no entry points), register the model class and the integer
        # sampler in-process (the plugin runs in-process anyway: V1 multiprocessing off).
        from importlib.metadata import entry_points
        if not any(e.name == "bitexact_sampler" for e in entry_points(group="vllm.logits_processors")):
            from bitexact.vllm_plugin import register
            from bitexact.vllm_plugin.sampler import IntSampler
            register()
            kw["logits_processors"] = [IntSampler]
            a.registered_in_process = True
    llm = LLM(model=a.model, dtype=a.dtype, max_model_len=max(len(r[2]) for r in reqs) + a.max_new + 8,
              max_num_seqs=a.max_seqs, gpu_memory_utilization=a.gpu_mem, tensor_parallel_size=a.tp,
              enforce_eager=a.eager, seed=0, **kw)
    sp = lambda s: SamplingParams(temperature=a.temp, top_p=1.0, seed=s, max_tokens=a.max_new,
                                  logprobs=1 if a.engine == "bf16" else None)
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids) for _, _, ids in reqs], [sp(s) for s, _, _ in reqs],
                        use_tqdm=False)
    sec = time.time() - t0
    rows, bad = read_plugin_log(log) if a.engine == "bitexact" else ({}, 0)
    seqs = {}
    for (s, p, _), o in zip(reqs, outs):
        c = o.outputs[0]
        toks = list(c.token_ids)
        rec = {"p": p, "tok": toks}
        if a.engine == "bf16":
            rec["lp"] = [c.logprobs[i][t].logprob for i, t in enumerate(toks)]
            rec["am"] = [next(k for k, v in c.logprobs[i].items() if v.rank == 1) for i in range(len(toks))]
        else:
            got = [rows.get((s, i)) for i in range(len(toks))]
            assert all(g is not None and g[1] == t for g, t in zip(got, toks)), f"plugin log mismatch seed {s}"
            rec["sha"], rec["am"], rec["lp"] = [g[0] for g in got], [g[2] for g in got], [g[3] for g in got]
        seqs[s] = rec
    meta = dict(vars(a), sec=round(sec, 2), inconsistent_repeats=bad, vllm=vllm.__version__, torch=torch.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                batch_invariant=os.environ.get("VLLM_BATCH_INVARIANT"),
                n_tokens=sum(len(v["tok"]) for v in seqs.values()))
    common.save(a.out, a.engine, meta, seqs)
    print(f"RESULT {a.out} seqs {len(seqs)} tokens {meta['n_tokens']} sec {sec:.1f} bad {bad}", flush=True)
    os._exit(0)  # vLLM 0.32 core-dumps at interpreter shutdown on some hosts; output is already written


if __name__ == "__main__":
    main()
