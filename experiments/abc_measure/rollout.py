"""Sampler side of the A/B/C measurement: vLLM bf16 or BitExact plugin rollouts, any length.

  python rollout.py --engine bf16|bitexact --model DIR --data GSM_DIR --out R.json \
      [--n-prompts 128] [--g 8] [--max-new 256] [--ignore-eos] [--dtype bfloat16]

Prompts, seeds (1 + 8p + g), T = 1, no top-k/top-p and the file format are those of rl_mismatch/rollout_vllm.py.
--ignore-eos keeps generating past EOS (the 2-shot GSM8K prompt makes the base model continue with new
Question/Answer pairs), so every sequence reaches --max-new tokens.
"""
import argparse
import os
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
    ap.add_argument("--n-prompts", type=int, default=common.N_PROMPTS)
    ap.add_argument("--g", type=int, default=common.G)
    ap.add_argument("--max-new", type=int, default=common.MAX_NEW)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-seqs", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    ap.add_argument("--no-prefix-cache", action="store_true")
    ap.add_argument("--attn", default=None)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="bitexact: skip the per-row hash/logp log (speed runs)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    log = a.out + ".rows"
    if a.engine == "bitexact":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        if not a.no_log:
            os.environ["BITEXACT_HASH_LOG"] = log
            if os.path.exists(log):
                os.remove(log)
            patch_plugin_log(log)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok, a.n_prompts)
    reqs = [(common.seed_of(p, g), p, ids) for p, (ids, _) in enumerate(prompts) for g in range(a.g)]
    llm = LLM(model=a.model, dtype=a.dtype, max_model_len=max(len(r[2]) for r in reqs) + a.max_new + 8,
              max_num_seqs=a.max_seqs, gpu_memory_utilization=a.gpu_mem, enable_prefix_caching=not a.no_prefix_cache, enforce_eager=a.eager, seed=0, **({"attention_backend": a.attn} if a.attn else {}))
    sp = lambda s: SamplingParams(temperature=common.TEMP, top_p=1.0, seed=s, max_tokens=a.max_new,
                                  ignore_eos=a.ignore_eos, logprobs=1 if a.engine == "bf16" else None)
    t0 = time.time()
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids) for _, _, ids in reqs], [sp(s) for s, _, _ in reqs],
                        use_tqdm=False)
    sec = time.time() - t0
    seqs = {}
    rows, bad = read_plugin_log(log) if a.engine == "bitexact" and not a.no_log else ({}, 0)
    for (s, p, _), o in zip(reqs, outs):
        c = o.outputs[0]
        toks = list(c.token_ids)
        rec = {"p": p, "tok": toks, "finish": c.finish_reason}
        if a.engine == "bf16":
            rec["lp"] = [c.logprobs[i][t].logprob for i, t in enumerate(toks)]
            rec["am"] = [next(k for k, v in c.logprobs[i].items() if v.rank == 1) for i in range(len(toks))]
        elif not a.no_log:
            got = [rows.get((s, i)) for i in range(len(toks))]
            assert all(g is not None and g[1] == t for g, t in zip(got, toks)), f"plugin log mismatch seed {s}"
            rec["sha"], rec["am"], rec["lp"] = [g[0] for g in got], [g[2] for g in got], [g[3] for g in got]
        if a.max_new <= 256:
            rec["reward"], rec["flag"] = common.grade(c.text, prompts[p][1])
        seqs[s] = rec
    n_tok = sum(len(v["tok"]) for v in seqs.values())
    meta = dict(vars(a), sec=round(sec, 2), inconsistent_repeats=bad, n_tokens=n_tok,
                tok_per_s=round(n_tok / sec, 1))
    try:
        import torch
        meta["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    common.save(a.out, a.engine, meta, seqs)
    print(f"RESULT {a.out} seqs {len(seqs)} tokens {n_tok} sec {sec:.1f} tok/s {n_tok / sec:.1f} bad {bad}")


if __name__ == "__main__":
    main()
