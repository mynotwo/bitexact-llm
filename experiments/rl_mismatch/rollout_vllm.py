"""RL-style rollouts with vLLM 0.32: stock bf16 Llama-3.2-1B or the BitExact int8 plugin.

  python rollout_vllm.py --engine bf16 --model BF16_DIR --data GSM_DIR --out R.json [--order shuffle --max-seqs 32]
  python rollout_vllm.py --engine bitexact --model INTMODEL_DIR --data GSM_DIR --out B.json

128 GSM8K prompts x G = 8 seeds, T = 1, up to 256 new tokens, stop at EOS. Sampler-side log p:
- bf16: vLLM raw_logprobs of the sampled token (logprobs=1 also returns vLLM's own argmax token).
- bitexact: the plugin samples from int64 logits; its logging hook is replaced (in-process engine) by one
  that records, per sampled row, the sha16 of the row, the first argmax and common.int_logp(row, token).
"""

import argparse
import hashlib
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def patch_plugin_log(path):
    """Plugin sampler: write 'seed t sha16 token argmax logp' per sampled row to `path`."""
    from bitexact.intops import first_argmax
    from bitexact.vllm_plugin import sampler

    def _log(self, li, seeds, ts, y):
        am = first_argmax(li).tolist()
        rows = li.cpu().numpy()
        ys = y.tolist()
        with open(path, "a") as f:
            for r in range(len(seeds)):
                if ts[r] >= 0 and seeds[r] > 0:          # seed 0: vLLM warm-up / capture rows
                    h = hashlib.sha256(rows[r].tobytes()).hexdigest()[:16]
                    f.write(f"{seeds[r]} {ts[r]} {h} {ys[r]} {am[r]} {common.int_logp(rows[r], ys[r])!r}\n")
    sampler.IntSampler._log = _log


def read_plugin_log(path):
    """{(seed, t): (sha, tok, am, lp)}; repeated rows (recompute after preemption) must agree."""
    d, bad = {}, 0
    for ln in open(path):
        s, t, h, y, am, lp = ln.split()
        k, v = (int(s), int(t)), (h, int(y), int(am), float(lp))
        bad += k in d and d[k] != v
        d[k] = v
    return d, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["bf16", "bitexact"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="dir with GSM8K test.jsonl and train.jsonl")
    ap.add_argument("--order", choices=["canon", "shuffle"], default="canon")
    ap.add_argument("--max-seqs", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=0, help="submit requests in generate() calls of this size")
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    ap.add_argument("--n-prompts", type=int, default=common.N_PROMPTS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    log = a.out + ".rows"
    if a.engine == "bitexact":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["BITEXACT_HASH_LOG"] = log
        if os.path.exists(log):
            os.remove(log)
        patch_plugin_log(log)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = common.build_prompts(f"{a.data}/test.jsonl", f"{a.data}/train.jsonl", tok, a.n_prompts)
    reqs = common.requests(prompts)
    if a.order == "shuffle":
        random.Random(7).shuffle(reqs)
    llm = LLM(model=a.model, max_model_len=2048, max_num_seqs=a.max_seqs, gpu_memory_utilization=a.gpu_mem,
              enable_prefix_caching=True, seed=0)
    sp = lambda s: SamplingParams(temperature=common.TEMP, top_p=1.0, seed=s,
                                  max_tokens=common.MAX_NEW, logprobs=1 if a.engine == "bf16" else None)
    t0 = time.time()
    chunk = a.chunk or len(reqs)
    outs = []
    for i in range(0, len(reqs), chunk):
        part = reqs[i:i + chunk]
        outs += llm.generate([TokensPrompt(prompt_token_ids=ids) for _, _, ids in part],
                             [sp(s) for s, _, _ in part], use_tqdm=False)
    sec = time.time() - t0
    seqs = {}
    rows, bad = read_plugin_log(log) if a.engine == "bitexact" else ({}, 0)
    for (s, p, _), o in zip(reqs, outs):
        c = o.outputs[0]
        toks = list(c.token_ids)
        rec = {"p": p, "tok": toks, "text": c.text, "finish": c.finish_reason}
        if a.engine == "bf16":
            rec["lp"] = [c.logprobs[i][t].logprob for i, t in enumerate(toks)]
            rec["am"] = [next(k for k, v in c.logprobs[i].items() if v.rank == 1) for i in range(len(toks))]
        else:
            got = [rows.get((s, i)) for i in range(len(toks))]
            assert all(g is not None and g[1] == t for g, t in zip(got, toks)), f"plugin log mismatch seed {s}"
            rec["sha"], rec["am"], rec["lp"] = [g[0] for g in got], [g[2] for g in got], [g[3] for g in got]
        rec["reward"], rec["flag"] = common.grade(c.text, prompts[p][1])
        seqs[s] = rec
    meta = dict(vars(a), sec=round(sec, 1), inconsistent_repeats=bad, n_tokens=sum(len(v["tok"]) for v in seqs.values()))
    try:
        import torch
        meta["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    common.save(a.out, a.engine, meta, seqs)
    acc = sum(v["reward"] for v in seqs.values()) / len(seqs)
    print(f"RESULT {a.out} seqs {len(seqs)} tokens {meta['n_tokens']} acc {acc:.4f} sec {sec:.1f} bad {bad}")


if __name__ == "__main__":
    main()
