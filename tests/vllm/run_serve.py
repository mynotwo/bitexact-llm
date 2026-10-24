"""Serve the target prompts through vLLM with the integer model (vLLM test a).

  --mode alone   each target is the only request in the engine (one at a time)
  --mode busy    targets and fillers arrive at random engine steps (fixed RNG per --arrival-seed);
                 chunked prefill with a small token budget, so prefills are split and mixed with decodes
Writes OUT.json {seed: tokens} and OUT.hash (seed t sha16 token per sampled row, from the sampler).

  python tests/vllm/run_serve.py --model DIR --mode busy --prefix-cache 1 --out /path/run1
"""

import argparse
import json
import os
import random
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["alone", "busy"], required=True)
    ap.add_argument("--prefix-cache", type=int, default=1)
    ap.add_argument("--budget", type=int, default=256, help="max_num_batched_tokens")
    ap.add_argument("--max-seqs", type=int, default=64)
    ap.add_argument("--arrival-seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--kv-blocks", type=int, default=0, help="num_gpu_blocks_override (small: preemption)")
    ap.add_argument("--long", action="store_true", help="long-context targets (max_model_len 16384)")
    ap.add_argument("--eager", type=int, default=1, help="pass enforce_eager to vLLM")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for ext in (".hash", ".plan"):
        if os.path.exists(a.out + ext):
            os.remove(a.out + ext)
    os.environ["BITEXACT_HASH_LOG"] = a.out + ".hash"

    import workload
    from vllm import LLM
    from vllm.inputs import TokensPrompt

    llm = LLM(model=a.model, enforce_eager=bool(a.eager), enable_chunked_prefill=True,
              max_num_batched_tokens=a.budget, max_num_seqs=a.max_seqs,
              max_model_len=16384 if a.long else 2048, enable_prefix_caching=bool(a.prefix_cache),
              gpu_memory_utilization=0.35, seed=0, num_gpu_blocks_override=a.kv_blocks or None,
              disable_log_stats=False)
    eng = llm.llm_engine
    tg = workload.pick(a.long, a.max_new)
    res = {}
    t0 = time.time()
    if a.mode == "alone":
        for w in tg:
            out = llm.generate([TokensPrompt(prompt_token_ids=w["ids"])], workload.params(w), use_tqdm=False)
            res[w["seed"]] = list(out[0].outputs[0].token_ids)
    else:
        rng = random.Random(a.arrival_seed)
        pending = tg + workload.fillers(seed=100 + a.arrival_seed, n=20 if a.long else 40)
        rng.shuffle(pending)
        step = 0
        while pending or eng.has_unfinished_requests():
            k = rng.choice([0, 0, 1, 1, 2, 5]) if step else 6
            for w in pending[:k]:
                eng.add_request(str(w["seed"]), TokensPrompt(prompt_token_ids=w["ids"]), workload.params(w))
            pending = pending[k:]
            for o in eng.step():
                if o.finished and int(o.request_id) < 1000:
                    res[int(o.request_id)] = list(o.outputs[0].token_ids)
            step += 1
        print("steps", step)
    try:
        print("preemptions", sum(getattr(mt, "value", 0) for mt in llm.get_metrics() if "preempt" in mt.name))
    except Exception as e:
        print("metrics unavailable", e)
    print("sec", round(time.time() - t0, 1), "targets", len(res))
    try:
        for mt in llm.get_metrics():
            if "prefix_cache" in mt.name:
                print("metric", mt.name, getattr(mt, "value", None))
    except Exception as e:                       # metrics API differs across versions
        print("metrics unavailable", e)
    json.dump({str(k): v for k, v in sorted(res.items())}, open(a.out + ".json", "w"))


if __name__ == "__main__":
    main()
