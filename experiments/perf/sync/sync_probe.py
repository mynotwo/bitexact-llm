"""Host-device syncs inside vLLM's execute_model / sample_tokens, found with torch's sync debug mode.

Same workload as tests/vllm/bench.py (B random prompts, --in-len / --out-len, ignore_eos), one engine in this
process (VLLM_ENABLE_V1_MULTIPROCESSING=0). Pass 0 warms up (compile, graph capture); in pass 1 every
Worker.execute_model / Worker.sample_tokens call runs under torch.cuda.set_sync_debug_mode("warn"), and each
warning is recorded with the innermost stack frames in bitexact or vllm. torch.cuda.synchronize and
Event.synchronize are not reported by this mode (none are called in these two methods by the plugin).

  VLLM_ENABLE_V1_MULTIPROCESSING=0 python sync_probe.py --model DIR --batch 32 [--temp 1]
Prints "SYNC <count> <calls with a sync> <kind> <frames>" per distinct site and one RESULT JSON line.
"""

import argparse
import collections
import json
import random
import time
import traceback
import warnings

import torch

SITES = collections.Counter()
CALLS = {"exec": [0, 0, 0.0], "sample": [0, 0, 0.0]}      # calls, calls with a sync, host seconds
ON = [False]


def _frames():
    keep = [f for f in traceback.extract_stack()[:-3] if "/bitexact/" in f.filename or "/vllm/" in f.filename]
    return " < ".join(f"{f.filename.split('/site-packages/')[-1].split('/src/')[-1]}:{f.lineno}:{f.name}"
                      for f in reversed(keep[-4:]))


def _install():
    from vllm.v1.worker import gpu_worker
    W = gpu_worker.Worker

    def wrap(kind, fn):
        def run(self, *a, **k):
            if not ON[0]:
                return fn(self, *a, **k)
            seen = []
            show = warnings.showwarning
            torch.cuda.set_sync_debug_mode("warn")
            t = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.showwarning = lambda msg, *x, **y: seen.append(_frames())
                    return fn(self, *a, **k)
            finally:
                CALLS[kind][2] += time.perf_counter() - t
                torch.cuda.set_sync_debug_mode("default")
                warnings.showwarning = show
                CALLS[kind][0] += 1
                CALLS[kind][1] += bool(seen)
                for s in seen:
                    SITES[(kind, s)] += 1
        return run

    W.execute_model = wrap("exec", W.execute_model)
    W.sample_tokens = wrap("sample", W.sample_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--in-len", type=int, default=128)
    ap.add_argument("--out-len", type=int, default=32)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    _install()
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=a.model, max_num_seqs=a.batch, max_model_len=2048,
              max_num_batched_tokens=max(2048, a.batch * a.in_len), gpu_memory_utilization=0.5, seed=0)
    rng = random.Random(0)
    prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 120000) for _ in range(a.in_len)])
               for _ in range(a.batch)]
    sp = [SamplingParams(temperature=a.temp, seed=i, max_tokens=a.out_len, ignore_eos=True, detokenize=False)
          for i in range(a.batch)]
    llm.generate(prompts, sp, use_tqdm=False)
    ON[0] = True
    llm.generate(prompts, sp, use_tqdm=False)
    ON[0] = False
    for (kind, s), n in SITES.most_common():
        print(f"SYNC {n} {kind} {s}")
    print("RESULT " + json.dumps({"tag": a.tag, "batch": a.batch, "temp": a.temp, "sync_sites": len(SITES),
                                  "syncs": sum(SITES.values()),
                                  **{f"{k}_calls": v[0] for k, v in CALLS.items()},
                                  **{f"{k}_calls_with_sync": v[1] for k, v in CALLS.items()},
                                  **{f"{k}_host_ms_per_call": round(1e3 * v[2] / max(v[0], 1), 3)
                                     for k, v in CALLS.items()}}), flush=True)


if __name__ == "__main__":
    main()
