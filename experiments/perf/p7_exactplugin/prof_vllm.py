"""Where a vLLM step's time goes: the integer plugin vs stock bf16, same workload as tests/vllm/bench.py.

B random prompts of --in-len tokens, --out-len new tokens each (ignore_eos), max_num_seqs = B, one engine
in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0, so the worker's GPU work is visible to torch.profiler
here). Worker.execute_model / Worker.sample_tokens are wrapped: each call runs inside a profiler range
"bx::exec#k" / "bx::sample#k" (k = index of the non-empty step in the pass) and its host enter/exit times
are kept. Passes over the same prompts:
  0  warm-up (CUDA graph capture, torch.compile)
  1  timing only: tok/s (compare with bench.py) and host time per step, no profiler
  2  profiler on step 0 (the prefill of all B prompts, one step when B * in_len <= max_num_batched_tokens)
  3  profiler on decode steps --dec-start .. --dec-start + --dec-steps - 1
The profiler starts when exec#start is entered (no sync) and stops after sample#(last) returns (sync, then
stop); kernels are later attributed to steps through their launch call (buckets.py), so work of steps outside
the window is not counted.

  VLLM_ENABLE_V1_MULTIPROCESSING=0 python prof_vllm.py --model DIR --batch 32 --out OUT/int_b32 [--eager]
Writes OUT.json (timing + bucket summary) and OUT.{prefill,decode}.trace.json.gz (chrome traces).
"""

import argparse
import gzip
import json
import os
import random
import statistics
import time

import torch


class Hook:
    """Per-pass call log and profiler window, shared by the two wrapped Worker methods."""

    def __init__(self):
        self.reset()

    def reset(self, prof=None, start=0, n=0):
        self.prof, self.start, self.n = prof, start, n
        self.ne = self.ns = 0
        self.calls = []                          # (kind, k, t_enter, t_exit)
        self.t0 = self.t1 = None
        self.on = False


H = Hook()


def _install():
    from vllm.v1.worker import gpu_worker
    W = gpu_worker.Worker
    ex, sa = W.execute_model, W.sample_tokens

    def execute_model(self, scheduler_output):
        if scheduler_output.total_num_scheduled_tokens == 0:
            return ex(self, scheduler_output)
        k = H.ne
        H.ne += 1
        if H.prof is not None and k == H.start and not H.on:
            H.prof.start()
            H.on = True
            H.t0 = time.perf_counter()
        t = time.perf_counter()
        with torch.profiler.record_function(f"bx::exec#{k}"):
            out = ex(self, scheduler_output)
        H.calls.append(("exec", k, t, time.perf_counter()))
        return out

    def sample_tokens(self, grammar_output):
        k = H.ns
        H.ns += 1
        t = time.perf_counter()
        with torch.profiler.record_function(f"bx::sample#{k}"):
            out = sa(self, grammar_output)
        H.calls.append(("sample", k, t, time.perf_counter()))
        if H.on and k == H.start + H.n - 1:
            H.t1 = time.perf_counter()
            torch.cuda.synchronize()
            H.prof.stop()
            H.on = False
        return out

    W.execute_model, W.sample_tokens = execute_model, sample_tokens


def _med(xs):
    return round(statistics.median(xs), 3) if xs else None


def timing(calls, dec_lo):
    """Host-side step times (ms) from one unprofiled pass. period = enter(exec k+1) - enter(exec k)."""
    ent = {(c[0], c[1]): c for c in calls}
    ks = sorted(k for kind, k in ent if kind == "exec")
    per = {k: (ent["exec", k + 1][2] - ent["exec", k][2]) * 1e3 for k in ks if ("exec", k + 1) in ent}
    host = lambda kind, sel: [(ent[kind, k][3] - ent[kind, k][2]) * 1e3   # noqa: E731
                              for k in ks if sel(k) and (kind, k) in ent]
    dec = lambda k: k >= dec_lo   # noqa: E731
    return {"steps": len(ks),
            "prefill_step_ms": round(per.get(0, float("nan")), 3),
            "prefill_host_exec_ms": _med(host("exec", lambda k: k == 0)),
            "prefill_host_sample_ms": _med(host("sample", lambda k: k == 0)),
            "decode_step_ms_median": _med([v for k, v in per.items() if dec(k)]),
            "decode_step_ms_mean": round(statistics.fmean([v for k, v in per.items() if dec(k)]), 3)
            if any(dec(k) for k in per) else None,
            "decode_host_exec_ms": _med(host("exec", dec)),
            "decode_host_sample_ms": _med(host("sample", dec))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--in-len", type=int, default=128)
    ap.add_argument("--out-len", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--dec-start", type=int, default=8)
    ap.add_argument("--dec-steps", type=int, default=16)
    ap.add_argument("--out", required=True, help="output prefix (OUT.json, OUT.*.trace.json.gz)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "set VLLM_ENABLE_V1_MULTIPROCESSING=0"
    assert a.dec_start >= 1 and a.dec_start + a.dec_steps <= a.out_len - 1
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    import buckets
    _install()
    llm = LLM(model=a.model, enforce_eager=a.eager, max_num_seqs=max(a.batch, 1), max_model_len=2048, enable_prefix_caching=False,
              max_num_batched_tokens=max(2048, a.batch * a.in_len), gpu_memory_utilization=0.5, seed=0)
    rng = random.Random(0)
    prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 120000) for _ in range(a.in_len)])
               for _ in range(a.batch)]
    sp = [SamplingParams(temperature=a.temp, seed=i, max_tokens=a.out_len, ignore_eos=True, detokenize=False)
          for i in range(a.batch)]
    gen = lambda: llm.generate(prompts, sp, use_tqdm=False)   # noqa: E731

    H.reset()
    gen()                                                       # pass 0: warm-up
    H.reset()
    t0 = time.time()
    outs = gen()                                                # pass 1: timing
    dt = time.time() - t0
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    import bitexact
    res = {"tag": a.tag, "model": os.path.basename(a.model.rstrip("/")), "batch": a.batch, "eager": a.eager,
           "temp": a.temp, "in_len": a.in_len, "out_len": a.out_len, "out_tokens": n, "sec": round(dt, 3),
           "tok_per_s": round(n / dt, 1), "bitexact_src": os.path.dirname(bitexact.__file__),
           "timing": timing(H.calls, a.dec_start)}
    print("TIMING " + json.dumps(res), flush=True)

    acts = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    for phase, start, steps in (("prefill", 0, 1), ("decode", a.dec_start, a.dec_steps)):
        prof = torch.profiler.profile(activities=acts, record_shapes=False, with_stack=False)
        H.reset(prof, start, steps)
        gen()                                                   # passes 2, 3
        assert H.t1 is not None, f"{phase}: profiler window never closed (exec {H.ne}, sample {H.ns} calls)"
        path = f"{a.out}.{phase}.trace.json"
        prof.export_chrome_trace(path)
        with open(path, "rb") as f, gzip.open(path + ".gz", "wb") as g:
            g.write(f.read())
        os.remove(path)
        b = buckets.summarize(path + ".gz", start, steps)
        b["profiled_wall_ms_per_step"] = round((H.t1 - H.t0) * 1e3 / steps, 3)
        res[phase] = b
        print(f"{phase.upper()} " + json.dumps({k: v for k, v in b.items() if k != "top"}), flush=True)
    with open(a.out + ".json", "w") as f:
        json.dump(res, f, indent=1)
    print("RESULT " + json.dumps({"tag": a.tag, "batch": a.batch, "tok_per_s": res["tok_per_s"],
                                  "decode_step_ms": res["timing"]["decode_step_ms_median"],
                                  "decode_kernel_ms": res["decode"]["kernel_ms_per_step"]}), flush=True)


if __name__ == "__main__":
    main()
