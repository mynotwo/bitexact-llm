"""Kernel timeline of a decode step with host stacks: where the GPU waits and which Python code launches next.

Same workload as tests/vllm/bench.py (B random prompts of --in-len tokens, --out-len new tokens, ignore_eos).
Two unprofiled passes (warm-up, timing), then one profiled pass with with_stack=True. For decode steps
--dec-start .. --dec-start + --dec-steps - 1 it prints, per step, every kernel in launch order:

  gap_us   time between the end of the previous kernel and the start of this one (start of step for the first)
  dur_us   kernel duration
  name
  stack    for gaps over --min-gap, the Python frames of the launch call (cuda_runtime / cuda_driver event
           with the same correlation id), innermost first

  python stacks.py --model DIR --batch 128 --temp 0 --out OUT
writes OUT.json (summary: per-step wall, kernel sum, idle, gap histogram, top gap sites).
"""

import argparse
import collections
import json
import os
import random
import re
import statistics
import time

import torch


def _frames(stack):
    """torch profiler stack -> ["file:line func", ...], innermost first."""
    out = []
    f = stack.get("stack") if isinstance(stack, dict) else None
    for x in (f or []):
        if isinstance(x, dict):
            out.append(f"{os.path.basename(x.get('file', ''))}:{x.get('line', '')} {x.get('name', '')}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--in-len", type=int, default=128)
    ap.add_argument("--out-len", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--dec-start", type=int, default=8)
    ap.add_argument("--dec-steps", type=int, default=8)
    ap.add_argument("--min-gap", type=float, default=4.0, help="us")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0", "set VLLM_ENABLE_V1_MULTIPROCESSING=0"
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    calls = []
    from vllm.v1.worker import gpu_worker
    W = gpu_worker.Worker
    ex, sa = W.execute_model, W.sample_tokens
    state = {"ne": 0, "ns": 0, "on": False, "prof": None, "start": a.dec_start, "n": a.dec_steps}

    def execute_model(self, scheduler_output):
        if scheduler_output.total_num_scheduled_tokens == 0:
            return ex(self, scheduler_output)
        k = state["ne"]
        state["ne"] += 1
        if state["prof"] is not None and k == state["start"] and not state["on"]:
            state["prof"].start()
            state["on"] = True
        t = time.perf_counter()
        with torch.profiler.record_function(f"bx::exec#{k}"):
            out = ex(self, scheduler_output)
        calls.append(("exec", k, t, time.perf_counter()))
        return out

    def sample_tokens(self, grammar_output):
        k = state["ns"]
        state["ns"] += 1
        t = time.perf_counter()
        with torch.profiler.record_function(f"bx::sample#{k}"):
            out = sa(self, grammar_output)
        calls.append(("sample", k, t, time.perf_counter()))
        if state["on"] and k == state["start"] + state["n"] - 1:
            torch.cuda.synchronize()
            state["prof"].stop()
            state["on"] = False
        return out

    W.execute_model, W.sample_tokens = execute_model, sample_tokens

    # Host-time markers for the pieces outside the captured model forward.
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    pi = GPUModelRunner.prepare_inputs

    def prepare_inputs(self, *a, **kw):
        with torch.profiler.record_function("bx::prepare_inputs"):
            return pi(self, *a, **kw)

    GPUModelRunner.prepare_inputs = prepare_inputs
    from bitexact.vllm_plugin.model import BitExactLlamaForCausalLM
    from bitexact.vllm_plugin.sampler import IntSampler
    cl, ap_ = BitExactLlamaForCausalLM.compute_logits, IntSampler.apply
    counts = collections.Counter()

    def compute_logits(self, hidden_states):
        counts["compute_logits"] += 1
        with torch.profiler.record_function(f"bx::compute_logits#{counts['compute_logits']}"):
            return cl(self, hidden_states)

    def apply(self, logits, ctx):
        counts["apply"] += 1
        with torch.profiler.record_function(f"bx::sampler_apply#{counts['apply']}"):
            return ap_(self, logits, ctx)

    BitExactLlamaForCausalLM.compute_logits, IntSampler.apply = compute_logits, apply

    llm = LLM(model=a.model, enforce_eager=a.eager, max_num_seqs=max(a.batch, 1), max_model_len=2048,
              enable_prefix_caching=False, max_num_batched_tokens=max(2048, a.batch * a.in_len),
              gpu_memory_utilization=0.5, seed=0)
    rng = random.Random(0)
    prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 120000) for _ in range(a.in_len)])
               for _ in range(a.batch)]
    sp = [SamplingParams(temperature=a.temp, seed=i, max_tokens=a.out_len, ignore_eos=True, detokenize=False)
          for i in range(a.batch)]
    gen = lambda: llm.generate(prompts, sp, use_tqdm=False)   # noqa: E731

    calls.clear()
    t0 = time.perf_counter()
    gen()
    dt = time.perf_counter() - t0
    res = {"tag": os.path.basename(a.out), "batch": a.batch, "temp": a.temp, "eager": a.eager,
           "pass_s": round(dt, 3)}
    ent = {(c[0], c[1]): c for c in calls}
    ek = sorted(k for kind, k in ent if kind == "exec")
    per = [ent["exec", k2][2] - ent["exec", k1][2] for k1, k2 in zip(ek, ek[1:]) if k1 >= 1]
    res["step_ms_median"] = round(statistics.median(per) * 1e3, 3) if per else None

    state["ne"] = state["ns"] = 0
    state["prof"] = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=False, with_stack=True)
    gen()
    assert not state["on"], "profiler window never closed"
    path = a.out + ".trace.json"
    state["prof"].export_chrome_trace(path)
    evs = json.load(open(path))["traceEvents"]
    os.remove(path)

    # step windows from the record_function ranges
    ann = collections.defaultdict(list)
    for e in evs:
        m = e.get("cat") == "user_annotation" and re.match(r"bx::(exec|sample)#(\d+)$", e.get("name", ""))
        if m:
            ann[m.group(2) + m.group(1)].append((e["ts"], e["ts"] + e["dur"], m.group(1)))
    launch = {e["args"]["correlation"]: e for e in evs
              if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    ann_by_tid = collections.defaultdict(list)
    for e in evs:
        m = e.get("cat") == "user_annotation" and re.match(r"bx::(exec|sample)#(\d+)$", e.get("name", ""))
        if m:
            ann_by_tid[e["tid"]].append((e["ts"], e["ts"] + e["dur"], m.group(1), int(m.group(2))))

    def owner(corr):
        r = launch.get(corr)
        if r is None:
            return None
        for ts, end, kind, k in ann_by_tid.get(r["tid"], ()):
            if ts <= r["ts"] <= end:
                return kind, k
        return None

    steps = range(a.dec_start, a.dec_start + a.dec_steps)
    ks = []
    for e in evs:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        corr = e.get("args", {}).get("correlation")
        o = owner(corr)
        if o is None or o[1] not in steps:
            continue
        r = launch.get(corr, {})
        ks.append({"step": o[1], "kind": o[0], "ts": e["ts"], "dur": e["dur"], "name": e.get("name", ""),
                   "lts": r.get("ts"), "ldur": r.get("dur"), "stack": _frames(r.get("args", {})),
                   "graph": bool(r.get("args", {}).get("graph")),
                   "cat": e.get("cat")})
    ks.sort(key=lambda x: x["ts"])

    # per-step timeline with gaps
    per_step = collections.defaultdict(list)
    for x in ks:
        per_step[x["step"]].append(x)
    gap_hist = collections.Counter()
    gap_sites = collections.defaultdict(lambda: [0.0, 0, []])
    lines = []
    idle_tot = busy_tot = span_tot = 0.0
    for k in sorted(per_step):
        seq = per_step[k]
        prev_end = None
        for x in seq:
            gap = x["ts"] - prev_end if prev_end is not None else 0.0
            if prev_end is not None:
                gap_hist[min(50, int(gap // 10) * 10)] += 1
                if gap >= a.min_gap:
                    key = (x["name"][:60], " <- ".join(x["stack"][:4]))
                    g = gap_sites[key]
                    g[0] += gap
                    g[1] += 1
                idle_tot += gap
            busy_tot += x["dur"]
            prev_end = x["ts"] + x["dur"]
            lines.append({"step": k, "gap_us": round(gap, 1), "dur_us": round(x["dur"], 1),
                          "start_us": round(x["ts"], 1), "name": x["name"], "cat": x["cat"],
                          "graph": x["graph"], "stack": x["stack"][:8]})
        if seq:
            span_tot += seq[-1]["ts"] + seq[-1]["dur"] - seq[0]["ts"]

    # host (CPU) durations of the marked ranges, per decode step of the profiled pass
    host = collections.defaultdict(float)
    hostn = collections.Counter()
    for e in evs:
        nme = e.get("name", "")
        if e.get("cat") == "user_annotation" and nme.startswith("bx::"):
            base = nme.split("#")[0]
            if base in ("bx::exec", "bx::sample", "bx::prepare_inputs", "bx::compute_logits",
                        "bx::sampler_apply") and e.get("ph") == "X":
                host[base] += e.get("dur", 0.0)
                hostn[base] += 1
    res["host_ms_per_step"] = {k: round(v / 1e3 / max(1, len(per_step)), 4) for k, v in host.items()}
    res["host_calls_per_step"] = {k: round(hostn[k] / max(1, len(per_step)), 2) for k in hostn}
    res.update({"kernels": len(ks), "kernels_per_step": round(len(ks) / max(1, len(per_step)), 1),
                "span_ms_per_step": round(span_tot / 1e3 / max(1, len(per_step)), 4),
                "busy_ms_per_step": round(busy_tot / 1e3 / max(1, len(per_step)), 4),
                "idle_ms_per_step": round(idle_tot / 1e3 / max(1, len(per_step)), 4),
                "gap_hist_us": {str(k2): v for k2, v in sorted(gap_hist.items())},
                "gap_sites": [{"gap_us_per_step": round(v[0] / max(1, len(per_step)), 1),
                               "n_per_step": round(v[1] / max(1, len(per_step)), 2),
                               "kernel": kk[0], "stack": kk[1]} for kk, v in
                              sorted(gap_sites.items(), key=lambda kv: -kv[1][0])[:25]],
                "timeline": [x for x in lines if x["step"] == a.dec_start + a.dec_steps // 2]})
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    print("STACK " + json.dumps({k2: v for k2, v in res.items() if k2 not in ("timeline", "gap_sites")}), flush=True)
    for g in res["gap_sites"]:
        print(f"GAP {g['gap_us_per_step']:8.1f} us/step x{g['n_per_step']:5.2f} -> {g['kernel']} <- "
              f"{g['stack'][:200]}", flush=True)
    print(f"# timeline of step {a.dec_start + a.dec_steps // 2} written to {a.out}.json", flush=True)
    for x in res["timeline"]:
        print(f"TL {x['step']} {x['gap_us']:8.1f} {x['dur_us']:8.1f} {'G' if x['graph'] else 'e'} {x['name'][:70]}",
              flush=True)


if __name__ == "__main__":
    main()
