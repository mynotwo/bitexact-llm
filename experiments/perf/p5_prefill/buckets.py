"""Bucket the GPU kernels of a torch.profiler chrome trace into per-step ms (used by prof_vllm.py).

A kernel belongs to step k when its launch call (the cuda_runtime / cuda_driver event with the same
correlation id; for a CUDA graph replay, cudaGraphLaunch) lies inside the profiler range "bx::exec#k" or
"bx::sample#k" of the same thread. Only steps start .. start + n - 1 count; per-step values are sums / n.

Buckets (first match wins):
  sampling_logits  launched from sample_tokens: LM-head GEMM, logits processors, sampler (any kernel name)
  kv_write         cache writes (bf16: reshape_and_cache*; plugin: index_put / cat of the K|V rows)
  attention        attention kernels (plugin: Triton _max_kernel / _acc_kernel; bf16: flash / fmha)
  gemm_int8        integer GEMMs (torch._int_mm limbs: cuBLASLt / CUTLASS s8 / i8 / imma kernels)
  gemm_bf16        other GEMMs
  elementwise      Inductor Triton (triton_*) and ATen elementwise / reduction / copy / index kernels:
                   plugin = quant / requant / integer RMSNorm, RoPE, softmax, SiLU; bf16 = norm / RoPE / act
  memcpy           memcpy / memset activities
  other            everything else
gpu_idle = window span (first kernel start .. last kernel end of the counted steps) - union of their kernel
intervals, measured under the profiler (its host overhead stretches gaps). The report also gives
wall - busy, with wall from the unprofiled timing pass: the per-step time not covered by GPU kernels.

  python buckets.py TRACE.json.gz START N          one trace
  python buckets.py --report DIR                   markdown tables from DIR/*.json (prof_vllm.py outputs)
"""

import collections
import glob
import gzip
import json
import os
import re
import sys

BUCKETS = ["gemm_int8", "gemm_bf16", "elementwise", "attention", "kv_write", "sampling_logits", "memcpy",
           "other"]
_ATT = re.compile(r"_max_kernel|_acc_kernel|flash|fmha|attention|attn_fwd|paged_attention|unified_attention",
                  re.I)
_KV = re.compile(r"reshape_and_cache|index_put|CatArrayBatchedCopy|concat_and_cache", re.I)
_GEMM = re.compile(r"gemm|gemv|xmma|cutlass|nvjet|cublas|matmul|splitk|_mm_|wgmma", re.I)
_INT = re.compile(r"s8|i8|int8|imma|igemm|_int_mm", re.I)
_EW = re.compile(r"^triton_|elementwise|reduce|copy|fill|where|index|gather|scatter|arange|cat|unrolled|"
                 r"vectorized|softmax|argmax|rms|rotary|silu|act_and_mul|norm|compare|sort|topk|cumsum", re.I)


def bucket(name, cat, in_sample):
    if in_sample:
        return "sampling_logits"
    if cat != "kernel":
        return "memcpy"
    if _KV.search(name):                     # before attention: reshape_and_cache_flash contains "flash"
        return "kv_write"
    if _ATT.search(name):
        return "attention"
    if _GEMM.search(name):
        return "gemm_int8" if _INT.search(name) else "gemm_bf16"
    if name.startswith("triton_") or _EW.search(name):
        return "elementwise"
    return "other"


def _union(iv):
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return tot + (cur_e - cur_s if cur_e is not None else 0.0)


def summarize(path, start, n, top=15):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        evs = json.load(f)["traceEvents"]
    ann = collections.defaultdict(list)                 # tid -> [(ts, end, kind, k)]
    for e in evs:
        m = e.get("cat") == "user_annotation" and re.match(r"bx::(exec|sample)#(\d+)$", e.get("name", ""))
        if m:
            ann[e["tid"]].append((e["ts"], e["ts"] + e["dur"], m.group(1), int(m.group(2))))
    launch = {e["args"]["correlation"]: e for e in evs
              if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}

    def owner(corr):
        r = launch.get(corr)
        if r is None:
            return None
        for ts, end, kind, k in ann.get(r["tid"], ()):
            if ts <= r["ts"] <= end:
                return kind, k
        return None

    steps = set(range(start, start + n))
    by_b = collections.Counter()
    by_k = collections.defaultdict(lambda: [0.0, 0, ""])
    sub = collections.Counter()                         # inside sampling_logits: gemm vs rest
    iv, lost, n_k = [], 0.0, 0
    for e in evs:
        cat = e.get("cat")
        if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        o = owner(e.get("args", {}).get("correlation"))
        if o is None or o[1] not in steps:
            lost += e["dur"] if o is None else 0.0
            continue
        name = e.get("name", "")
        b = bucket(name, cat, o[0] == "sample")
        by_b[b] += e["dur"]
        if b == "sampling_logits":
            sub["gemm" if _GEMM.search(name) else "rest"] += e["dur"]
        t = by_k[(b, name)]                 # full name + bucket: Inductor names share long prefixes, and the
        t[0] += e["dur"]                    # same kernel runs in the forward and in compute_logits
        t[1] += 1
        t[2] = b
        iv.append((e["ts"], e["ts"] + e["dur"]))
        n_k += 1
    busy = _union(iv)
    span = (max(x[1] for x in iv) - min(x[0] for x in iv)) if iv else 0.0
    ms = lambda us: round(us / 1e3 / n, 4)   # noqa: E731   us total -> ms per step
    hs = [x for t in ann.values() for x in t if x[3] in steps]
    return {"start": start, "steps": n, "kernels_per_step": round(n_k / n, 1),
            "kernel_ms_per_step": ms(sum(by_b.values())), "busy_ms_per_step": ms(busy),
            "span_ms_per_step": ms(span), "gpu_idle_ms_per_step": ms(span - busy),
            "buckets_ms_per_step": {b: ms(by_b[b]) for b in BUCKETS},
            "sampling_logits_split_ms": {k: ms(v) for k, v in sub.items()},
            "host_exec_ms_per_step": ms(sum(x[1] - x[0] for x in hs if x[2] == "exec")),
            "host_sample_ms_per_step": ms(sum(x[1] - x[0] for x in hs if x[2] == "sample")),
            "unattributed_kernel_ms_total": round(lost / 1e3, 3),
            "top": [{"name": k[1][:110], "bucket": v[2], "ms_per_step": ms(v[0]), "calls_per_step": round(v[1] / n, 2)}
                    for k, v in sorted(by_k.items(), key=lambda kv: -kv[1][0])[:top]]}


def report(d):
    rs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(d, "*.json")))]
    rs = [r for r in rs if "decode" in r]
    out = []
    for phase in ("decode", "prefill"):
        out.append(f"\n## {phase} (ms per step)\n")
        cols = ["wall", "prof_wall", "kernel_sum", "busy", "wall-busy", "prof_idle"] + BUCKETS
        out.append("| run | B | tok/s | " + " | ".join(cols) + " |")
        out.append("|" + "---|" * (len(cols) + 3))
        for r in rs:
            p, t = r[phase], r["timing"]
            wall = t["decode_step_ms_median"] if phase == "decode" else t["prefill_step_ms"]
            v = [wall, p["profiled_wall_ms_per_step"], p["kernel_ms_per_step"], p["busy_ms_per_step"],
                 wall - p["busy_ms_per_step"], p["gpu_idle_ms_per_step"]] + \
                [p["buckets_ms_per_step"][b] for b in BUCKETS]
            out.append(f"| {r['tag']} | {r['batch']} | {r['tok_per_s']} | " + " | ".join(f"{x:.3f}" for x in v)
                       + " |")
    for r in rs:
        for phase in ("decode", "prefill"):
            out.append(f"\n### top kernels: {r['tag']} B={r['batch']} {phase}\n")
            out += [f"- {k['ms_per_step']:.3f} ms x{k['calls_per_step']} [{k['bucket']}] {k['name']}"
                    for k in r[phase]["top"]]
    return "\n".join(out)


if __name__ == "__main__":
    if sys.argv[1] == "--report":
        print(report(sys.argv[2]))
    else:
        print(json.dumps(summarize(sys.argv[1], int(sys.argv[2]), int(sys.argv[3])), indent=1))
