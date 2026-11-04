"""Sampling-phase profile of one RL step (no trainer): python prof_sample.py --arm C|D --gpu-mem F [--seqs N] ...
Per engine step: running sequences, scheduled tokens, preemptions, wall time. Same init order / prompts / seeds as train.py."""
import argparse, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "abc_rl"))
sys.path.insert(0, os.path.join(HERE, "..", "rl_mismatch"))
import intq, floatm  # noqa: E402
import train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["C", "D"], required=True)
ap.add_argument("--hf"); ap.add_argument("--const"); ap.add_argument("--data"); ap.add_argument("--work")
ap.add_argument("--gpu-mem", type=float, default=0.20)
ap.add_argument("--seqs", type=int, default=512, help="max_num_seqs")
ap.add_argument("--steps", type=int, default=3)
ap.add_argument("--out", default=None)
ap.add_argument("--nolog", action="store_true", help="ablation: skip int_logp in the sampler hook (logp = 0)")
ap.add_argument("--trace", type=int, default=0, help="torch-profile engine steps 60..60+N of the last step")
ap.add_argument("--hold-gb", type=float, default=0.0, help="allocate this much dummy memory AFTER the sampler (trainer stand-in)")
a = ap.parse_args()
dev = torch.device("cuda")
from safetensors.torch import load_file
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.hf)
data = train.train_prompts(a.data, tok)
sd = load_file(os.path.join(a.hf, "model.safetensors"))
P = {k: v.float().to(dev) for k, v in sd.items() if k != "lm_head.weight"}
c, fw, _ = intq.quantize(P)
mdir = floatm.write_model_dir(a.hf, fw, os.path.join(a.work, "deq_model"))
if a.arm == "C":
    s = train.PluginSampler(a.hf, a.const, a.work, a.gpu_mem, a.seqs)
else:
    s = floatm.VllmSampler(mdir, a.gpu_mem, max_seqs=a.seqs, batch_invariant=True, attn_backend="TRITON_ATTN")
if a.nolog:
    from bitexact.vllm_plugin import sampler as _sm
    _sm.IntSampler._log = lambda self, li, seeds, ts, y: s.rows.append((seeds, ts, torch.zeros(len(ts), dtype=torch.float64, device=dev)))
eng = intq.IntEngine(a.const, c, dev)
if a.arm == "C":
    s.load(c)
else:
    s.load(fw)
hold = torch.empty(int(a.hold_gb * 2**30), dtype=torch.uint8, device=dev) if a.hold_gb else None

from vllm.v1.core.sched.scheduler import Scheduler
R = {"ev": [], "pre": 0}
_sched, _pre = Scheduler.schedule, Scheduler._preempt_request
def schedule(self, *x, **k):
    o = _sched(self, *x, **k)
    torch.cuda.synchronize() if False else None
    n = len(R["ev"])
    if R.get("prof") is not None:
        if n == 60: R["prof"].start()
        if n == 60 + a.trace: R["prof"].stop(); R["done"] = True
    R["ev"].append((time.perf_counter(), len(o.num_scheduled_tokens), o.total_num_scheduled_tokens, len(self.waiting)))
    return o
def pre(self, *x, **k):
    R["pre"] += 1
    return _pre(self, *x, **k)
Scheduler.schedule, Scheduler._preempt_request = schedule, pre

gen = train.batches(len(data), 64, 1)
res = []
for step in range(a.steps):
    pidx = next(gen)
    prompts = [data[i][0] for i in pidx for _ in range(8)]
    seeds = [1 + (1 * 1000 + step) * 10000 + j for j in range(len(prompts))]
    R["ev"].clear(); R["pre"] = 0
    R["prof"] = None
    if a.trace and step == a.steps - 1:
        from torch.profiler import profile, ProfilerActivity
        R["prof"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = s.generate(prompts, seeds, 256)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    ev = R["ev"]
    ts = np.array([e[0] for e in ev] + [t1]); dt = np.diff(ts) * 1e3
    nseq = np.array([e[1] for e in ev]); ntok = np.array([e[2] for e in ev])
    pre_mask = ntok > nseq
    dec = ~pre_mask
    r = {"step": step, "total_s": t1 - t0, "engine_steps": len(ev), "preemptions": R["pre"],
         "prefill_steps": int(pre_mask.sum()), "prefill_s": float(dt[pre_mask].sum() / 1e3),
         "decode_steps": int(dec.sum()), "decode_s": float(dt[dec].sum() / 1e3),
         "decode_mean_batch": float(nseq[dec].mean()), "decode_max_batch": int(nseq.max()),
         "decode_ms_per_step": float(dt[dec].mean()), "n_tok": sum(len(x) for x in out[0]),
         "batch_by_decile": [int(x) for x in np.percentile(nseq[dec], [0, 10, 25, 50, 75, 90, 100])],
         "ms_by_batch": {str(b): float(dt[dec & (nseq >= lo) & (nseq < hi)].mean()) for b, lo, hi in
                         [("<64", 0, 64), ("64-192", 64, 192), ("192-320", 192, 320), (">=320", 320, 9999)] if (dec & (nseq >= lo) & (nseq < hi)).any()}}
    print(json.dumps(r), flush=True); res.append(r)
    if R.get("done"):
        print(R["prof"].key_averages().table(sort_by="cuda_time_total", row_limit=14, max_name_column_width=110), flush=True)
    s.llm.reset_prefix_cache()
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
