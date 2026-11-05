"""Profile C_plugin's rescoring workload: 64 GSM8K 2-shot prompts + sampled responses (<=256 tokens), PlugEngine.score.
  prof_score.py --hf .. --const .. --data .. --work .. --out DIR [--reps 5] [--profile]
Rollouts are generated once and cached in DIR/rollouts.pt; score() is timed (wall, with sync) and optionally profiled."""
import argparse, json, os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "..", "..", "abc_rl")
sys.path.insert(0, C)
sys.path.insert(0, os.path.join(C, "..", "rl_mismatch"))
import plug, train  # noqa: E402

ap = argparse.ArgumentParser()
for k in ("hf", "const", "data", "work", "out"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--profile", action="store_true")
ap.add_argument("--budget", type=int, default=8192)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.hf)
tp = train.train_prompts(a.data, tok)
pe = plug.PlugEngine(a.hf, a.const, a.work, 0.2, batched=a.budget)
rp = os.path.join(a.out, "rollouts.pt")
prompts = [tp[i][0] for i in range(a.n // a.G) for _ in range(a.G)]
if os.path.exists(rp):
    resps = torch.load(rp)
else:
    resps, _ = pe.generate(prompts, [1 + j for j in range(len(prompts))], 256)
    torch.save(resps, rp)
keep = [i for i, r in enumerate(resps) if len(r) > 0]
P, R = [prompts[i] for i in keep], [resps[i] for i in keep]
print("n", len(P), "prompt len", sum(map(len, P)) / len(P), "resp len", sum(map(len, R)) / len(R), "total tok", sum(map(len, P)) + sum(map(len, R)))
ts = []
for r in range(a.reps):
    torch.cuda.synchronize(); t0 = time.time()
    s = pe.score(P, R)
    torch.cuda.synchronize(); ts.append(time.time() - t0)
print("SCORE_SEC", json.dumps([round(x, 3) for x in ts]), "median", sorted(ts)[len(ts) // 2])
import hashlib
print("SCORE_HASH", hashlib.sha256(json.dumps(s).encode()).hexdigest()[:16])
if a.profile:
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
        pe.score(P, R)
        torch.cuda.synchronize()
    print(pr.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=70))
    pr.export_chrome_trace(os.path.join(a.out, "trace.json"))
