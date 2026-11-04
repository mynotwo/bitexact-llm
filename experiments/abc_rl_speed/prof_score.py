"""Profile of the scoring phase of one RL step (no trainer): python prof_score.py --hf --const --data --chunks 64,512 ..."""
import argparse, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "abc_rl"))
sys.path.insert(0, os.path.join(HERE, "..", "rl_mismatch"))
import intq  # noqa: E402
import train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hf"); ap.add_argument("--const"); ap.add_argument("--data")
ap.add_argument("--chunks", default="64")
ap.add_argument("--rows", type=int, default=2048)
ap.add_argument("--lm-rows", type=int, default=512)
ap.add_argument("--logp", default="int", choices=["int", "f64"])
ap.add_argument("--nseq", type=int, default=512)
a = ap.parse_args()
dev = torch.device("cuda")
from safetensors.torch import load_file
from transformers import AutoTokenizer
from bitexact.int8 import scorer as sm
tok = AutoTokenizer.from_pretrained(a.hf)
data = train.train_prompts(a.data, tok)
sd = load_file(os.path.join(a.hf, "model.safetensors"))
P0 = {k: v.float().to(dev) for k, v in sd.items() if k != "lm_head.weight"}
c0, _, _ = intq.quantize(P0)
eng = intq.IntEngine(a.const, c0, dev)
sc = sm.Scorer(eng.m, rows=a.rows, lm_rows=a.lm_rows)
if a.logp == "int":
    sm.logp_rows = lambda lg, tgt: intq.int_logp(lg, tgt, eng.m.ln)
rng = np.random.RandomState(0)
pidx = next(train.batches(len(data), 64, 1))
prompts = [data[i][0] for i in pidx for _ in range(8)][:a.nseq]
# responses ~ real length distribution (mean 214), random tokens
resps = [rng.randint(1000, 100000, size=int(min(256, max(20, rng.normal(214, 50))))).tolist() for _ in prompts]
ntot = sum(map(len, prompts)) + sum(map(len, resps))
print("seqs", len(prompts), "tokens", ntot, "resp", sum(map(len, resps)), flush=True)
tm = {"lm": 0.0, "lp": 0.0}
orig_logits, orig_lp = eng.m.logits, sm.logp_rows
def tl(x):
    torch.cuda.synchronize(); t = time.time(); r = orig_logits(x); torch.cuda.synchronize(); tm["lm"] += time.time() - t; return r
def tp(lg, tgt):
    torch.cuda.synchronize(); t = time.time(); r = orig_lp(lg, tgt); torch.cuda.synchronize(); tm["lp"] += time.time() - t; return r
for ch in map(int, a.chunks.split(",")):
    for rep in range(3):
        sc.c = []; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for k in tm: tm[k] = 0.0
        if rep == 2:
            eng.m.logits, sm.logp_rows = tl, tp
        torch.cuda.synchronize(); t0 = time.time()
        out = train.packed_score(eng, sc, prompts, resps, ch)
        torch.cuda.synchronize(); dt = time.time() - t0
        eng.m.logits, sm.logp_rows = orig_logits, orig_lp
        print(f"chunk {ch} rep {rep} {dt:.2f}s {ntot/dt:.0f} tok/s peak {torch.cuda.max_memory_allocated()/2**30:.1f}G "
              f"reserved {torch.cuda.max_memory_reserved()/2**30:.1f}G" + (f" | lm {tm['lm']:.2f}s logp {tm['lp']:.2f}s" if rep == 2 else ""), flush=True)
