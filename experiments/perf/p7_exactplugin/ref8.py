"""Standalone exact-engine reference for batch_inv.py (8B): the same 4 prompts, greedy, 48 tokens, hashes as
'seed t sha16 token' lines. python ref8.py --const C --out OUT   then compare with batch_inv.py's OUT.hash."""
import argparse
import hashlib
import json
import random

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--const", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
from bitexact.generate import load  # noqa: E402
from bitexact.intops import first_argmax  # noqa: E402

rng = random.Random(1)
ids = [[rng.randrange(1000, 120000) for _ in range(rng.choice([128, 40, 300, 7]))] for _ in range(128)]
m = load(a.const, "cuda")
lines, toks = [], {}
with torch.no_grad():
    for i in range(4):
        P = len(ids[i])
        cache = m.new_cache(1, P + 48)
        h = m.forward(torch.tensor([ids[i]], device=m.dev), torch.arange(P, device=m.dev)[None], cache)[:, -1]
        toks[i] = []
        for t in range(48):
            li = m.logits(h)
            y = int(first_argmax(li)[0])
            toks[i].append(y)
            lines.append(f"{i} {t} {hashlib.sha256(li[0].cpu().numpy().tobytes()).hexdigest()[:16]} {y}")
            if t + 1 < 48:
                h = m.forward(torch.tensor([[y]], device=m.dev), torch.tensor([[P + t]], device=m.dev), cache)[:, -1]
m.check()
open(a.out + ".hash", "w").write("\n".join(lines) + "\n")
json.dump(toks, open(a.out + ".json", "w"))
print("REF8 done", len(lines))
