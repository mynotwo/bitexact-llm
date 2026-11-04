"""int_logp (compiled) == int_logp_ref (eager) bit for bit: synthetic int64 logits at several shapes / spreads and real
model logits. python test_intlogp.py --hf HF --const CONST --data DATA"""
import argparse, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "abc_rl"))
import intq  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hf"); ap.add_argument("--const"); ap.add_argument("--data")
a = ap.parse_args()
dev = torch.device("cuda")
from safetensors.torch import load_file
from bitexact.int8.consts import FR
sd = load_file(os.path.join(a.hf, "model.safetensors"))
P0 = {k: v.float().to(dev) for k, v in sd.items() if k != "lm_head.weight"}
c0, _, _ = intq.quantize(P0)
eng = intq.IntEngine(a.const, c0, dev)
V = eng.m.lm[0].shape[0] if hasattr(eng.m.lm[0], "shape") else 128256
g = torch.Generator(device=dev).manual_seed(0)
bad = 0
def check(name, lg, tok):
    global bad
    r, n = intq.int_logp_ref(lg, tok, eng.m.ln), intq.int_logp(lg, tok, eng.m.ln)
    ok = r.view(torch.int64).equal(n.view(torch.int64))
    bad += not ok
    print(f"{name:34s} n={lg.shape[0]:5d} V={lg.shape[1]} {'EQUAL' if ok else 'DIFFER'}", flush=True)
for n in (1, 7, 128, 512, 1000, 1024):
    for sc in (2 ** 18, 2 ** 20, 2 ** 22, 2 ** 24, 2 ** 28):          # spread of logits in 2^-16 units
        lg = (torch.randn(n, V, device=dev, generator=g) * sc).long()
        lg[:, 0] += 3 * sc                                            # a clear maximum (peaked row)
        tok = torch.randint(0, V, (n,), device=dev, generator=g)
        check(f"synthetic sd=2^{sc.bit_length() - 1}", lg, tok)
lg = (torch.randn(64, V, device=dev, generator=g) * 2 ** 20).long(); lg[:, 5] = lg.max() + 2 ** 40
check("extreme gap (clamp both ends)", lg, torch.randint(0, V, (64,), device=dev, generator=g))
check("flat rows", torch.zeros(33, V, dtype=torch.long, device=dev), torch.randint(0, V, (33,), device=dev, generator=g))
h = eng.m.emb[torch.randint(1000, 100000, (700,), device=dev, generator=g)].long()
for k in (0, 300, 512):
    lg = eng.m.logits(h[k:k + 512 if k < 300 else k + 188])
    check("real logits", lg, torch.randint(0, V, (lg.shape[0],), device=dev, generator=g))
    check("real logits, argmax target", lg, lg.argmax(-1))
print("ALL EQUAL" if not bad else f"{bad} DIFFER")
sys.exit(bool(bad))
