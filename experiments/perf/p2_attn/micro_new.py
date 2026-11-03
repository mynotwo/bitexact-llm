import os, sys, itertools, torch
sys.argv=sys.argv[:1]
d=os.path.dirname(os.path.abspath(__file__))
exec(open(d+"/micro.py").read().split("cases = ")[0])
import torch
def ref_attend(c, **kw): return new.attend(*kw["a"][:2], c, *kw["a"][2:])
# compare new attend (limb layout) against the int32 reference from the base tree
for R, ctx in [(128, 192), (128, 256), (32, 192), (8, 16384), (1, 16384)]:
    q, qpos, c, bt = mk(R, ctx, 1 << 20); c8 = new.to_limbs(c)
    t = bench(lambda: new.attend(q, qpos, c8, bt, tab))
    line = f"R={R} ctx={ctx} new {t:.1f} us"
    if base is not None:
        tb = bench(lambda: base.attend(q, qpos, c, bt, tab))
        a, b = new.attend(q, qpos, c8, bt, tab), base.attend(q, qpos, c, bt, tab)
        line += f" base {tb:.1f} us equal {torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])}"
    print(line, flush=True)
# store kernel vs old index_put
nb, bs = 40, 16
for mag in (1 << 20, 8355711):
    g = torch.Generator(device=dev).manual_seed(3)
    k = torch.randint(-mag, mag + 1, (50, HKV, HD), device=dev, generator=g); v = torch.randint(-mag, mag + 1, (50, HKV, HD), device=dev, generator=g)
    slot = torch.randperm(nb * bs, device=dev, generator=g)[:50]; slot[::7] = -1
    c = torch.zeros(nb, HKV, bs, 2 * HD, dtype=torch.int32, device=dev); c8 = torch.zeros(nb, HKV, bs, 8 * HD, dtype=torch.int8, device=dev)
    kp = slot >= 0; s2 = slot[kp]
    c[s2 // bs, :, s2 % bs] = torch.cat([k[kp], v[kp]], -1).to(torch.int32)
    new.store(c8, slot, k, v)
    print("store equal", torch.equal(c8[..., :6 * HD], new.to_limbs(c)[..., :6 * HD]), flush=True)
