import os, sys, itertools, torch
sys.argv=sys.argv[:1]
d=os.path.dirname(os.path.abspath(__file__))
exec(open(d+"/micro.py").read().split("cases = ")[0])
for R, ctx in [(128, 192), (128, 512), (128, 2048), (32, 1024), (32, 4096), (8, 16384), (1, 16384)]:
    q, qpos, c, bt = mk(R, ctx, 1 << 20); c8 = new.to_limbs(c); out=[]
    for bn, w in [(32,4),(64, 4), (128, 4),(128,8),(256,8)]:
        out.append(f"bn{bn}w{w} {bench(lambda: new.attend(q, qpos, c8, bt, tab, bn=bn, warps=w), 20):.0f}")
    print(R, ctx, *out, flush=True)
