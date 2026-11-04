"""attend() decode time vs nsplit / bn / warps at long context (R requests, ctx keys). usage: sweep.py"""
import sys, torch
sys.path.insert(0, "/data41/audreymao/perf3_20261103/splitkv/repo/experiments/perf/p2_attn")
from bitexact.int8.consts import exp_table
from bitexact.vllm_plugin import tri_paged as T
dev = "cuda"; HKV, HD, bs = 8, 64, 16
tab = T.weight_table(exp_table().to(dev))
def mk(R, ctx, mag=1 << 20):
    g = torch.Generator(device=dev).manual_seed(0)
    nb = R * ((ctx + bs - 1) // bs) + 1
    c = T.to_limbs(torch.randint(-mag, mag + 1, (nb, HKV, bs, 2 * HD), device=dev, generator=g, dtype=torch.int32))
    q = torch.randint(-mag, mag + 1, (R, HKV, 4, HD), device=dev, generator=g)
    w = (ctx + bs - 1) // bs
    bt = (torch.arange(R * w, device=dev).view(R, w) + 1).to(torch.int32)
    qpos = torch.full((R, 4), ctx - 1, device=dev, dtype=torch.long)
    return q, qpos, c, bt
def bench(f, n=30):
    for _ in range(3): f()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): f()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n * 1000
cases = [(1, 4096), (1, 16384), (1, 32768), (2, 16384), (4, 16384), (8, 4096), (8, 16384), (8, 32768), (16, 16384), (32, 16384), (32, 4096), (64, 2048), (128, 192), (128, 1024)]
for R, ctx in cases:
    q, qpos, c, bt = mk(R, ctx)
    ref = T.attend(q, qpos, c, bt, tab, nsplit=1)
    auto = bench(lambda: T.attend(q, qpos, c, bt, tab))
    res = {}
    for ns in (None, 1, 2, 4, 8, 16, 33, 64):
        for bn in (32, 64, 128):
            for w in (2, 4):
                try:
                    out = T.attend(q, qpos, c, bt, tab, nsplit=ns, bn=bn, warps=w)
                    assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
                    res[(ns, bn, w)] = bench(lambda: T.attend(q, qpos, c, bt, tab, nsplit=ns, bn=bn, warps=w), 20)
                except Exception as ex:
                    pass
    b = sorted(res.items(), key=lambda x: x[1])[:4]
    print(f"R={R} ctx={ctx} auto {auto:.0f}us best", [(k, round(v)) for k, v in b], flush=True)
    del c
