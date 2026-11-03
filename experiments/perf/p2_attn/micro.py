"""Microbench + bit-identity of attend() decode path against a reference tree's tri_paged (BASE_SRC) on random caches."""
import os, sys, time, importlib.util, torch
from bitexact.int8.consts import exp_table
from bitexact.intops import LIM
from bitexact.vllm_plugin import tri_paged as new
base = None
if os.environ.get("BASE_SRC"):
    spec = importlib.util.spec_from_file_location("tri_base", os.environ["BASE_SRC"] + "/bitexact/vllm_plugin/tri_paged.py")
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
dev = "cuda"; HKV, HD, bs = 8, 64, 16
lut = exp_table().to(dev); tab = new.weight_table(lut)
def mk(R, ctx, mag, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    nb = R * ((ctx + bs - 1) // bs) + 1
    c = torch.randint(-mag, mag + 1, (nb, HKV, bs, 2 * HD), device=dev, generator=g, dtype=torch.int32)
    q = torch.randint(-mag, mag + 1, (R, HKV, 4, HD), device=dev, generator=g)
    w = (ctx + bs - 1) // bs
    bt = (torch.randperm(nb - 1, device=dev, generator=g)[: R * w].view(R, w) + 1).to(torch.int32)
    qpos = (torch.randint(ctx // 2, ctx, (R, 1), device=dev, generator=g)).expand(R, 4).contiguous()
    return q, qpos, c, bt
def bench(f, n=50):
    for _ in range(3): f()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n): f()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n * 1000
cases = [(128, 192), (128, 256), (32, 192), (8, 16384), (1, 16384)]
for R, ctx in cases:
    q, qpos, c, bt = mk(R, ctx, int(os.environ.get("MAG", 1 << 20)))
    t = bench(lambda: new.attend(q, qpos, c, bt, tab))
    line = f"R={R} ctx={ctx} new {t:.1f} us"
    if base is not None:
        tb = bench(lambda: base.attend(q, qpos, c, bt, tab))
        a, b = new.attend(q, qpos, c, bt, tab), base.attend(q, qpos, c, bt, tab)
        line += f" base {tb:.1f} us  equal {torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])}"
    print(line, flush=True)
