"""CUDA-graph timing (rotating weight copies -> HBM-cold like decode). Usage: bench_graph.py in.json out.json"""
import sys, json, torch
from bitexact.int8.gemm_skinny import gemm
SH = {"qkv": (3072, 2048), "o": (2048, 2048), "gu": (16384, 2048), "down": (2048, 8192), "lm": (128256, 2048)}
def graph_us(fn, n):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for i in range(n): fn(i)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n): fn(i)
    g.replay(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(20): g.replay()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / 20 / n
if __name__ == '__main__':
    cfgs = json.load(open(sys.argv[1])); out = {}
    for key, v in cfgs.items():
        if key == "ERR": continue
        name, R = key.split(","); R = int(R); N, K = SH[name]
        ncp = max(2, min(16, int(3e9 // (N * K))))   # > L2 footprint
        ws = [torch.randint(-128, 128, (N, K), dtype=torch.int8, device="cuda") for _ in range(ncp)]
        a = torch.randint(-128, 128, (R, K), dtype=torch.int8, device="cuda")
        n = 4 * ncp
        ti = graph_us(lambda i: torch._int_mm(a, ws[i % ncp].t()), n)
        tt = graph_us(lambda i: gemm(a, ws[i % ncp], tuple(v["cfg"])), n)
        assert torch.equal(gemm(a, ws[0], tuple(v["cfg"])), torch._int_mm(a, ws[0].t()))
        roof = N * K / 3.35e12 * 1e6
        out[key] = dict(int_mm=ti, tri=tt, roof=roof, cfg=v["cfg"])
        print(f"{key:8s} int_mm={ti:7.1f}us triton={tt:7.1f}us roof={roof:6.1f}us {v['cfg']}", flush=True)
        del ws
    json.dump(out, open(sys.argv[2], "w"), indent=1)
