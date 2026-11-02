"""Time per call at M decode rows, before (compiled int8 segments / torch ops, as on 29e5cb5) and after
(this branch's Triton kernels):
  logits  vLLM's float copy of the logits + greedy first argmax, V = 128256 (tri_logits.argmax_float)
  mlp     segs.post + segs.silu_mul after the gate_up GEMM (tri_mlp.silu_quant)
Each variant captured 20 times in one CUDA graph, median of 20 replays.
python experiments/perf/elt2/bench_elt.py"""

import statistics

import torch

from bitexact.int8 import segs
from bitexact.int8.consts import FA, FR, SILU_LIM
from bitexact.intops import first_argmax
from bitexact.llama import D, FF, V
from bitexact.vllm_plugin import tri_logits, tri_mlp


def timeit(fn, reps=20, iters=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(reps):
            fn()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        gr.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts) * 1e3 / reps


xs = torch.arange(-SILU_LIM << FA, (SILU_LIM << FA) + 1, dtype=torch.float64) / (1 << FA)
lut = torch.round(xs / (1.0 + torch.exp(-xs)) * (1 << FR)).long().cuda()
sg = segs.Segs(2, True, True, post_check=False)
for M in (1, 32, 128):
    li = torch.randint(-(1 << 40), 1 << 40, (M, V), device="cuda")
    old = timeit(lambda: (li.float() * (2.0 ** -16), first_argmax(li)))
    new = timeit(lambda: tri_logits.argmax_float(li))
    print(f"logits M={M} old {old:.1f} us  new {new:.1f} us")
    x = torch.randint(-(1 << 30), 1 << 30, (M, D), device="cuda")
    w = torch.randint(-128, 128, (2 * FF, D), device="cuda").to(torch.int8)
    m = torch.randint(-(1 << 20), 1 << 20, (2 * FF,), device="cuda")
    fmsh = torch.tensor(30, device="cuda")
    a, mant, e = sg.pre_plain(x)
    c = torch._int_mm(a, w.t())
    old = timeit(lambda: sg.silu_mul(sg.post(c, m, fmsh, mant, e)[0], lut))
    new = timeit(lambda: tri_mlp.silu_quant(c, m, fmsh, mant, e, lut, False))
    print(f"mlp    M={M} old {old:.1f} us  new {new:.1f} us")
