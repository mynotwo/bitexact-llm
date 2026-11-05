"""CUDA-graph timing helper shared by the p6_wgmma bench scripts (serve time is a CUDA-graph replay).

graph_time(f, n) captures f(0) .. f(n-1) into one CUDA graph (so callers can rotate weight copies and keep
the reads HBM-cold, as in serve) and returns ms per call (median of warm replays). Same method as
experiments/perf/p2_gemm/bench_graph.py; a plain per-call event timing at 20-40 us is launch-bound.

timeit(f, M, N, ncp) picks between the two: graph capture keeps every output alive, so an output of M*N*4 bytes
larger than the capture budget (the 128k-row prefill shapes) falls back to per-call events, where the ~17 us
launch cost is under a tenth of a percent anyway.
"""
import statistics

import torch

BUDGET = 6e9                                    # bytes of captured outputs allowed in one graph


def graph_time(f, n=8, warm=3, rep=30):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(n):
            f(i)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n):
            f(i)
    torch.cuda.synchronize()
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / n)
    del g

    return statistics.median(ts)


def event_time(f, warm=5, rep=20):
    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(rep)]
    for e0, e1 in ev:
        e0.record()
        f()
        e1.record()
    torch.cuda.synchronize()
    return statistics.median(e0.elapsed_time(e1) for e0, e1 in ev)


def timeit(f, M, N, ncp=2):
    """ms per call for f(i) -> int32 [M, N]; graph replay when the captured outputs fit, else events."""
    out = M * N * 4
    if out > BUDGET:
        return event_time(lambda: f(0))
    n = max(1, min(4 * ncp, int(BUDGET // out)))
    return graph_time(f, n)