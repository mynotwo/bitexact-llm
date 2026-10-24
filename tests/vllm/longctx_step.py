"""Long-context step time of the integer forward used by the plugin: CUDA-event median of 10 runs
after 3 warm-up runs, cache filled with random in-range
integers (timing only).

  decode:  B rows, one new token each at position S - 1, whole forward + logits, as a CUDA graph
           (graph.decode_forward, what vLLM or the plugin replays)
  prefill: one request, a 512-token chunk at positions S - 512 .. S - 1 (eager plugin path, as vLLM's
           chunked prefill runs it)
  python tests/vllm/longctx_step.py --const C --model-dir DIR --S 16384 --B 1 [--prof]
"""

import argparse
import json
import statistics
import types

import torch


def timeit(fn, warm=3, iters=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--const", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--prof", action="store_true")
    a = ap.parse_args()
    from bitexact.int8.model import IntLlama
    from bitexact.vllm_plugin import paged, tri_paged
    from bitexact.vllm_plugin.graph import decode_forward
    from bitexact.vllm_plugin.model import _extend_rope
    dev = torch.device("cuda")
    c = torch.load(a.const, map_location="cpu", weights_only=False)
    _extend_rope(c, a.model_dir, max(32768, a.S))
    m = IntLlama(c, dev, nl=2, compiled=True)
    m.ptab = tri_paged.weight_table(m.exp)
    del c
    S, B, bs = a.S, a.B, 16
    nblk = S // bs
    nb = B * nblk + 1                                      # block 0 = null block
    g = torch.Generator(device="cuda").manual_seed(0)
    kvs = [torch.randint(-(1 << 20), 1 << 20, (nb, 8, bs, 128), device=dev, dtype=torch.int32, generator=g)
           .view(torch.bfloat16) for _ in range(m.L)]
    bt = (torch.arange(B * nblk, device=dev) + 1).view(B, nblk)
    ids = torch.randint(1000, 9000, (B,), device=dev)
    pos = torch.full((B,), S - 1, device=dev)
    slot = bt[:, -1] * bs + bs - 1
    step = lambda: m.logits(decode_forward(m, kvs, ids, pos, slot, bt))   # noqa: E731
    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(graph):
        step()
    out = {"S": S, "B": B, "decode_ms_per_step": timeit(graph.replay), "decode_eager_ms": timeit(step)}
    T = 512
    p = torch.arange(S - T, S, device=dev)
    meta = types.SimpleNamespace(query_start_loc=torch.tensor([0, T]), seq_lens=torch.tensor([S]),
                                 block_table=bt[:1], slot_mapping=bt[0, -T // bs:].repeat_interleave(bs) * bs
                                 + torch.arange(T, device=dev) % bs)
    x0 = m.emb[torch.randint(1000, 9000, (T,), device=dev)].long()

    def prefill():
        plan = paged.make_plan(meta, T, p)
        x = x0
        for i in range(m.L):
            x = paged.attention(m, i, x, p, kvs[i], plan)
            x = paged.mlp(m, m.layers[i], x)
        return x
    out["prefill512_ms_per_chunk"] = timeit(prefill)
    m.check()
    if a.prof:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            graph.replay()
            torch.cuda.synchronize()
        ev = [(e.device_time_total, e.count, e.key[:70]) for e in prof.key_averages() if e.device_time_total > 0]
        out["decode_kernels"] = sum(c for _, c, _ in ev)
        out["decode_top"] = [f"{t / 1000:.2f}ms x{c} {k}" for t, c, k in sorted(ev, reverse=True)[:8]]
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            prefill()
            torch.cuda.synchronize()
        ev = [(e.device_time_total, e.count, e.key[:70]) for e in prof.key_averages() if e.device_time_total > 0]
        out["prefill_kernels"] = sum(c for _, c, _ in ev)
        out["prefill_top"] = [f"{t / 1000:.2f}ms x{c} {k}" for t, c, k in sorted(ev, reverse=True)[:8]]
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
