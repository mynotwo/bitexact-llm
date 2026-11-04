"""Profile one plugin decode step outside vLLM: the plugin's own decode path (vllm_plugin.graph.decode_forward on a
paged int32 KV cache laid out as vLLM's), then compute_logits (m.logits) and the integer sampler (integer_sample),
B rows each one new token at context length ctx (all rows the same length, cache contents random int32).

  python prof_decode.py --const C.pt --model TOK_DIR --B 64 --ctx 2304 --mode eager|compiled --out OUT_PREFIX

eager    : BITEXACT_COMPILE_LAYER=0, segments uncompiled, category ranges (instrument.annotate) -> time by category
compiled : the plugin's settings (segments and pre/post regions under torch.compile) -> step time without and with
           a CUDA graph of decode_forward (as vLLM's FULL_DECODE_ONLY graph: logits and sampler stay outside),
           kernel time by kernel name.
"""
import argparse
import collections
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--const", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=2304)
    ap.add_argument("--mode", choices=["eager", "compiled"], required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["BITEXACT_COMPILE_LAYER"] = "1" if a.mode == "compiled" else "0"
    os.environ["BITEXACT_COMPILE"] = "1" if a.mode == "compiled" else "0"
    import torch
    import instrument
    from bitexact.int8.model import IntLlama
    from bitexact.vllm_plugin import graph, paged, sampler, tri_paged
    from bitexact.vllm_plugin.model import _extend_rope
    from bitexact.llama import HD, HKV

    torch.manual_seed(0)
    c = torch.load(a.const, map_location="cpu", weights_only=False)
    _extend_rope(c, a.model, a.ctx + 8)
    dev = torch.device("cuda")
    m = IntLlama(c, dev, nl=2, compiled=(a.mode == "compiled"))
    m.ptab = tri_paged.weight_table(m.exp)
    B, ctx, bs = a.B, a.ctx, 16
    nblk = (ctx + bs - 1) // bs
    kvs = []
    for _ in range(m.L):
        kv = torch.zeros(1 + B * nblk, HKV, bs, 4 * HD, dtype=torch.bfloat16, device=dev)
        kv.view(torch.int32).copy_(torch.randint(-(1 << 14), 1 << 14, kv.view(torch.int32).shape, device=dev,
                                                 dtype=torch.int32))
        kvs.append(kv)
    bt = (1 + torch.arange(B * nblk, device=dev)).view(B, nblk).long()
    ids = torch.randint(0, 128000, (B,), device=dev)
    pos = torch.full((B,), ctx - 1, dtype=torch.long, device=dev)
    slot = bt[:, (ctx - 1) // bs] * bs + (ctx - 1) % bs
    seeds, ts, temps = list(range(1, B + 1)), [ctx // 2] * B, [(1, 1)] * B

    def fwd():
        return graph.decode_forward(m, kvs, ids, pos, slot, bt)

    def tail(x):
        with instrument.cat("lm_head"):
            li = m.logits(x)
        lf = li.float() * (2.0 ** -16)                          # compute_logits' float copy for vLLM
        with instrument.cat("sampler"):
            y = sampler.integer_sample(li, seeds, ts, temps, m.ln)
        return y, lf

    if a.mode == "eager":
        instrument.annotate(m)
        dsa = paged.decode_store_attend
        paged.decode_store_attend = lambda *x: _cat(instrument, "attn_paged", dsa, *x)
        # decode_forward looks up paged.decode_store_attend at call time
    res = {"args": vars(a), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    with torch.no_grad():
        for _ in range(3):
            tail(fwd())
        torch.cuda.synchronize()

        def timeit(f, n):
            out = []
            for _ in range(n):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                f()
                torch.cuda.synchronize()
                out.append(time.perf_counter() - t0)
            return statistics.median(out) * 1e3
        res["ms_forward_nograph"] = round(timeit(fwd, a.steps), 3)
        res["ms_step_nograph"] = round(timeit(lambda: tail(fwd()), a.steps), 3)
        res["ms_tail"] = round(timeit(lambda: tail(torch.zeros(B, 2048, dtype=torch.long, device=dev)), a.steps), 3)
        g = None
        if a.mode == "compiled":
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    fwd()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                xg = fwd()
            res["ms_forward_graph"] = round(timeit(g.replay, a.steps), 3)
            res["ms_step_graph"] = round(timeit(lambda: (g.replay(), tail(xg)), a.steps), 3)
        print(json.dumps({k: v for k, v in res.items() if k.startswith("ms")}), flush=True)

        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            t0 = time.perf_counter()
            for _ in range(5):
                if g is None:
                    tail(fwd())
                else:
                    g.replay()
                    tail(xg)
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) / 5
    trace = a.out + ".trace.json"
    prof.export_chrome_trace(trace)
    at = instrument.attribute(trace)
    by_class = collections.defaultdict(lambda: [0.0, 0])
    for name, dur, path, op, dims, _ in at["kernels"]:
        top = path.split("/")[0] if path not in ("(none)", "(unattributed)") else path
        k = instrument.kernel_class(name) + (" | lm_head" if top == "lm_head" else " | sampler" if top == "sampler" else "")
        by_class[k][0] += dur / 5
        by_class[k][1] += 1 / 5
    res["profile"] = {"wall_ms_per_step": round(wall * 1e3, 3), "busy_ms_per_step": round(at["busy_us"] / 5e3, 3),
                      "kernels_per_step": at["n_kernels"] / 5,
                      "by_path_us_per_step": {k: [round(v[0] / 5, 1), v[1] / 5] for k, v in
                                              sorted(at["by_path"].items(), key=lambda x: -x[1][0])},
                      "by_kernel_class_us_per_step": {k: [round(v[0], 1), round(v[1], 1)] for k, v in
                                                      sorted(by_class.items(), key=lambda x: -x[1][0])}}
    import prof_scorer
    res["gemm"] = prof_scorer.gemm_table(at["kernels"])
    top = collections.defaultdict(lambda: [0.0, 0])
    for name, dur, *_ in at["kernels"]:
        top[name[:120]][0] += dur / 5
        top[name[:120]][1] += 1
    res["top_kernels"] = [[k, round(v[0], 1), v[1] // 5] for k, v in sorted(top.items(), key=lambda x: -x[1][0])[:30]]
    os.remove(trace)
    if a.mode == "eager":
        bc = instrument.ByteCounter()
        with torch.no_grad(), bc:
            tail(fwd())
        # the fused paged attention kernels read K/V of ctx keys per row (int32 K|V, 2 sweeps over K, 1 over V)
        kv_bytes = m.L * B * ctx * HKV * HD * 4 * 3
        res["bytes"] = {"by_path": dict(bc.bytes.most_common()), "attn_paged_kv_read_est": kv_bytes}
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    print("WROTE", a.out + ".json", flush=True)


def _cat(instrument, name, f, *x):
    with instrument.cat(name):
        return f(*x)


if __name__ == "__main__":
    main()
