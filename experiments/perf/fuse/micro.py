"""Decode-step micro benchmark of the plugin forward (graph.decode_forward under one CUDA graph), no vLLM.

B rows decode --steps tokens from position 0 on a paged cache (teacher-forced random tokens), replaying one
captured graph per step. Prints:
  - hash: sha256 over every step's hidden state and the logits every 16 steps (compare across commits);
  - per-step GPU ms by bucket (gemm = cutlass/int_mm, attention = _max/_acc kernels, elementwise = the rest)
    and kernels per step, from torch.profiler over --prof-steps replays;
  - --top kernels by time per step.

  BITEXACT_INT8_CONST=intllama_const.pt python micro.py --batch 128 [--steps 128] [--out OUT.json]
"""

import argparse
import collections
import hashlib
import json
import os

import torch


def bucket(name):
    if "cutlass" in name or "gemm" in name or "Kernel2" in name:
        return "gemm"
    if name in ("_max_kernel", "_acc_kernel"):
        return "attention"
    return "elementwise"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--prof-steps", type=int, default=16)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--dyn", action="store_true",
                    help="run one forward at batch // 2 first, so torch.compile recompiles with dynamic shapes "
                         "(what happens under vLLM, which runs several batch sizes)")
    ap.add_argument("--out")
    a = ap.parse_args()
    from bitexact.int8.model import IntLlama
    from bitexact.vllm_plugin import tri_paged
    from bitexact.vllm_plugin.graph import decode_forward
    m = IntLlama(torch.load(os.environ["BITEXACT_INT8_CONST"], map_location="cpu", weights_only=False), "cuda",
                 nl=2, compiled=True)
    m.ptab = tri_paged.weight_table(m.exp)
    dev, bs, B = m.dev, 16, a.batch
    nblk = (a.steps + a.prof_steps + 1 + bs - 1) // bs
    kvs = [torch.zeros(B * nblk + 1, 8, bs, 256, dtype=torch.bfloat16, device=dev) for _ in range(m.L)]
    bt = (torch.arange(B * nblk, device=dev) + 1).view(B, nblk)
    g = torch.Generator().manual_seed(0)
    toks = torch.randint(1000, 120000, (a.steps + a.prof_steps + 1, B), generator=g).to(dev)
    tok = torch.zeros(B, dtype=torch.long, device=dev)
    pos = torch.zeros(B, dtype=torch.long, device=dev)
    slot = torch.zeros(B, dtype=torch.long, device=dev)
    rows = torch.arange(B, device=dev)

    def set_step(t):
        tok.copy_(toks[t])
        pos.fill_(t)
        slot.copy_(bt[rows, pos // bs] * bs + pos % bs)

    set_step(0)
    if a.dyn:
        h0 = B // 2
        decode_forward(m, kvs, tok[:h0], pos[:h0], slot[:h0], bt[:h0])
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):                                   # compile + warm-up (writes the step-0 K/V twice)
            decode_forward(m, kvs, tok, pos, slot, bt)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = decode_forward(m, kvs, tok, pos, slot, bt)
    h = hashlib.sha256()
    for t in range(a.steps):
        set_step(t)
        graph.replay()
        h.update(out.cpu().numpy().tobytes())
        if t % 16 == 15:
            h.update(m.logits(out).cpu().numpy().tobytes())
    m.check()
    # wall per step: graph replays only
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    set_step(a.steps)
    torch.cuda.synchronize()
    e0.record()
    for _ in range(20):
        graph.replay()
    e1.record()
    torch.cuda.synchronize()
    wall = e0.elapsed_time(e1) / 20
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(a.prof_steps):
            graph.replay()
        torch.cuda.synchronize()
    tot, cnt = collections.Counter(), collections.Counter()
    per = collections.defaultdict(lambda: [0.0, 0])
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA or e.device_time_total <= 0:
            continue
        b = bucket(e.name)
        tot[b] += e.device_time_total / 1e3
        cnt[b] += 1
        per[e.name][0] += e.device_time_total / 1e3
        per[e.name][1] += 1
    n = a.prof_steps
    res = {"batch": B, "dyn": a.dyn, "hash": h.hexdigest()[:16], "wall_ms": round(wall, 4),
           "ms_per_step": {k: round(v / n, 4) for k, v in tot.items()},
           "kernels_per_step": {k: v / n for k, v in cnt.items()},
           "top": [(round(v[0] / n, 4), v[1] / n, bucket(k), k[:120])
                   for k, v in sorted(per.items(), key=lambda kv: -kv[1][0])[:a.top]]}
    print("RESULT " + json.dumps({k: v for k, v in res.items() if k != "top"}))
    for r in res["top"]:
        print("  %.4f ms x%.1f [%s] %s" % r)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
