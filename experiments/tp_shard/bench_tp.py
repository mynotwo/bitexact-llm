"""Decode throughput of the sharded engine, TP 1 / 2 / 4, at batch 1 / 8 / 32.

  torchrun --nproc_per_node=N bench_tp.py --const C --bs 1 8 32 --steps 16

Every rank runs the full loop (all-reduces included); rank 0 times it. Reported as tok/s = B * steps / sec,
plus the per-step wall time. The point is not to beat the unsharded engine: it is to check that exact
integer all-reduces cost a bounded, documented amount, not to establish a best config.
"""
import argparse, json, os, sys, time
import torch, torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tpmodel import TPIntLlama, gather_logits, shard  # noqa: E402


@torch.no_grad()
def bench(c, dev, rank, world, B, steps):
    m = TPIntLlama(shard(c, rank, world), dev, rank, world)
    P = 64
    ids = torch.randint(1000, 120000, (B, P), device=dev)
    pos = torch.arange(P, device=dev).expand(B, P).contiguous()
    cache = m.new_cache(B, P + steps + 1)
    h = m.forward(ids, pos, cache)
    h = h[torch.arange(B, device=dev), P - 1]
    nxt = torch.full((B, 1), 1000, dtype=torch.long, device=dev)
    p2 = torch.full((B, 1), P, dtype=torch.long, device=dev)
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.time()
    for t in range(steps):
        lg = m.logits(h)
        full = gather_logits(lg, world, dev)
        nxt = torch.argmax(full, -1, keepdim=True)
        h = m.forward(nxt, p2 + t, cache)[:, 0]
    torch.cuda.synchronize()
    sec = time.time() - t0
    m.check()
    return {"world": world, "batch": B, "steps": steps, "sec": round(sec, 2),
            "tok_s": round(B * steps / sec, 1), "ms_step": round(1000 * sec / steps, 2)}


@torch.no_grad()
def bench_ref(c, dev, B, steps):
    """The shipped unsharded engine in the identical harness, for a like-for-like ms/step."""
    from bitexact.int8.model import IntLlama
    m = IntLlama(c, dev)
    P = 64
    ids = torch.randint(1000, 120000, (B, P), device=dev)
    pos = torch.arange(P, device=dev).expand(B, P).contiguous()
    cache = m.new_cache(B, P + steps + 1)
    h = m.forward(ids, pos, cache)[torch.arange(B, device=dev), P - 1]
    torch.cuda.synchronize()
    t0 = time.time()
    for t in range(steps):
        lg = m.logits(h)
        nxt = torch.argmax(lg, -1, keepdim=True)
        h = m.forward(nxt, torch.full((B, 1), P + t, dtype=torch.long, device=dev), cache)[:, 0]
    torch.cuda.synchronize()
    sec = time.time() - t0
    m.check()
    return {"world": "ref", "batch": B, "steps": steps, "sec": round(sec, 2),
            "tok_s": round(B * steps / sec, 1), "ms_step": round(1000 * sec / steps, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--const", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--bs", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--ref", action="store_true", help="also time the shipped unsharded engine (TP 1 run)")
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    n = torch.cuda.device_count()
    torch.cuda.set_device(rank % n)
    dev = f"cuda:{rank % n}"
    c = torch.load(a.const, map_location="cpu", weights_only=False)
    rows = []
    for B in a.bs:
        bench(c, dev, rank, world, B, 2)          # warm-up (allocator, kernels)
        r = bench(c, dev, rank, world, B, a.steps)
        rows.append(r)
        if rank == 0:
            print(json.dumps(r))
    if a.ref and world == 1 and rank == 0:
        for B in a.bs:
            bench_ref(c, dev, B, 2)
            r = bench_ref(c, dev, B, a.steps)
            rows.append(r)
            print(json.dumps(r))
    if rank == 0 and a.out:
        json.dump({"device": torch.cuda.get_device_name(dev), "rows": rows}, open(a.out, "w"), indent=1)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
