"""TP-shard identity: does a sharded integer engine reproduce the unsharded one bit for bit?

  torchrun --nproc_per_node=N tp_run.py --const C --out DIR [--max-new 64] [--prompts 4]

Each rank loads the same constants, shards them, and runs the same prompts. Rank 0 records, per
generated position, the SHA-256 of the full int64 logits row (gathered from every rank), the greedy
token and the canonical log p. The script writes one JSON per world size; `compare_tp.py` diffs them.

Also records a TP=1 run through `bitexact.int8.model.IntLlama` (the shipped engine) so the sharded code
path itself is checked, not only its self-consistency.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tpmodel import TPIntLlama, gather_logits, shard  # noqa: E402

PROMPTS = [
    [128000, 791, 23070, 19044, 6267, 304, 813, 1515, 430, 279, 13766, 1047],
    [128000, 755, 11117, 91907, 1577, 43298, 997, 262, 4304, 52586, 50917, 28090, 14781, 198],
    [128000, 644, 220, 9674, 17, 279, 33715, 8813, 279, 18671, 13962, 11, 1405, 279, 6098, 15916],
    [128000, 29880, 369, 264, 4062, 42120, 19724, 25, 1176, 11, 8798, 1403, 56488, 315, 33213, 5707, 323],
]


def logp(row, tok):
    """Canonical log p of `tok` from one int64 logits row (unit 2^-16), float64 on the CPU."""
    import numpy as np
    x = np.asarray(row.double().cpu().numpy(), dtype=np.float64) * (2.0 ** -16)
    m = x.max()
    return float(x[tok] - (m + np.log(np.exp(x - m).sum())))


@torch.no_grad()
def run_sharded(c, dev, rank, world, prompt, max_new, seed=0):
    m = m_full if m_full is not None else TPIntLlama(shard(c, rank, world), dev, rank, world)
    ids = torch.tensor([prompt], device=dev)
    P = len(prompt)
    cache = m.new_cache(1, P + max_new)
    h = m.forward(ids, torch.arange(P, device=dev)[None], cache)[:, -1]
    toks, hashes, lps = [], [], []
    for t in range(max_new):
        lg = m.logits(h)
        full = gather_logits(lg, world, dev)
        if rank == 0:
            row = full[0]
            tok = int(torch.argmax(row))
            toks.append(tok)
            hashes.append(hashlib.sha256(row.cpu().numpy().tobytes()).hexdigest()[:16])
            lps.append(logp(row, tok))
        else:
            tok = None
        t_t = torch.tensor([-1 if tok is None else tok], device=dev)
        dist.broadcast(t_t, src=0)
        tok = int(t_t.item())
        if t + 1 < max_new:
            h = m.forward(torch.tensor([[tok]], device=dev), torch.tensor([[P + t]], device=dev), cache)[:, -1]
    m.check()
    return {"tok": toks, "sha": hashes, "lp": lps}


@torch.no_grad()
def run_reference(c, dev, prompt, max_new):
    from bitexact.int8.model import IntLlama
    m = IntLlama(c, dev)
    ids = torch.tensor([prompt], device=dev)
    P = len(prompt)
    cache = m.new_cache(1, P + max_new)
    h = m.forward(ids, torch.arange(P, device=dev)[None], cache)[:, -1]
    toks, hashes, lps = [], [], []
    for t in range(max_new):
        lg = m.logits(h)
        row = lg[0]
        tok = int(torch.argmax(row))
        toks.append(tok)
        hashes.append(hashlib.sha256(row.cpu().numpy().tobytes()).hexdigest()[:16])
        lps.append(logp(row, tok))
        if t + 1 < max_new:
            h = m.forward(torch.tensor([[tok]], device=dev), torch.tensor([[P + t]], device=dev), cache)[:, -1]
    m.check()
    return {"tok": toks, "sha": hashes, "lp": lps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--const", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--reference", action="store_true", help="also run the shipped IntLlama at TP 1")
    ap.add_argument("--batch", action="store_true", help="one right-padded batch instead of one prompt at a time")
    ap.add_argument("--sample", action="store_true", help="Gumbel-max at T=1 (the plugin's sampler) instead of greedy")
    a = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = rank % torch.cuda.device_count()
    torch.cuda.set_device(local)
    dev = f"cuda:{local}"
    c = torch.load(a.const, map_location="cpu", weights_only=False)
    global TN, m_full
    if a.sample:
        TN = 1
    os.makedirs(a.out, exist_ok=True)
    if a.batch:
        main_batch(a, c, dev, rank, world)
        dist.barrier()
        dist.destroy_process_group()
        return
    res = {"world": world, "const_sha": c.get("sha256"), "max_new": a.max_new, "seqs": {}}
    for pi, prompt in enumerate(PROMPTS[:a.prompts]):
        global m_full
        m_full = TPIntLlama(shard(c, rank, world), dev, rank, world) if a.sample else None
        res["seqs"][f"p{pi}_tp{world}"] = run_sharded(c, dev, rank, world, prompt, a.max_new, seed=1000 + pi)
        if a.reference and world == 1 and rank == 0:
            res["seqs"][f"p{pi}_ref"] = run_reference(c, dev, prompt, a.max_new)
        dist.barrier()
    if rank == 0:
        with open(os.path.join(a.out, f"tp{world}.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(f"WROTE {a.out}/tp{world}.json seqs {len(res['seqs'])}")
    dist.barrier()
    dist.destroy_process_group()



# ---- batched mode: the same prompts as one right-padded batch (tests sharding + batching together) ----

@torch.no_grad()
def _batch_run(m, rows, max_new, world, dev, rank):
    """rows: list of token-id lists. Right-pad, one prefill, then greedy for max_new steps, all rows live."""
    from bitexact.llama import PAD
    B = len(rows)
    Ps = torch.tensor([len(r) for r in rows], device=dev)
    L = int(Ps.max())
    ids = torch.full((B, L), PAD, dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)] = torch.tensor(r)
    ids, pos = ids.to(dev), torch.arange(L, device=dev).expand(B, L).contiguous()
    cache = m.new_cache(B, L + max_new)
    h = m.forward(ids, pos, cache)[torch.arange(B, device=dev), Ps - 1]
    rec = [{"tok": [], "sha": [], "lp": []} for _ in range(B)]
    for t in range(max_new):
        lg = m.logits(h)
        full = gather_logits(lg, world, dev)
        toks = []
        for j in range(B):
            if rank == 0:
                row = full[j]
                tok = int(torch.argmax(row))
                toks.append(tok)
                rec[j]["tok"].append(tok)
                rec[j]["sha"].append(hashlib.sha256(row.cpu().numpy().tobytes()).hexdigest()[:16])
                rec[j]["lp"].append(logp(row, tok))
            else:
                toks.append(0)
        tt = torch.tensor(toks, device=dev)
        dist.broadcast(tt, src=0)
        if t + 1 < max_new:
            h = m.forward(tt.view(B, 1), (Ps + t).view(B, 1), cache)[:, 0]
    m.check()
    return rec


def main_batch(a, c, dev, rank, world):
    from bitexact.llama import PAD
    m = TPIntLlama(shard(c, rank, world), dev, rank, world)
    rows = PROMPTS[:a.prompts]
    t0 = time.time()
    rec = _batch_run(m, rows, a.max_new, world, dev, rank)
    sec = time.time() - t0
    res = {"world": world, "const_sha": c.get("sha256"), "max_new": a.max_new, "mode": "batch",
           "pp": [len(r) for r in rows], "sec_decode": round(sec, 2), "seqs": {}}
    if rank == 0:
        for pi, r in enumerate(rec):
            res["seqs"][f"p{pi}_tp{world}"] = r
        if a.reference and world == 1:
            from bitexact.int8.model import IntLlama
            ref = IntLlama(c, dev)
            t0 = time.time()
            rr = _batch_run(ref, rows, a.max_new, 1, dev, 0)
            res["sec_decode_ref"] = round(time.time() - t0, 2)
            for pi, r in enumerate(rr):
                res["seqs"][f"p{pi}_ref"] = r
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"batch_tp{world}.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(f"WROTE {a.out}/batch_tp{world}.json sec {sec:.1f}")
    dist.barrier()


if __name__ == "__main__":
    main()
